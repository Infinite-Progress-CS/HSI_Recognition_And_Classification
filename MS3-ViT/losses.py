"""
Loss Functions for MS3-ViT
===========================
Three loss components:
  1. L_sup: Supervised cross-entropy on labeled samples
  2. L_cons: Consistency regularization (weak vs strong augmentation)
  3. L_pco: Prototype contrastive optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def supervised_loss(logits, labels, num_classes):
    """
    Standard cross-entropy loss on labeled samples.

    Args:
        logits: (B, K) class logits
        labels: (B,) ground truth labels (0-indexed)
        num_classes: K
    Returns:
        loss: scalar
    """
    if labels.numel() == 0:
        return torch.tensor(0.0, device=logits.device)
    return F.cross_entropy(logits, labels)


def consistency_loss(logits_weak, logits_strong, confidence_threshold=0.9):
    """
    FixMatch-style consistency regularization.

    Weak augmentation generates pseudo-labels (teacher).
    Strong augmentation prediction must match (student).
    Only high-confidence pseudo-labels are used.

    Args:
        logits_weak: (B, K) logits from weakly-augmented inputs
        logits_strong: (B, K) logits from strongly-augmented inputs
        confidence_threshold: gamma
    Returns:
        loss: scalar
    """
    # Pseudo-labels from weak branch
    with torch.no_grad():
        probs_weak = F.softmax(logits_weak, dim=1)
        max_probs, pseudo_labels = probs_weak.max(dim=1)
        mask = max_probs >= confidence_threshold

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits_weak.device)

    # Consistency loss on selected samples
    loss = F.cross_entropy(
        logits_strong[mask],
        pseudo_labels[mask],
    )
    return loss


def pco_loss(proj_features, labels, prototypes, temperature=0.1):
    """
    Prototype Contrastive Optimization loss.

    Pulls same-class samples closer, pushes different-class apart.

    Args:
        proj_features: (B, D) projected features in prototype space
        labels: (B,) class labels
        prototypes: (K, D) class prototypes
        temperature: tau
    Returns:
        loss: scalar
    """
    if labels.numel() == 0:
        return torch.tensor(0.0, device=proj_features.device)

    K = prototypes.shape[0]
    B = proj_features.shape[0]

    # L2 distance squared
    # (B, 1, D) - (1, K, D) -> (B, K, D) -> (B, K)
    diff = proj_features.unsqueeze(1) - prototypes.unsqueeze(0)
    dist_sq = (diff ** 2).sum(dim=2)

    # Convert to logits: negative distance / temperature
    logits = -dist_sq / temperature

    loss = F.cross_entropy(logits, labels)
    return loss


class TotalLoss(nn.Module):
    """
    Combined loss with configurable weights.
    """

    def __init__(self, num_classes, lambda_u=1.0, lambda_p=1.0,
                 conf_threshold=0.9, temperature=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_u = lambda_u
        self.lambda_p = lambda_p
        self.conf_threshold = conf_threshold
        self.temperature = temperature

    def forward(self, outputs, labels=None, prototypes=None,
                outputs_strong=None):
        """
        Compute total loss.

        Args:
            outputs: dict from model forward (labeled batch)
                - "logits": labeled logits
                - "proj_features": projected features
            labels: (B,) ground truth labels
            prototypes: (K, D) class prototypes
            outputs_strong: dict from model forward (strongly-augmented batch)
                - "logits": strong-aug logits
        Returns:
            total_loss: scalar
            loss_dict: {"L_sup": ..., "L_cons": ..., "L_pco": ...}
        """
        loss_dict = {}

        # L_sup: supervised loss
        L_sup = supervised_loss(outputs["logits"], labels, self.num_classes)
        loss_dict["L_sup"] = L_sup

        # L_cons: consistency loss
        if outputs_strong is not None:
            L_cons = consistency_loss(
                outputs["logits"],          # weak
                outputs_strong["logits"],   # strong
                self.conf_threshold,
            )
        else:
            L_cons = torch.tensor(0.0, device=L_sup.device)
        loss_dict["L_cons"] = L_cons

        # L_pco: prototype contrastive loss
        if prototypes is not None and labels is not None:
            L_pco = pco_loss(
                outputs["proj_features"],
                labels,
                prototypes,
                self.temperature,
            )
        else:
            L_pco = torch.tensor(0.0, device=L_sup.device)
        loss_dict["L_pco"] = L_pco

        # Total
        total = L_sup + self.lambda_u * L_cons + self.lambda_p * L_pco
        loss_dict["total"] = total

        return total, loss_dict
