"""
Module 5: Class-Aware Open-Set Classification Head
====================================================
Class prototype construction, known class sampler, class-adaptive
deviation threshold (CADT), and prototype contrastive optimization (PCO).

Key improvement over CACL: Class-Adaptive Deviation Threshold.
  - CACL: fixed kappa=2 for all classes
  - Ours: kappa_c adapts based on each class's learning state
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassAwareOpenSetHead(nn.Module):
    """
    Open-set classification head with class-adaptive threshold.

    Components:
        1. Classifier: produces logits for K known classes
        2. Prototypes: per-class feature centroids (from labeled samples)
        3. Known Class Sampler: selects unlabeled samples likely in known classes
        4. CADT: class-adaptive deviation threshold for novel class rejection
        5. PCO: prototype contrastive optimization
    """

    def __init__(self, in_dim, num_classes, prototype_dim=128,
                 temperature=0.1, kappa_base=2.0, gamma_cadt=0.5):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.prototype_dim = prototype_dim
        self.temperature = temperature
        self.kappa_base = kappa_base
        self.gamma_cadt = gamma_cadt

        # Feature projection for prototype space (no LayerNorm!)
        self.prototype_proj = nn.Sequential(
            nn.Linear(in_dim, prototype_dim),
            nn.GELU(),
        )

        # Classifier: produces logits over K known classes
        self.classifier = nn.Linear(prototype_dim, num_classes)

        # ---- Class-Adaptive Threshold parameters ----
        # Learnable base offset for threshold computation
        self.log_offset = nn.Parameter(torch.tensor(0.0))

        # Running statistics for pseudo-label count per class
        self.register_buffer('pseudo_counts', torch.zeros(num_classes))
        self.register_buffer('total_pseudo', torch.tensor(0.0))

    def compute_prototypes(self, features, labels):
        """
        Compute class prototypes from labeled sample features.

        Args:
            features: (B, in_dim) feature vectors
            labels: (B,) class labels (0-indexed)
        Returns:
            prototypes: (num_classes, prototype_dim) class prototypes
        """
        f_proj = self.prototype_proj(features)  # (B, prototype_dim)
        prototypes = torch.zeros(self.num_classes, self.prototype_dim,
                                 device=features.device)

        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                prototypes[c] = f_proj[mask].mean(dim=0)
            else:
                # No labeled samples for this class — use random init
                prototypes[c] = torch.randn(self.prototype_dim,
                                            device=features.device) * 0.01

        return prototypes

    def compute_logits(self, features):
        """
        Compute class logits from features.

        Args:
            features: (B, in_dim)
        Returns:
            logits: (B, num_classes)
            proj_features: (B, prototype_dim)
        """
        proj = self.prototype_proj(features)
        logits = self.classifier(proj)
        return logits, proj

    def compute_distances_to_prototypes(self, proj_features, prototypes):
        """
        Compute cosine distance from each sample to each prototype.
        Cosine distance = 1 - cosine_similarity, range [0, 2].
        More robust to feature magnitude differences than Euclidean.

        Args:
            proj_features: (B, prototype_dim)
            prototypes: (num_classes, prototype_dim)
        Returns:
            distances: (B, num_classes) cosine distances
        """
        # Normalize
        f_norm = F.normalize(proj_features, p=2, dim=1)
        p_norm = F.normalize(prototypes, p=2, dim=1)
        # Cosine similarity: (B, D) @ (D, K) -> (B, K)
        cos_sim = f_norm @ p_norm.T
        # Cosine distance: 1 - cos_sim, range [0, 2]
        distances = 1.0 - cos_sim
        return distances

    def compute_adaptive_thresholds(self, prototypes, labeled_features,
                                     labeled_labels):
        """
        Compute class-adaptive deviation thresholds.

        For each class c:
            mu_c = mean distance of labeled samples to prototype P_c
            sigma_c = std of these distances
            kappa_c = kappa_base * (1 + gamma * (n_c - n_avg) / n_avg)
            eta_c = mu_c + kappa_c * sigma_c

        Where n_c is the number of high-confidence pseudo-labels for class c,
        and n_avg is the average across classes.

        Args:
            prototypes: (K, D)
            labeled_features: (B_L, in_dim) labeled sample features
            labeled_labels: (B_L,) labels
        Returns:
            thresholds: (K,) per-class deviation threshold
        """
        K = self.num_classes
        f_proj = self.prototype_proj(labeled_features)
        distances = self.compute_distances_to_prototypes(f_proj, prototypes)

        thresholds = torch.zeros(K, device=prototypes.device)
        kappa_values = torch.zeros(K, device=prototypes.device)

        # Compute n_avg (average pseudo-label count, smoothed)
        n_avg = self.pseudo_counts.mean()
        if n_avg == 0:
            n_avg = 1.0

        # Compute global statistics for fallback
        all_valid_dists = []
        for c in range(K):
            mask = (labeled_labels == c)
            if mask.sum() > 1:
                all_valid_dists.append(distances[mask, c].mean())
        global_mu = torch.stack(all_valid_dists).mean() if all_valid_dists else torch.tensor(1.0, device=prototypes.device)
        global_sigma = torch.stack(all_valid_dists).std() if len(all_valid_dists) > 1 else torch.tensor(0.1, device=prototypes.device)

        for c in range(K):
            mask = (labeled_labels == c)
            if mask.sum() > 1:
                d_c = distances[mask, c]  # distances to own prototype
                mu_c = d_c.mean()
                sigma_c = d_c.std() + 1e-6

                # Class-adaptive kappa
                n_c = self.pseudo_counts[c]
                kappa_c = self.kappa_base * (
                    1.0 + self.gamma_cadt * (n_c - n_avg) / n_avg
                )
                # Clamp kappa to reasonable range
                kappa_c = torch.clamp(kappa_c, 0.5, 5.0)

                thresholds[c] = mu_c + kappa_c * sigma_c
                kappa_values[c] = kappa_c
            else:
                # Fallback: use global stats with wider margin
                thresholds[c] = global_mu + self.kappa_base * global_sigma
                kappa_values[c] = self.kappa_base

        # Enforce minimum threshold (cosine distance range [0,2], min=0.3)
        min_threshold = torch.tensor(0.3, device=prototypes.device)
        thresholds = torch.maximum(thresholds, min_threshold)

        return thresholds, kappa_values

    def known_class_sampler(self, confidence, proj_features, prototypes,
                             thresholds, conf_threshold=0.9):
        """
        Select unlabeled samples likely belonging to known classes.

        Two conditions must be satisfied:
          1. max(confidence) >= conf_threshold
          2. distance to nearest prototype <= its class threshold

        Args:
            confidence: (B, K) softmax confidence scores
            proj_features: (B, D) projected features
            prototypes: (K, D)
            thresholds: (K,) per-class deviation thresholds
            conf_threshold: confidence threshold gamma
        Returns:
            mask: (B,) boolean mask — True = likely known class
        """
        max_conf, pred_class = confidence.max(dim=1)  # (B,), (B,)

        # Condition 1: confidence above threshold
        conf_mask = max_conf >= conf_threshold

        # Condition 2: distance to predicted class prototype <= threshold
        distances = self.compute_distances_to_prototypes(proj_features, prototypes)
        dist_to_pred = distances[torch.arange(len(pred_class)), pred_class]
        thresh_for_pred = thresholds[pred_class]
        dist_mask = dist_to_pred <= thresh_for_pred

        # Both conditions
        mask = conf_mask & dist_mask

        return mask, pred_class

    def pco_loss(self, features, labels, prototypes):
        """
        Prototype Contrastive Optimization loss.

        Pulls same-class samples closer to their prototype,
        pushes different-class samples away.

        L_pco = -1/(K*B) * sum_c sum_i log(
            exp(-dist(f_i, P_c)^2 / T) / sum_k exp(-dist(f_i, P_k)^2 / T)
        )

        Args:
            features: (B, in_dim)
            labels: (B,) class labels
            prototypes: (K, D)
        Returns:
            loss: scalar
        """
        proj = self.prototype_proj(features)  # (B, D)
        distances = self.compute_distances_to_prototypes(proj, prototypes)  # (B, K)

        # Negative squared distance / temperature
        logits = - (distances ** 2) / self.temperature  # (B, K)

        # Cross-entropy loss with labels
        loss = F.cross_entropy(logits, labels)

        return loss

    def update_pseudo_counts(self, pred_class, mask):
        """
        Update running statistics of pseudo-label counts per class.
        Called at the end of each training epoch.

        Args:
            pred_class: (B,) predicted classes for high-confidence samples
            mask: (B,) mask of selected samples
        """
        if mask.sum() == 0:
            return

        selected_preds = pred_class[mask]
        for c in range(self.num_classes):
            self.pseudo_counts[c] = (
                0.9 * self.pseudo_counts[c] +
                0.1 * (selected_preds == c).sum().float()
            )

    def forward_train(self, features, labels=None, prototypes=None,
                      conf_threshold=0.9):
        """
        Training forward pass: compute logits and optionally PCO loss.

        Returns:
            logits: (B, K)
            proj_features: (B, D)
            pco_loss: scalar (if labels and prototypes provided, else None)
        """
        logits, proj_features = self.compute_logits(features)

        pco_loss = None
        if labels is not None and prototypes is not None:
            pco_loss = self.pco_loss(features, labels, prototypes)

        return logits, proj_features, pco_loss

    @torch.no_grad()
    def forward_test(self, features, prototypes, thresholds,
                     conf_threshold=0.5, dist_margin=2.0):
        """
        Test forward pass with dual-criteria unknown rejection.

        Two criteria (either triggers "unknown"):
          1. Low confidence: max(confidence) < conf_threshold
          2. High distance: distance to predicted prototype > dist_margin * threshold

        Args:
            features: (B, in_dim)
            prototypes: (K, D)
            thresholds: (K,) per-class deviation thresholds
            conf_threshold: minimum confidence for known class
            dist_margin: multiplier on threshold (higher = stricter)
        Returns:
            pred_class: (B,) predicted class (0..K-1 for known, K for unknown)
            confidence: (B,) max confidence score
        """
        logits, proj_features = self.compute_logits(features)
        confidence = F.softmax(logits, dim=1)
        max_conf, pred_class = confidence.max(dim=1)

        # Criterion 1: Low confidence → unknown
        low_conf_mask = max_conf < conf_threshold

        # Criterion 2: Distance exceeds threshold → unknown
        distances = self.compute_distances_to_prototypes(proj_features, prototypes)
        dist_to_pred = distances[torch.arange(len(pred_class)), pred_class]

        dist_mask = torch.zeros(len(pred_class), dtype=torch.bool,
                                 device=features.device)
        for c in range(self.num_classes):
            class_mask = (pred_class == c)
            exceeds = dist_to_pred[class_mask] > dist_margin * thresholds[c]
            dist_mask[class_mask] = exceeds

        # Combine: either criterion marks as unknown
        unknown_mask = low_conf_mask | dist_mask

        pred_class = pred_class.clone()
        pred_class[unknown_mask] = self.num_classes  # K = unknown class

        return pred_class, max_conf
