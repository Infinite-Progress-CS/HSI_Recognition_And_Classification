"""
MS3-ViT Training Script
========================
Open-set semi-supervised training with FixMatch-style paradigm.

Pipeline:
  1. Load HSI data and split known/unknown classes
  2. Initialize MS3-ViT model with physical spectral grouping
  3. Semi-supervised training loop:
     a. Labeled batch: supervised CE loss + PCO loss
     b. Unlabeled batch: consistency regularization (weak vs strong aug)
     c. Update class prototypes and adaptive thresholds
  4. Evaluate on test set (known class accuracy + unknown rejection)
  5. Repeat for 10 independent runs

Usage:
    python train.py --dataset IndianPines --gpu 0
"""

import os
import sys
import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    DATA_DIR, DATASET_CONFIGS, MODEL_CONFIG, TRAIN_CONFIG,
    SPECTRAL_GROUPS,
)
from data_loader import HSIDataManager
from model import MS3ViT
from losses import TotalLoss


# ============================================================
# Utilities (inlined for standalone use)
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(pred, labels, num_known):
    """
    Compute open-set classification metrics.

    Args:
        pred: (N,) predicted labels (0..K for unknown)
        labels: (N,) ground truth labels (0..K for unknown)
        num_known: K (number of known classes)
    Returns:
        dict with OA, AA, Kappa, per-class accuracy
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    N = len(labels)
    correct = (pred == labels).sum().item()
    OA = correct / N * 100

    # Per-class accuracy
    per_class = {}
    all_classes = set(labels.tolist())
    for c in all_classes:
        mask = labels == c
        if mask.sum() > 0:
            acc = (pred[mask] == c).sum().item() / mask.sum() * 100
            per_class[c] = acc

    # AA (average over known classes)
    known_accs = [per_class.get(c, 0) for c in range(num_known)]
    AA = np.mean(known_accs)

    # Kappa
    Kappa = cohen_kappa_score(labels, pred) * 100

    # Unknown class recall (treat unknown as class K)
    unknown_mask = labels == num_known
    if unknown_mask.sum() > 0:
        UnknownRecall = (pred[unknown_mask] == num_known).sum().item() / unknown_mask.sum() * 100
    else:
        UnknownRecall = 0.0

    return {
        "OA": OA,
        "AA": AA,
        "Kappa": Kappa,
        "UnknownRecall": UnknownRecall,
        "PerClass": per_class,
    }


# ============================================================
# Training Function
# ============================================================

def train_one_epoch(model, train_loader, unlabeled_loader, optimizer,
                    criterion, device, epoch, total_epochs,
                    prototypes, thresholds):
    """One epoch of semi-supervised training with known-class filtering."""
    model.train()
    total_loss = total_sup = total_cons = total_pco = 0.0
    n_batches = n_filtered = 0
    unlabeled_iter = iter(unlabeled_loader)

    for x_l, y_l in train_loader:
        try:
            x_u_w, x_u_s = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            x_u_w, x_u_s = next(unlabeled_iter)

        x_l, y_l = x_l.to(device), y_l.to(device)
        x_u_w, x_u_s = x_u_w.to(device), x_u_s.to(device)

        # ---- Labeled forward (with prototypes for PCO) ----
        out_l = model(x_l, labels=y_l, mode="train")
        L_sup = nn.functional.cross_entropy(out_l["logits"], y_l)

        # PCO loss using prototypes
        proj_l = model.head.prototype_proj(out_l["features"])
        from losses import pco_loss
        L_pco = pco_loss(proj_l, y_l, prototypes, temperature=criterion.temperature)

        # ---- Unlabeled: filter with known-class sampler ----
        with torch.no_grad():
            out_u_w = model(x_u_w, mode="train")
            logits_w = out_u_w["logits"]
            proj_w = model.head.prototype_proj(out_u_w["features"])
            # Filter: only use unlabeled samples likely in known classes
            mask, _ = model.head.known_class_sampler(
                nn.functional.softmax(logits_w, dim=1),
                proj_w, prototypes, thresholds,
                conf_threshold=criterion.conf_threshold,
            )
        n_filtered += mask.sum().item()

        # Strong forward (only on filtered samples for efficiency)
        if mask.sum() > 0:
            out_u_s = model(x_u_s, mode="train")

            from losses import consistency_loss
            L_cons = consistency_loss(
                logits_w[mask], out_u_s["logits"][mask],
                confidence_threshold=criterion.conf_threshold,
            )
        else:
            L_cons = torch.tensor(0.0, device=device)

        loss = L_sup + criterion.lambda_u * L_cons + criterion.lambda_p * L_pco
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_sup += L_sup.item()
        total_cons += L_cons.item()
        total_pco += L_pco.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "L_sup": total_sup / n_batches,
        "L_cons": total_cons / n_batches,
        "L_pco": total_pco / n_batches,
        "filtered_ratio": n_filtered / max(n_batches, 1),
    }


@torch.no_grad()
def evaluate(model, test_loader, prototypes, thresholds, device, num_known):
    """
    Evaluate model on test set.

    Returns predictions, labels, and metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in test_loader:
        x = x.to(device)
        pred_class, confidence = model.predict(x, prototypes, thresholds)
        all_preds.append(pred_class.cpu())
        all_labels.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    metrics = compute_metrics(all_preds, all_labels, num_known)
    return all_preds, all_labels, metrics


@torch.no_grad()
def compute_prototypes_and_thresholds(model, train_loader, num_known, device):
    """
    Compute class prototypes and adaptive thresholds from labeled data.

    Returns:
        prototypes: (K, D) class prototypes
        thresholds: (K,) per-class deviation thresholds
    """
    model.eval()
    all_features = []
    all_labels = []

    for x, y in train_loader:
        x = x.to(device)
        F, _, _ = model.extract_features(x)
        all_features.append(F.cpu())
        all_labels.append(y)

    all_features = torch.cat(all_features)  # (N_total, D)
    all_labels = torch.cat(all_labels)      # (N_total,)

    # Compute prototypes
    prototypes = model.head.compute_prototypes(
        all_features.to(device), all_labels.to(device)
    )

    # Compute thresholds
    thresholds, kappa_values = model.head.compute_adaptive_thresholds(
        prototypes,
        all_features.to(device),
        all_labels.to(device),
    )

    return prototypes, thresholds, kappa_values


# ============================================================
# Main Training Loop
# ============================================================

def main(args):
    # ---- Setup ----
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[MS3-ViT] Using device: {device}")
    print(f"[MS3-ViT] Dataset: {args.dataset}")

    ds_config = DATASET_CONFIGS[args.dataset]
    num_classes = ds_config["num_classes"]
    unknown_classes = args.unknown_classes or ds_config["unknown_classes"]
    num_known = num_classes - len(unknown_classes)
    spatial_size = ds_config["spatial_size"]
    num_bands = None  # determined after data load

    print(f"[MS3-ViT] Known classes: {num_known}, Unknown classes: {len(unknown_classes)}")
    print(f"[MS3-ViT] Spatial patch size: {spatial_size}")

    # ---- Data ----
    manager = HSIDataManager(args.dataset, DATA_DIR, TRAIN_CONFIG)
    num_bands = manager.num_bands
    print(f"[MS3-ViT] Data loaded: {manager.height}x{manager.width}, {num_bands} bands")

    # ---- Results storage ----
    all_results = []

    for run in range(args.num_runs):
        seed = args.seed + run
        set_seed(seed)
        print(f"\n{'='*50}")
        print(f"[MS3-ViT] Run {run + 1}/{args.num_runs} (seed={seed})")
        print(f"{'='*50}")

        # Generate splits
        splits = manager.generate_splits(
            unknown_classes,
            num_labeled=TRAIN_CONFIG["num_labeled_per_class"],
            seed=seed,
        )
        train_loader, test_loader, unlabeled_loader = manager.get_dataloaders(splits)

        print(f"[MS3-ViT] Train: {len(splits['train'])} labeled samples")
        print(f"[MS3-ViT] Test: {len(splits['test'])} samples")
        print(f"[MS3-ViT] Unlabeled: {len(splits['unlabeled'])} samples")

        # ---- Model ----
        model = MS3ViT(
            in_channels=num_bands,
            num_classes=num_known,  # only known classes for classifier
            spatial_size=spatial_size,
            config=MODEL_CONFIG,
        )

        # Apply physical spectral grouping
        if args.dataset in SPECTRAL_GROUPS:
            ranges = [(s, e) for _, s, e, _ in SPECTRAL_GROUPS[args.dataset]]
            model.set_physical_groups(ranges)
            print(f"[MS3-ViT] Physical spectral groups applied: {len(ranges)} groups")

        model = model.to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MS3-ViT] Trainable parameters: {n_params:,}")

        # ---- Optimizer & Scheduler ----
        optimizer = optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"],
        )

        if TRAIN_CONFIG["lr_scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=TRAIN_CONFIG["epochs"],
                eta_min=TRAIN_CONFIG["learning_rate"] * 0.01,
            )
        else:
            scheduler = MultiStepLR(
                optimizer,
                milestones=TRAIN_CONFIG["lr_milestones"],
                gamma=TRAIN_CONFIG["lr_gamma"],
            )

        # ---- Loss Criterion ----
        criterion = TotalLoss(
            num_classes=num_known,
            lambda_u=TRAIN_CONFIG["lambda_u"],
            lambda_p=TRAIN_CONFIG["lambda_p"],
            conf_threshold=TRAIN_CONFIG["confidence_threshold"],
            temperature=MODEL_CONFIG["temperature"],
        ).to(device)

        # ---- Pre-compute initial prototypes from labeled data ----
        prototypes, thresholds, kappa_vals = compute_prototypes_and_thresholds(
            model, train_loader, num_known, device,
        )

        # ---- Training Loop ----
        best_oa = 0.0
        epoch_pbar = tqdm(range(1, TRAIN_CONFIG["epochs"] + 1),
                          desc=f"MS3-ViT", ncols=100, colour='blue')

        for epoch in epoch_pbar:
            loss_dict = train_one_epoch(
                model, train_loader, unlabeled_loader,
                optimizer, criterion, device,
                epoch, TRAIN_CONFIG["epochs"],
                prototypes, thresholds,  # <-- pass prototypes for PCO + sampler
            )
            scheduler.step()

            # Update prototypes and thresholds (every epoch for known-class sampler)
            prototypes, thresholds, kappa_vals = compute_prototypes_and_thresholds(
                model, train_loader, num_known, device,
            )

            # Evaluate every 10 epochs
            oa = 0.0
            if epoch % 10 == 0 or epoch == 1 or epoch == TRAIN_CONFIG["epochs"]:
                _, _, metrics = evaluate(
                    model, test_loader, prototypes, thresholds,
                    device, num_known,
                )
                oa = metrics["OA"]
                aa = metrics["AA"]
                kp = metrics["Kappa"]
                ur = metrics["UnknownRecall"]
                if oa > best_oa:
                    best_oa = oa

                tqdm.write(
                    f"  [Eval] Epoch {epoch:3d} | "
                    f"OA={oa:.2f}% | AA={aa:.2f}% | Kappa={kp:.2f} | "
                    f"UnkRecall={ur:.2f}% | BestOA={best_oa:.2f}%"
                )

            # Update progress bar
            epoch_pbar.set_postfix({
                'Loss': f'{loss_dict["loss"]:.4f}',
                'Filt': f'{loss_dict.get("filtered_ratio", 0):.2f}',
                'OA': f'{oa:.2f}%' if oa > 0 else '--',
                'Best': f'{best_oa:.2f}%' if best_oa > 0 else '--',
            })

        epoch_pbar.close()

        # ---- Final Evaluation ----
        prototypes, thresholds, kappa_vals = compute_prototypes_and_thresholds(
            model, train_loader, num_known, device,
        )
        _, _, final_metrics = evaluate(
            model, test_loader, prototypes, thresholds,
            device, num_known,
        )

        print(f"\n[MS3-ViT] Run {run + 1} Final Results:")
        print(f"  OA: {final_metrics['OA']:.2f}%")
        print(f"  AA: {final_metrics['AA']:.2f}%")
        print(f"  Kappa: {final_metrics['Kappa']:.2f}")
        print(f"  Unknown Recall: {final_metrics['UnknownRecall']:.2f}%")

        all_results.append(final_metrics)

    # ---- Summary ----
    oas = [r["OA"] for r in all_results]
    aas = [r["AA"] for r in all_results]
    kappas = [r["Kappa"] for r in all_results]
    ur = [r["UnknownRecall"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"[MS3-ViT] Final Summary ({args.num_runs} runs, {args.dataset})")
    print(f"{'='*60}")
    print(f"  OA:             {np.mean(oas):.2f}% ± {np.std(oas):.2f}%")
    print(f"  AA:             {np.mean(aas):.2f}% ± {np.std(aas):.2f}%")
    print(f"  Kappa:          {np.mean(kappas):.2f} ± {np.std(kappas):.2f}")
    print(f"  Unknown Recall: {np.mean(ur):.2f}% ± {np.std(ur):.2f}%")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MS3-ViT Training")
    parser.add_argument("--dataset", type=str, default="IndianPines",
                        choices=["IndianPines", "PaviaU", "Salinas"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--unknown_classes", type=int, nargs="*",
                        default=None,
                        help="Override default unknown class indices (1-indexed)")
    args = parser.parse_args()
    main(args)
