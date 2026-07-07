"""
Utilities for MS3-ViT
=====================
Evaluation metrics, visualization, and logging.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.manifold import TSNE


# ============================================================
# Metrics
# ============================================================

def compute_open_set_metrics(pred, labels, num_known):
    """
    Comprehensive open-set classification metrics.

    Args:
        pred: (N,) predicted labels (0..K for unknown)
        labels: (N,) ground truth labels (0..K for unknown)
        num_known: K (number of known classes)

    Returns:
        dict with all metrics
    """
    N = len(labels)

    # Overall Accuracy
    correct = (pred == labels).sum()
    OA = correct / N * 100

    # Per-class accuracy
    per_class_acc = {}
    unique_classes = np.unique(labels)
    for c in unique_classes:
        mask = labels == c
        n_c = mask.sum()
        if n_c > 0:
            per_class_acc[int(c)] = (pred[mask] == c).sum() / n_c * 100

    # Average Accuracy (known classes only)
    known_accs = [per_class_acc.get(c, 0.0) for c in range(num_known)]
    AA = np.mean(known_accs)

    # Kappa coefficient
    Kappa = cohen_kappa_score(labels, pred) * 100

    # Unknown class recall (class K = unknown)
    unknown_mask = labels == num_known
    if unknown_mask.sum() > 0:
        UnknownRecall = (
            (pred[unknown_mask] == num_known).sum() / unknown_mask.sum() * 100
        )
    else:
        UnknownRecall = 0.0

    # Known class accuracy (overall on known samples)
    known_mask = labels < num_known
    if known_mask.sum() > 0:
        KnownOA = (pred[known_mask] == labels[known_mask]).sum() / known_mask.sum() * 100
    else:
        KnownOA = 0.0

    return {
        "OA": OA,
        "AA": AA,
        "Kappa": Kappa,
        "UnknownRecall": UnknownRecall,
        "KnownOA": KnownOA,
        "PerClass": per_class_acc,
        "ConfusionMatrix": confusion_matrix(labels, pred),
    }


# ============================================================
# Visualization
# ============================================================

def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot and optionally save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    # Annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=7,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne(features, labels, class_names, save_path=None,
              title="t-SNE Feature Visualization"):
    """
    t-SNE visualization of learned features.

    Args:
        features: (N, D) feature vectors
        labels: (N,) class labels
        class_names: list of class name strings
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        features_2d[:, 0], features_2d[:, 1],
        c=labels, cmap="tab20", s=5, alpha=0.7,
    )
    legend1 = ax.legend(
        *scatter.legend_elements(),
        loc="lower left", title="Classes",
        bbox_to_anchor=(1.02, 0), fontsize=7,
    )
    ax.add_artist(legend1)
    # Replace legend labels with class names
    for text, name in zip(legend1.get_texts(), class_names):
        text.set_text(name)

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return features_2d


def plot_fusion_weights(weights, scale_names, title="Fusion Weight Distribution"):
    """
    Bar chart of gated fusion weights per scale.

    Args:
        weights: (N, num_scales) or (num_scales,) fusion weights
        scale_names: list of scale names
    """
    if weights.ndim == 2:
        weights = weights.mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(scale_names, weights, color=["#2196F3", "#4CAF50", "#FF9800", "#F44336"])
    ax.set_ylabel("Fusion Weight")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Add value labels on bars
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{w:.3f}", ha="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_classification_map(pred_map, gt_map, save_path=None,
                             title="Classification Map"):
    """
    Plot ground truth vs predicted classification map.

    Args:
        pred_map: (H, W) predicted labels
        gt_map: (H, W) ground truth labels
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(gt_map, cmap="tab20", interpolation="nearest")
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    axes[1].imshow(pred_map, cmap="tab20", interpolation="nearest")
    axes[1].set_title(f"Predicted ({title})")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Logging
# ============================================================

class ExperimentLogger:
    """
    Simple logger for recording and saving experimental results.
    """

    def __init__(self, save_dir="results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.records = []

    def log_epoch(self, epoch, loss_dict, metrics=None):
        """Log one epoch's results."""
        record = {"epoch": epoch, **loss_dict}
        if metrics:
            record.update(metrics)
        self.records.append(record)

    def log_run(self, run_id, final_metrics):
        """Log final results of one run."""
        self.records.append({"run": run_id, **final_metrics})

    def save(self, filename="experiment_log.npy"):
        """Save all records."""
        path = os.path.join(self.save_dir, filename)
        np.save(path, self.records)
        print(f"[Logger] Records saved to {path}")

    def summary_table(self):
        """Generate a summary table string."""
        if not self.records:
            return "No records."

        run_records = [r for r in self.records if "run" in r]
        if not run_records:
            return f"{len(self.records)} epoch records (first: {self.records[0]})"

        oas = [r.get("OA", 0) for r in run_records]
        aas = [r.get("AA", 0) for r in run_records]

        lines = [
            f"Runs: {len(run_records)}",
            f"OA:  {np.mean(oas):.2f}% ± {np.std(oas):.2f}%",
            f"AA:  {np.mean(aas):.2f}% ± {np.std(aas):.2f}%",
        ]
        return "\n".join(lines)
