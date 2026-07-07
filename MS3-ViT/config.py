"""
MS3-ViT: Multi-Scale Spectral-Spatial Vision Transformer
for Open-Set Semi-Supervised Hyperspectral Image Classification

Configuration file with all hyperparameters.
"""

import os

# ============================================================
# Paths
# ============================================================
DATA_DIR = r"D:\本科生科研\HSI分类\HSI数据集"

DATASET_CONFIGS = {
    "IndianPines": {
        "mat_file": "Indian_pines_corrected.mat",
        "gt_file": "Indian_pines_gt.mat",
        "mat_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "num_classes": 16,
        "unknown_classes": [3, 6],       # classes to treat as unknown
        "spatial_size": 13,               # patch size around center pixel
    },
    "PaviaU": {
        "mat_file": "PaviaU.mat",
        "gt_file": "PaviaU_gt.mat",
        "mat_key": "paviaU",
        "gt_key": "paviaU_gt",
        "num_classes": 9,
        "unknown_classes": [9],
        "spatial_size": 13,
    },
    "Salinas": {
        "mat_file": "Salinas_corrected.mat",
        "gt_file": "Salinas_gt.mat",
        "mat_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "num_classes": 16,
        "unknown_classes": [4, 12, 14],
        "spatial_size": 13,
    },
}

# ============================================================
# Model Architecture
# ============================================================
MODEL_CONFIG = {
    # Spectral Grouping Layer (SGL)
    "sgl_num_groups": 4,        # G: number of spectral groups
    "sgl_group_dim": 32,        # dim per group after projection
    "sgl_num_heads": 4,         # attention heads for cross-group interaction

    # Multi-Scale Patch Embedding
    "patch_sizes": [3, 5, 7, 9],  # 4 scales
    "embed_dim": 128,           # d: token embedding dimension (GPU-safe: 128)

    # Per-Scale ViT Encoder
    "vit_depth": 2,             # Transformer layers per scale
    "vit_num_heads": 4,         # attention heads per ViT
    "vit_mlp_ratio": 2.0,       # MLP expansion ratio (reduced: 4→2)
    "vit_dropout": 0.1,

    # Cross-Scale Gated Fusion (CSGF)
    "fusion_mode": "cross_attention",  # "gated" | "cross_attention"
    "fusion_hidden_dim": 64,    # hidden dim for gate MLP

    # Class-Aware Open-Set Head
    "prototype_dim": 128,       # prototype feature dimension
    "temperature": 0.1,         # tau for PCO contrastive loss
    "kappa_base": 3.0,          # base deviation coefficient (cosine distance)
    "gamma_cadt": 0.5,          # class-adaptive threshold factor
}

# ============================================================
# Training Hyperparameters
# ============================================================
TRAIN_CONFIG = {
    "num_labeled_per_class": 10,   # labeled samples per known class
    "num_runs": 10,                # independent runs for averaging
    "epochs": 300,                 # increased for better convergence
    "batch_size_labeled": 16,      # GPU can handle this
    "batch_size_unlabeled": 16,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "lr_scheduler": "cosine",      # "cosine" | "multistep"
    "lr_milestones": [100, 200],
    "lr_gamma": 0.2,

    # Semi-supervised (FixMatch style)
    "confidence_threshold": 0.9,   # gamma for pseudo-label confidence
    "lambda_u": 1.0,               # weight for unsupervised consistency loss
    "lambda_p": 1.0,               # weight for PCO loss

    # Data augmentation
    "weak_aug": ["flip", "rotate90"],
    "strong_aug": ["flip", "rotate90", "spectral_noise", "spatial_dropout"],
}

# ============================================================
# Spectral Group Definitions (Physical Prior)
# ============================================================
# Maps dataset -> list of (group_name, band_start, band_end, wavelength_range)
# Band indices are 0-based (Python convention)
SPECTRAL_GROUPS = {
    # Indian Pines: AVIRIS sensor, 200 effective bands, 400-2500nm
    # After water absorption band removal, 200 bands remain
    "IndianPines": [
        ("VIS",   0,   40,   "400-700nm (Visible: chlorophyll absorption)"),
        ("VNIR",  40,  100,  "700-1300nm (VNIR: red-edge, cell scattering)"),
        ("SWIR1", 100, 150,  "1300-1900nm (SWIR1: water absorption at 1450nm)"),
        ("SWIR2", 150, 200,  "1900-2500nm (SWIR2: clay minerals, molecular vibration)"),
    ],
    # Pavia University: ROSIS sensor, 103 bands, 430-860nm
    "PaviaU": [
        ("Blue",   0,   25,   "430-540nm (Blue-green: water penetration)"),
        ("Green",  25,  50,   "540-620nm (Green-yellow: vegetation green peak)"),
        ("Red",    50,  75,   "620-720nm (Red: chlorophyll absorption, red-edge)"),
        ("NIR",    75,  103,  "720-860nm (NIR: leaf structure, high reflectance)"),
    ],
    # Salinas: AVIRIS sensor, 204 effective bands, 400-2500nm
    "Salinas": [
        ("VIS",   0,   40,   "400-700nm"),
        ("VNIR",  40,  100,  "700-1300nm"),
        ("SWIR1", 100, 150,  "1300-1900nm"),
        ("SWIR2", 150, 204,  "1900-2500nm"),
    ],
}
