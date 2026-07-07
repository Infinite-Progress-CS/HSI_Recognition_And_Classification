"""
MS3-ViT: Multi-Scale Spectral-Spatial Vision Transformer
=========================================================
Full model integration for open-set semi-supervised HSI classification.

Architecture:
    Input HSI cube (B, C, S, S)
        │
        ▼
    [1] SpectralGroupingLayer   — 物理光谱分组 + 组内/组间注意力
        │  (B, D, S, S)
        ▼
    [2] MultiScalePatchEmbedding — 4 并行尺度分支 (3×3,5×5,7×7,9×9)
        │  List[(B, N_s, embed_dim)]
        ▼
    [3] MultiScaleViTEncoder    — 每尺度 2 层轻量 ViT
        │  List[(B, embed_dim)]
        ▼
    [4] CrossScaleGatedFusion   — 跨尺度注意力 + 可学习门控融合
        │  (B, embed_dim)
        ▼
    [5] ClassAwareOpenSetHead   — 类别原型 + 已知类采样器 + CADT + PCO
        │
        ▼
    Output: 预测类别 ∈ {0,...,K-1} ∪ {K(unknown)}

Based on: MSPF-ViT (Han Gao, EI) + CACL (Sun et al., TGRS 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_grouping import SpectralGroupingLayer, create_sgl_from_config
from .multi_scale_patch import MultiScalePatchEmbedding
from .per_scale_vit import MultiScaleViTEncoder
from .csgf import CrossScaleGatedFusion
from .open_set_head import ClassAwareOpenSetHead


class MS3ViT(nn.Module):
    """
    Multi-Scale Spectral-Spatial Vision Transformer for
    Open-Set Semi-Supervised HSI Classification.

    Args:
        in_channels: number of spectral bands (C)
        num_classes: number of known classes (K)
        spatial_size: input spatial patch size (S)
        config: model configuration dict
    """

    def __init__(self, in_channels, num_classes, spatial_size=13, config=None):
        super().__init__()
        if config is None:
            from config import MODEL_CONFIG
            config = MODEL_CONFIG

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.spatial_size = spatial_size

        # Extract config
        embed_dim = config["embed_dim"]
        patch_sizes = config["patch_sizes"]
        num_scales = len(patch_sizes)
        vit_depth = config["vit_depth"]
        vit_num_heads = config["vit_num_heads"]
        vit_mlp_ratio = config["vit_mlp_ratio"]
        vit_dropout = config["vit_dropout"]
        sgl_num_groups = config["sgl_num_groups"]
        sgl_group_dim = config["sgl_group_dim"]
        sgl_num_heads = config["sgl_num_heads"]
        fusion_mode = config["fusion_mode"]
        fusion_hidden_dim = config["fusion_hidden_dim"]
        prototype_dim = config["prototype_dim"]
        temperature = config["temperature"]
        kappa_base = config["kappa_base"]
        gamma_cadt = config["gamma_cadt"]

        # ====================
        # Module 1: Spectral Grouping Layer
        # ====================
        self.sgl = SpectralGroupingLayer(
            in_channels=in_channels,
            num_groups=sgl_num_groups,
            group_dim=sgl_group_dim,
            num_heads=sgl_num_heads,
            dropout=vit_dropout,
            out_channels=embed_dim,
        )

        # ====================
        # Module 2: Multi-Scale Patch Embedding
        # ====================
        self.patch_embed = MultiScalePatchEmbedding(
            in_channels=embed_dim,
            embed_dim=embed_dim,
            patch_sizes=patch_sizes,
        )

        # ====================
        # Module 3: Multi-Scale ViT Encoder
        # ====================
        self.vit_encoder = MultiScaleViTEncoder(
            dim=embed_dim,
            num_scales=num_scales,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_ratio=vit_mlp_ratio,
            dropout=vit_dropout,
        )

        # ====================
        # Module 4: Cross-Scale Gated Fusion
        # ====================
        use_cross_attn = (fusion_mode == "cross_attention")
        self.fusion = CrossScaleGatedFusion(
            dim=embed_dim,
            num_scales=num_scales,
            hidden_dim=fusion_hidden_dim,
            use_cross_attention=use_cross_attn,
            num_heads=vit_num_heads,
        )

        # ====================
        # Module 5: Class-Aware Open-Set Head
        # ====================
        self.head = ClassAwareOpenSetHead(
            in_dim=embed_dim,
            num_classes=num_classes,
            prototype_dim=prototype_dim,
            temperature=temperature,
            kappa_base=kappa_base,
            gamma_cadt=gamma_cadt,
        )

        # ====================
        # Feature projection (optional, for feature space alignment)
        # ====================
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
        )

    def set_physical_groups(self, group_ranges):
        """Apply physically-motivated spectral band grouping."""
        self.sgl.set_physical_groups(group_ranges)

    def extract_features(self, x):
        """
        Extract fused multi-scale spectral-spatial features.

        Args:
            x: (B, C, S, S) HSI cube
        Returns:
            F: (B, embed_dim) fused feature vector
            scale_features: list of (B, embed_dim) per-scale features
            fusion_weights: (B, num_scales) fusion weights (for visualization)
        """
        # Module 1: SGL
        x_sgl = self.sgl(x)  # (B, embed_dim, S, S)

        # Module 2: Multi-scale patches
        tokens_list = self.patch_embed(x_sgl)  # List[(B, N_s, embed_dim)]

        # Module 3: Per-scale ViT
        scale_features = self.vit_encoder(tokens_list)  # List[(B, embed_dim)]

        # Module 4: Cross-scale gated fusion
        F, weights = self.fusion(scale_features, return_weights=True)

        # Final projection
        F = self.feature_proj(F)

        return F, scale_features, weights

    def forward(self, x, labels=None, prototypes=None, mode="train"):
        """
        Unified forward pass.

        Args:
            x: (B, C, S, S) HSI cubes
            labels: (B,) class labels (for supervised / PCO loss)
            prototypes: (K, D) class prototypes (from labeled data)
            mode: "train" | "test"
        Returns:
            dict with: logits, features, pco_loss, confidence
        """
        F, scale_features, fusion_weights = self.extract_features(x)

        if mode == "train":
            logits, proj_features, pco_loss = self.head.forward_train(
                F, labels, prototypes,
                conf_threshold=0.9,
            )
            return {
                "logits": logits,
                "features": F,
                "proj_features": proj_features,
                "pco_loss": pco_loss,
                "scale_features": scale_features,
                "fusion_weights": fusion_weights,
            }

        else:  # test
            # During testing, we need prototypes and thresholds
            # These should be passed or computed externally
            raise ValueError(
                "For test mode, use forward_test() with prototypes and thresholds"
            )

    @torch.no_grad()
    def predict(self, x, prototypes, thresholds):
        """
        Test-time prediction: classify as known class or reject as unknown.

        Args:
            x: (B, C, S, S)
            prototypes: (K, D)
            thresholds: (K,)
        Returns:
            pred_class: (B,) predicted classes (K = unknown)
            confidence: (B,) confidence scores
        """
        F, _, _ = self.extract_features(x)
        return self.head.forward_test(F, prototypes, thresholds)
