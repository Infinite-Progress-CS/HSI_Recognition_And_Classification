"""
Module 4: Cross-Scale Gated Fusion (CSGF)
==========================================
Adaptive multi-scale feature fusion with learnable gating weights
and optional cross-scale attention enhancement.

This module directly inherits and extends the "Efficient Multi-View
Fusion" mechanism from MSPF-ViT (Han Gao, EI conference paper).

Core mechanism (from MSPF-ViT):
    beta_r = w^T * GAP(U_r)      # branch score
    alpha_r = Softmax(beta_r)     # normalized fusion weight
    F = sum(alpha_r * U_r)        # weighted aggregation

Enhancement for HSI (new):
    - Cross-scale attention: each scale attends to all others
      before gated fusion, enabling richer cross-scale interaction
    - Scale statistics tracking for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFusion(nn.Module):
    """
    Basic gated fusion: learnable weights for each scale branch.

    Exactly matches the MSPF-ViT fusion mechanism.
    """
    def __init__(self, dim, num_scales=4, hidden_dim=64):
        super().__init__()
        self.num_scales = num_scales

        # Gate MLP: dim -> hidden_dim -> 1
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        """
        Args:
            features: list of (B, dim) feature vectors from each scale
        Returns:
            F: (B, dim) fused feature
            weights: (B, num_scales) fusion weights (for interpretability)
        """
        # Stack features: (B, num_scales, dim)
        f_stack = torch.stack(features, dim=1)
        B, S, D = f_stack.shape

        # Compute gate scores
        # Reshape: (B*S, D) -> (B*S, 1) -> (B, S)
        scores = self.gate(f_stack.view(B * S, D)).view(B, S)
        weights = F.softmax(scores, dim=1)  # (B, S)

        # Weighted fusion: (B, S, 1) * (B, S, D) -> (B, S, D) -> sum -> (B, D)
        fused = (weights.unsqueeze(-1) * f_stack).sum(dim=1)

        return fused, weights


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention enhancement: each scale's feature attends
    to features from all other scales before gated fusion.

    This enables richer cross-scale interaction beyond simple weighting.

    For scale i: f_i' = f_i + sum_j (attn(f_i, f_j) * f_j), j != i
    """
    def __init__(self, dim, num_scales=4, num_heads=4):
        super().__init__()
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(dim)

    def forward(self, features):
        """
        Args:
            features: list of (B, dim) — one per scale
        Returns:
            enhanced: list of (B, dim) — cross-scale enhanced features
        """
        # Stack: (B, num_scales, dim)
        f_stack = torch.stack(features, dim=1)
        B, S, D = f_stack.shape

        # Multi-head projection
        q = self.q_proj(f_stack).view(B, S, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # (B, heads, S, head_dim)

        k = self.k_proj(f_stack).view(B, S, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # (B, heads, S, head_dim)

        v = self.v_proj(f_stack).view(B, S, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # (B, heads, S, head_dim)

        # Cross-scale attention: S scales attend to each other
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, S, S)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, heads, S, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)

        # Residual connection + LayerNorm
        enhanced = self.norm(f_stack + out)

        # Split back to list
        return [enhanced[:, i, :] for i in range(S)]


class CrossScaleGatedFusion(nn.Module):
    """
    Complete CSGF module combining:
      1. (Optional) Cross-scale attention enhancement
      2. Gated fusion with learnable weights

    This is the direct evolution of MSPF-ViT's fusion for HSI.
    """

    def __init__(self, dim=128, num_scales=4, hidden_dim=64,
                 use_cross_attention=True, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.use_cross_attention = use_cross_attention

        # Cross-scale attention (optional, enabled by default)
        if use_cross_attention:
            self.cross_attn = CrossScaleAttention(dim, num_scales, num_heads)

        # Gated fusion (always present)
        self.gated_fusion = GatedFusion(dim, num_scales, hidden_dim)

        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
        )

    def forward(self, features, return_weights=False):
        """
        Args:
            features: list of (B, dim) feature vectors from per-scale ViTs
            return_weights: if True, also return fusion weights for analysis
        Returns:
            F: (B, dim) fused feature
            weights: (B, num_scales) fusion weights (optional)
        """
        # Step 1: Cross-scale attention enhancement
        if self.use_cross_attention:
            features = self.cross_attn(features)

        # Step 2: Gated fusion
        fused, weights = self.gated_fusion(features)

        # Step 3: Output projection
        fused = self.out_proj(fused)

        if return_weights:
            return fused, weights
        return fused
