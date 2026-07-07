"""
Module 3: Per-Scale Lightweight ViT Encoder
============================================
Each scale has its own lightweight Transformer encoder (2 layers)
to extract scale-specific spectral-spatial features.

Design: Shallow ViT — 2 layers per scale × 4 scales = 8 total layers.
All scales share LayerNorm parameters for regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-LayerNorm.
    """
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, N, dim)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class PerScaleViT(nn.Module):
    """
    Lightweight ViT encoder for a single scale.

    Args:
        dim: token embedding dimension
        depth: number of Transformer layers (default 2)
        num_heads: attention heads
        mlp_ratio: MLP hidden dim ratio
        dropout: dropout rate
    """
    def __init__(self, dim, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Positional encoding (learnable, shared across scales)
        # Applied per forward call based on token count
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens, pos_embed=None):
        """
        Args:
            tokens: (B, N, dim) token sequence from one scale
            pos_embed: (1, N, dim) optional positional embedding
        Returns:
            feat: (B, dim) pooled feature vector
            encoded_tokens: (B, N, dim) encoded token sequence (optional)
        """
        if pos_embed is not None:
            tokens = tokens + pos_embed

        for block in self.blocks:
            tokens = block(tokens)

        tokens = self.norm_out(tokens)

        # Global average pooling over spatial tokens
        feat = tokens.mean(dim=1)  # (B, dim)

        return feat


class MultiScaleViTEncoder(nn.Module):
    """
    Container for multiple PerScaleViT encoders.
    One encoder per scale, all operating independently.

    Uses shared LayerNorm for cross-scale regularization.
    """

    def __init__(self, dim=128, num_scales=4, depth=2, num_heads=4,
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales
        self.dim = dim

        # Per-scale encoders (independent weights)
        self.encoders = nn.ModuleList([
            PerScaleViT(dim, depth, num_heads, mlp_ratio, dropout)
            for _ in range(num_scales)
        ])

        # Shared LayerNorm across scales (lightweight regularization)
        self.shared_ln = nn.LayerNorm(dim)

        # Per-scale positional embeddings (different lengths for different scales)
        self.pos_embeds = nn.ParameterList([
            # Default size N=64 (for 8x8 spatial grid). Will be interpolated.
            nn.Parameter(torch.randn(1, 64, dim) * 0.02)
            for _ in range(num_scales)
        ])

    def _interpolate_pos_embed(self, pos_embed, N):
        """Interpolate positional embedding to match token count."""
        _, N0, D = pos_embed.shape
        if N == N0:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)  # (1, D, N0)
        pos_embed = F.interpolate(pos_embed, size=N, mode='linear',
                                  align_corners=False)
        return pos_embed.transpose(1, 2)  # (1, N, D)

    def forward(self, tokens_list):
        """
        Args:
            tokens_list: list of (B, N_s, dim) token tensors
        Returns:
            features: list of (B, dim) pooled feature vectors
        """
        features = []
        for i, (tokens, encoder) in enumerate(zip(tokens_list, self.encoders)):
            B, N, D = tokens.shape
            pos = self._interpolate_pos_embed(self.pos_embeds[i], N)
            feat = encoder(tokens, pos)
            feat = self.shared_ln(feat)
            features.append(feat)
        return features
