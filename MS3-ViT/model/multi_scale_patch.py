"""
Module 2: Multi-Scale Patch Embedding
======================================
Parallel multi-scale spatial feature extraction with different
receptive field sizes.

Motivation: Different land-cover classes require different spatial
scales for optimal recognition. Small features (roads, edges) need
fine-grained patches; large features (forests, fields) need broader
context. A single patch size cannot serve all classes well.

Directly inherits the multi-scale parallel design from MSPF-ViT
(Han Gao, EI conference paper).
"""

import torch
import torch.nn as nn


class MultiScalePatchEmbedding(nn.Module):
    """
    Extracts spatial features at multiple scales using parallel
    convolutional branches with different kernel sizes.

    Each branch captures a different spatial granularity:
      - Small kernels (3x3): fine details, edges, small objects
      - Medium kernels (5x5): balanced local context
      - Medium-large kernels (7x7): neighborhood patterns
      - Large kernels (9x9): broader spatial context

    Input:  (B, C_in, S, S)  feature map from SGL
    Output: List of 4 token tensors: [(B, N_s, d), ...]
            where N_s = (S / kernel_s)^2 (approximately)
    """

    def __init__(self, in_channels, embed_dim=128, patch_sizes=None,
                 stride_mode="same"):
        """
        Args:
            in_channels: input channels (from SGL output)
            embed_dim: token embedding dimension d
            patch_sizes: list of kernel sizes, default [3, 5, 7, 9]
            stride_mode: "same" for same-size output, "stride" for full stride
        """
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes or [3, 5, 7, 9]
        self.num_scales = len(self.patch_sizes)

        # ---- Strategy ----
        # For each scale, we use a conv layer with kernel_size=k and the
        # appropriate padding to maintain spatial size, followed by tokenization.
        # This gives same spatial dimensions but different receptive fields.

        self.conv_branches = nn.ModuleList()
        self.token_projections = nn.ModuleList()

        for ks in self.patch_sizes:
            padding = ks // 2  # same padding
            # Feature extraction conv
            conv = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=ks,
                          stride=1, padding=padding, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
            )
            self.conv_branches.append(conv)

            # Token projection: spatial flatten -> linear project
            # (embed_dim * S * S) -> embed_dim (global pooling style)
            # Or: keep spatial tokens: (S*S tokens, each embed_dim)
            # We use the latter for Transformer compatibility
            token_proj = nn.Linear(embed_dim, embed_dim)
            self.token_projections.append(token_proj)

        # Learnable scale embedding to distinguish features from different scales
        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for _ in range(self.num_scales)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C_in, S, S) feature map
        Returns:
            tokens_list: list of (B, N, embed_dim) token tensors
                         N = S * S (number of spatial positions)
        """
        B, C, H, W = x.shape
        N = H * W

        tokens_list = []

        for i, (conv, proj) in enumerate(zip(self.conv_branches,
                                              self.token_projections)):
            # Extract features at this scale
            f = conv(x)  # (B, embed_dim, H, W)

            # Tokenize: (B, embed_dim, H, W) -> (B, N, embed_dim)
            f = f.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

            # Project tokens
            f = proj(f)  # (B, N, embed_dim)

            # Add scale-specific embedding
            f = f + self.scale_embeddings[i]

            tokens_list.append(f)

        return tokens_list


class MultiScalePatchEmbeddingV2(nn.Module):
    """
    Alternative version: uses actual patch partitioning at different sizes.
    This creates different numbers of tokens per scale, allowing the
    alignment layer (in CSGF) to demonstrate its cross-resolution capability.

    Small patch = more tokens = fine-grained
    Large patch = fewer tokens = coarse semantic
    """

    def __init__(self, in_channels, embed_dim=128, patch_sizes=None):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_sizes = patch_sizes or [3, 5, 7, 9]
        self.num_scales = len(self.patch_sizes)

        # For each scale, create a patch embedding with the given patch size
        self.patch_embeds = nn.ModuleList()
        for ps in self.patch_sizes:
            # Use Conv2d with stride=ps to partition into patches
            proj = nn.Conv2d(in_channels, embed_dim,
                             kernel_size=ps, stride=ps, bias=False)
            self.patch_embeds.append(proj)

        # Scale embeddings
        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            for _ in range(self.num_scales)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, C_in, S, S)
        Returns:
            tokens_list: list of (B, N_s, embed_dim) — different N_s per scale
        """
        tokens_list = []
        for i, proj in enumerate(self.patch_embeds):
            # (B, C, S, S) -> (B, embed_dim, H_s, W_s) via strided conv
            f = proj(x)  # H_s = S // ps, W_s = S // ps
            B, D, H_s, W_s = f.shape
            N = H_s * W_s
            # (B, D, H_s, W_s) -> (B, N, D)
            f = f.flatten(2).transpose(1, 2)
            f = f + self.scale_embeddings[i]
            tokens_list.append(f)
        return tokens_list
