"""
Module 1: Spectral Grouping Layer (SGL) — Memory-Efficient Version
===================================================================
Physically-motivated spectral band grouping.

Memory optimization: processes spectral attention across all spatial
positions using shared weights (conv-style), avoiding B*N expansion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGroupingLayer(nn.Module):
    """
    Memory-efficient SGL using Conv1d for per-pixel spectral processing.

    Input:  (B, C, S, S)  HSI cube
    Output: (B, D_out, S, S) spectrally-processed features

    Architecture:
        1. Split C bands into G physical groups along channel dim
        2. Per-group: Conv1d (per-pixel spectral projection) + GELU
        3. Cross-group: 1x1 Conv mixing + lightweight attention
        4. Output projection
    """

    def __init__(self, in_channels, num_groups=4, group_dim=32,
                 num_heads=4, dropout=0.1, out_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.group_dim = group_dim
        out_channels = out_channels or in_channels

        # Compute group band distribution
        self.group_sizes = self._compute_group_sizes(in_channels, num_groups)
        self.group_indices = self._compute_group_indices(in_channels, self.group_sizes)

        # Per-group: 1x1 Conv for spectral projection (per-pixel, pixel-independent)
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sz, group_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(group_dim),
                nn.GELU(),
            ) for sz in self.group_sizes
        ])

        # Cross-group mixing: 1x1 Conv across concatenated groups
        total_group_dim = num_groups * group_dim
        self.cross_group = nn.Sequential(
            nn.Conv2d(total_group_dim, total_group_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_group_dim // 2),
            nn.GELU(),
            nn.Conv2d(total_group_dim // 2, total_group_dim // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(total_group_dim // 2),
            nn.GELU(),
        )

        # Lightweight spectral channel attention (Squeeze-and-Excitation style)
        # Operates across the cross_group output channels
        mid_channels = total_group_dim // 2
        self.se_fc1 = nn.Linear(mid_channels, mid_channels // 4)
        self.se_fc2 = nn.Linear(mid_channels // 4, mid_channels)

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        # Register group indices
        self.register_buffer('group_starts',
                             torch.tensor([idx[0] for idx in self.group_indices]))
        self.register_buffer('group_ends',
                             torch.tensor([idx[1] for idx in self.group_indices]))

    def _compute_group_sizes(self, C, G):
        base = C // G
        remainder = C % G
        return [base + 1 if i < remainder else base for i in range(G)]

    def _compute_group_indices(self, C, sizes):
        indices = []
        start = 0
        for sz in sizes:
            indices.append((start, start + sz))
            start += sz
        return indices

    def set_physical_groups(self, group_ranges):
        """Override default uniform grouping with physically-motivated groups."""
        self.group_indices = group_ranges
        self.group_sizes = [end - start for start, end in group_ranges]
        # Rebuild group convs
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sz, self.group_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.group_dim),
                nn.GELU(),
            ) for sz in self.group_sizes
        ])
        self.register_buffer('group_starts',
                             torch.tensor([r[0] for r in group_ranges]))
        self.register_buffer('group_ends',
                             torch.tensor([r[1] for r in group_ranges]))

    def forward(self, x):
        """
        Args:
            x: (B, C, S, S) HSI cube
        Returns:
            out: (B, D_out, S, S) processed features
        """
        # Step 1: Split bands into G groups and apply per-group 1x1 conv
        group_feats = []
        for g, (start, end) in enumerate(self.group_indices):
            x_g = x[:, start:end, :, :]  # (B, C_g, S, S)
            f_g = self.group_convs[g](x_g)  # (B, group_dim, S, S)
            group_feats.append(f_g)

        # Step 2: Concatenate group features along channel dim
        f_cat = torch.cat(group_feats, dim=1)  # (B, G*group_dim, S, S)

        # Step 3: Cross-group mixing via 1x1 conv
        f_mixed = self.cross_group(f_cat)  # (B, mid_channels, S, S)

        # Step 4: Spectral channel attention (SE-style, memory-efficient)
        B, C_mid, H, W = f_mixed.shape
        # Global average pooling over spatial dims
        gap = f_mixed.mean(dim=(2, 3))  # (B, C_mid)
        # SE attention
        attn = self.se_fc2(F.relu(self.se_fc1(gap)))  # (B, C_mid)
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)  # (B, C_mid, 1, 1)
        f_attn = f_mixed * attn  # channel-wise attention

        # Step 5: Output projection
        out = self.out_proj(f_attn)  # (B, D_out, S, S)

        return out


def create_sgl_from_config(in_channels, dataset_name, config):
    """Factory function to create SGL with physical prior."""
    num_groups = config.get("sgl_num_groups", 4)
    group_dim = config.get("sgl_group_dim", 32)
    num_heads = config.get("sgl_num_heads", 4)

    sgl = SpectralGroupingLayer(
        in_channels=in_channels,
        num_groups=num_groups,
        group_dim=group_dim,
        num_heads=num_heads,
        dropout=0.1,
    )

    from config import SPECTRAL_GROUPS
    if dataset_name in SPECTRAL_GROUPS:
        ranges = [(s, e) for _, s, e, _ in SPECTRAL_GROUPS[dataset_name]]
        sgl.set_physical_groups(ranges)

    return sgl
