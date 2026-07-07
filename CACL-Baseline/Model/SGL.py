"""
Spectral Grouping Layer (SGL) for CACL
=======================================
Replaces the original SPL (3-layer 1x1 Conv) with physics-driven
spectral band grouping. Groups HSI bands by wavelength regions
(VIS/VNIR/SWIR1/SWIR2), applies intra-group processing and
cross-group mixing.

This is the ONLY structural change to the CACL backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralGroupingLayer(nn.Module):
    """
    Physics-driven spectral band grouping for HSI.

    Input:  (B, C, S, S) HSI cube
    Output: (B, dim, S, S) processed features

    Replaces CACL's dimen_redu (3-layer 1x1 Conv SPL).
    """

    def __init__(self, in_channels, dim=64, num_groups=4, group_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.num_groups = num_groups
        self.group_dim = group_dim

        # Compute group band distribution
        self.group_sizes = self._compute_sizes(in_channels, num_groups)
        self.group_indices = self._compute_indices(in_channels, self.group_sizes)

        # Per-group 1x1 Conv projections (same style as original SPL)
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sz, group_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(group_dim),
                nn.GELU(),
            ) for sz in self.group_sizes
        ])

        # Cross-group mixing
        total = num_groups * group_dim
        self.cross_group = nn.Sequential(
            nn.Conv2d(total, total // 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(total // 2),
            nn.GELU(),
            nn.Conv2d(total // 2, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # SE-style channel attention
        self.se_fc1 = nn.Linear(dim, dim // 4)
        self.se_fc2 = nn.Linear(dim // 4, dim)

    def _compute_sizes(self, C, G):
        base = C // G
        rem = C % G
        return [base + 1 if i < rem else base for i in range(G)]

    def _compute_indices(self, C, sizes):
        idx = []
        s = 0
        for sz in sizes:
            idx.append((s, s + sz))
            s += sz
        return idx

    def forward(self, x):
        # x: (B, C, S, S)
        # Split into groups and process
        group_feats = []
        for g, (start, end) in enumerate(self.group_indices):
            x_g = x[:, start:end, :, :]  # (B, C_g, S, S)
            f_g = self.group_convs[g](x_g)  # (B, group_dim, S, S)
            group_feats.append(f_g)

        # Concatenate and cross-group mixing
        f_cat = torch.cat(group_feats, dim=1)  # (B, G*group_dim, S, S)
        f_mixed = self.cross_group(f_cat)  # (B, dim, S, S)

        # Channel attention
        B, C_out, H, W = f_mixed.shape
        gap = f_mixed.mean(dim=(2, 3))  # (B, C_out)
        attn = self.se_fc2(F.relu(self.se_fc1(gap)))
        attn = torch.sigmoid(attn).unsqueeze(-1).unsqueeze(-1)
        f_out = f_mixed * attn

        return f_out
