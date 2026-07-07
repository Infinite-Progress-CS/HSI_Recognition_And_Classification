"""
SGL-PAT: PAT with Spectral Grouping Layer
==========================================
Modified PAT that replaces the original SPL (dimen_redu: 3-layer 1x1 Conv)
with the Spectral Grouping Layer (SGL) for physics-driven spectral processing.

All other PAT components (PAE, Transformer, classifier) remain unchanged.
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from Model.PAT import Residual, PreNorm, FeedForward, Attention, Transformer
from Model.SGL import SpectralGroupingLayer


class SGLPAT(nn.Module):
    """
    PAT with Spectral Grouping Layer replacing the original SPL.

    Args:
        patchsz: spatial patch size (e.g., 7, 9, 11)
        bands: number of spectral bands
        num_classes: number of known classes
        use_pae_embedding: use symmetric PAE (default True)
        dim: feature dimension
        depth: Transformer depth
        heads: attention heads
        mlp_dim: MLP hidden dim in Transformer
        dim_head: per-head dimension
        dropout: dropout rate
        emb_dropout: embedding dropout rate
        # SGL params
        sgl_num_groups: number of spectral groups (default 4)
        sgl_group_dim: dim per spectral group
    """

    def __init__(self, patchsz, bands, num_classes,
                 use_pos_embedding=False, use_pae_embedding=True,
                 dis_type=0, dim=64, depth=5, heads=4, mlp_dim=8,
                 dim_head=16, dropout=0.1, emb_dropout=0.1,
                 sgl_num_groups=4, sgl_group_dim=32):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        self.use_pae_embedding = use_pae_embedding
        self.dis_type = dis_type
        self.dim = dim
        self.patchsz = patchsz

        # ===== REPLACEMENT: SGL instead of dimen_redu =====
        self.sgl = SpectralGroupingLayer(
            in_channels=bands,
            dim=dim,
            num_groups=sgl_num_groups,
            group_dim=sgl_group_dim,
        )

        # Position encoding (unchanged from PAT)
        assert not (use_pae_embedding and use_pos_embedding)
        if use_pos_embedding:
            self.pos_embedding = nn.Parameter(
                torch.randn(1, dim, patchsz, patchsz))
        if use_pae_embedding:
            self.pae_embedding = nn.Parameter(
                torch.randn(dim, patchsz // 2 + 1))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.feat_planes = dim
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1,
                                index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked

    def _add_pae(self, x):
        """Add symmetric PAE to feature map (unchanged from PAT)."""
        par = torch.zeros((self.dim, self.patchsz), device=x.device)
        par[:, self.patchsz // 2:] = self.pae_embedding
        reverse_indice = range(self.patchsz // 2 - 1, -1, -1)
        par[:, reverse_indice] = self.pae_embedding[:, 1:]
        if self.dis_type == 0:
            de = par.unsqueeze(1) + par.unsqueeze(2)
        elif self.dis_type == 1:
            de = torch.sqrt(par.unsqueeze(1) ** 2 + par.unsqueeze(2) ** 2)
        else:
            de = torch.sqrt(par.unsqueeze(1) * par.unsqueeze(2))
        x += de
        return x

    def _forward_common(self, x, with_feat):
        """Shared forward logic."""
        # x: (B, S, S, C) -> (B, C, S, S)
        x = x.permute(0, 3, 1, 2)

        # REPLACED: SGL instead of dimen_redu
        x = self.sgl(x)  # (B, dim, S, S)

        # PAE (unchanged)
        if self.use_pos_embedding:
            x += self.pos_embedding
        if self.use_pae_embedding:
            x = self._add_pae(x)

        x = self.dropout(x)
        # (B, dim, S, S) -> (B, S*S, dim)
        x = x.reshape((x.shape[0], x.shape[1], -1)).transpose(1, 2)

        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)

        if with_feat:
            return x, self.mlp_head(x)
        else:
            return self.mlp_head(x)

    def forward(self, x, with_feat=False):
        return self._forward_common(x, with_feat)

    def forward_mask(self, x, mask_ratio=0.9):
        x = x.permute(0, 3, 1, 2)
        x = self.sgl(x)  # REPLACED
        if self.use_pos_embedding:
            x += self.pos_embedding
        if self.use_pae_embedding:
            x = self._add_pae(x)
        x = self.dropout(x)
        x = x.reshape((x.shape[0], x.shape[1], -1)).transpose(1, 2)
        x = self.random_masking(x, mask_ratio)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.mlp_head(x)
