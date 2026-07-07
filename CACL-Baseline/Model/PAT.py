import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5  # 放缩因子

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # 爱因斯坦求和

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout = dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class PAT(nn.Module):
    def __init__(self, patchsz, bands, num_classes, use_pos_embedding=False, use_pae_embedding=True, dis_type=0, dim=64, depth=5, heads=4, mlp_dim=8,
                dim_head=16, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        self.use_pos_embedding = use_pos_embedding
        self.use_pae_embedding = use_pae_embedding
        self.dis_type = dis_type
        self.dim = dim
        self.patchsz = patchsz
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(bands, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
        )
        assert not (use_pae_embedding and use_pos_embedding)
        if use_pos_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, dim, patchsz, patchsz))
        if use_pae_embedding:
            self.pae_embedding = nn.Parameter(torch.randn(dim, patchsz // 2 + 1))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.feat_planes = dim
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Linear(dim, num_classes)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim  torch.Size([315, 81, 128])
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]  torch.Size([315, 81])
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove torch.Size([315, 81])

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        #torch.Size([315, 16, 128])
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        return x_masked
    
    def forward(self, x, with_feat = False):
        # input torch.Size([128, 7, 7, 200])
        x = x.permute(0, 3, 1, 2)  # torch.Size([128, 200, 7, 7])
        x = self.dimen_redu(x)  # torch.Size([128, 64, 7, 7])
        if self.use_pos_embedding:
            x += self.pos_embedding
        if self.use_pae_embedding:
            par = torch.zeros((self.dim, self.patchsz), device=x.device)
            par[:, self.patchsz // 2:] = self.pae_embedding
            reverse_indice = range(self.patchsz // 2 - 1, -1, -1)
            par[:, reverse_indice] = self.pae_embedding[:, 1:]
            if self.dis_type == 0:  de = par.unsqueeze(1) + par.unsqueeze(2)
            elif self.dis_type == 1:    de = torch.sqrt(par.unsqueeze(1) ** 2 + par.unsqueeze(2) ** 2)
            else:   de = torch.sqrt(par.unsqueeze(1) * par.unsqueeze(2))
            x += de
        x = self.dropout(x)
        x = x.reshape((x.shape[0], x.shape[1], -1)).transpose(1, 2)  # torch.Size([128, 7*7, 64])
            
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)

        if with_feat:
            return x, self.mlp_head(x)
        else:
            return self.mlp_head(x)
    
    def forward_mask(self, x, mask_ratio = 0.9):
        # input torch.Size([128, 7, 7, 200])
        x = x.permute(0, 3, 1, 2)  # torch.Size([128, 200, 7, 7])
        x = self.dimen_redu(x)  # torch.Size([128, 64, 7, 7])
        if self.use_pos_embedding:
            x += self.pos_embedding
        if self.use_pae_embedding:
            par = torch.zeros((self.dim, self.patchsz), device=x.device)
            par[:, self.patchsz // 2:] = self.pae_embedding
            reverse_indice = range(self.patchsz // 2 - 1, -1, -1)
            par[:, reverse_indice] = self.pae_embedding[:, 1:]
            if self.dis_type == 0:  de = par.unsqueeze(1) + par.unsqueeze(2)
            elif self.dis_type == 1:    de = torch.sqrt(par.unsqueeze(1) ** 2 + par.unsqueeze(2) ** 2)
            else:   de = torch.sqrt(par.unsqueeze(1) * par.unsqueeze(2))
            x += de
        x = self.dropout(x)
        x = x.reshape((x.shape[0], x.shape[1], -1)).transpose(1, 2)  # torch.Size([128, 7*7, 64])
        
        x = self.random_masking(x, mask_ratio)
            
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.mlp_head(x)
