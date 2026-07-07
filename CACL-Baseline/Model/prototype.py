import torch
from torch import nn
import torch.nn.functional as F

class PAT_with_prototype(nn.Module):
    def __init__(self, backbone, num_classes, kappa_base=2.0, gamma_cadt=0.5):
        super(PAT_with_prototype, self).__init__()
        self.backbone = backbone
        self.feat_planes = self.backbone.feat_planes
        self.prototype = nn.parameter.Parameter(torch.zeros(num_classes, self.feat_planes, dtype=torch.float32), requires_grad=False)
        self.class_dist_max = nn.parameter.Parameter(torch.zeros(num_classes, dtype=torch.float32), requires_grad=False)
        self.class_dist_mean = nn.parameter.Parameter(torch.zeros(num_classes, dtype=torch.float32), requires_grad=False)
        self.class_dist_std = nn.parameter.Parameter(torch.zeros(num_classes, dtype=torch.float32), requires_grad=False)
        self.interclass_dist_mean = nn.parameter.Parameter(torch.zeros(num_classes, dtype=torch.float32), requires_grad=False)
        # CADT: class-adaptive threshold parameters
        self.kappa_base = kappa_base
        self.gamma_cadt = gamma_cadt
        self.register_buffer('pseudo_counts', torch.zeros(num_classes))

    def forward(self, x, **kwargs):
        return self.backbone(x, **kwargs)
    
    def assign_prototype_and_filter_thre(self, prototype, class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean):
        self.prototype.data.copy_(prototype)
        self.class_dist_max.data.copy_(class_dist_max)
        self.class_dist_mean.data.copy_(class_dist_mean)
        self.class_dist_std.data.copy_(class_dist_std)
        self.interclass_dist_mean.data.copy_(interclass_dist_mean)
        
    
    def update_prototype(self, class_center, momentum):
        self.prototype.data.copy_(self.prototype.data * momentum + class_center * (1 - momentum))
        return self.prototype.data

    def update_filter_thre(self, class_dist_max, class_dist_mean, class_dist_std, interclass_dist_mean, momentum):
        self.class_dist_max.data.copy_(self.class_dist_max.data * momentum + class_dist_max * (1 - momentum))
        self.class_dist_mean.copy_(self.class_dist_mean.data * momentum + class_dist_mean * (1 - momentum))
        self.class_dist_std.copy_(self.class_dist_std.data * momentum + class_dist_std * (1 - momentum))
        self.interclass_dist_mean.copy_(self.interclass_dist_mean.data * momentum + interclass_dist_mean * (1 - momentum))
        return self.class_dist_max.data, self.class_dist_mean.data, self.class_dist_std.data, self.interclass_dist_mean.data

    def get_cadt_kappa(self):
        """
        Compute class-adaptive deviation coefficients (CADT).

        kappa_c = kappa_base * (1 + gamma * (n_c - n_avg) / max(n_avg, 1))

        Easy classes (n_c > n_avg): larger kappa -> tighter threshold -> stricter unknown rejection
        Hard classes (n_c < n_avg): smaller kappa -> looser threshold -> avoid false rejection

        Returns:
            kappa_c: (num_classes,) per-class kappa values
        """
        n_avg = self.pseudo_counts.mean()
        if n_avg < 1:
            n_avg = 1.0
        kappa_c = self.kappa_base * (1.0 + self.gamma_cadt * (self.pseudo_counts - n_avg) / n_avg)
        kappa_c = torch.clamp(kappa_c, 0.5, 5.0)
        return kappa_c

    def update_pseudo_counts(self, mask, pseudo_label):
        """
        Update pseudo-label counts for CADT.

        Args:
            mask: (B,) boolean mask of selected unlabeled samples
            pseudo_label: (B,) predicted classes (0..K-1)
        """
        if mask.sum() == 0:
            return
        selected = pseudo_label[mask]
        for c in range(len(self.pseudo_counts)):
            self.pseudo_counts[c] = (
                0.9 * self.pseudo_counts[c] +
                0.1 * (selected == c).sum().float()
            )
