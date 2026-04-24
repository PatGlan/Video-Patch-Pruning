import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from mmdet.registry import MODELS

from timm.models.vision_transformer import Mlp

@MODELS.register_module()
class Mapping_Selective_Module(nn.Module):
    def __init__(self, in_dim, hid_dim, keep_ratio=0.7, tau_matching=10., tau_selector=10., pooling_size=4):
        super().__init__()
        self.quantile = keep_ratio
        self.tau_matching = tau_matching
        self.tau_selector = tau_selector

        self.init_mlp = nn.Linear(in_dim, hid_dim, bias=False)
        self.norm_input = nn.LayerNorm(in_dim)
        self.repl_pad = nn.ReplicationPad2d(pooling_size)
        self.avg_pooler = nn.AvgPool2d(int(2*pooling_size + 1), stride=1)

        self.selector = Mlp(in_features=in_dim, hidden_features=hid_dim, out_features=2, act_layer=nn.GELU)
        self.norm_selector = nn.LayerNorm(in_dim)
        self.gate2 = nn.LogSoftmax(dim=-1)


    def forward(self, x_q, x_k, x_v, mask_selector=None, H=None, W=None):
        B, N, C = x_q.shape

        q = x_q.unsqueeze(1)
        k = x_k.unsqueeze(1)
        v = x_v.unsqueeze(1)
        phi_sim = (q @ k.transpose(-2, -1))

        #todo calculate phi_dist not in inference but in __init__ instead
        w_map = torch.linspace(0, 1, W).repeat(H, 1)
        h_map = torch.linspace(0, 1, H).repeat(W, 1).transpose(0, 1)
        wh_map = torch.stack([w_map, h_map], dim=0).flatten(1).transpose(0, 1)
        manh_dist = torch.cdist(wh_map, wh_map, p=1).to(q.device)
        phi_dist = (2- manh_dist) / 2
        phi_dist = phi_dist.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
        phi_dist = torch.log10(phi_dist * 10 + 1e-6)

        phi = phi_sim + phi_dist

        if mask_selector is not None:
            mask_selector = (~mask_selector).unsqueeze(1).unsqueeze(1)
            masked_phi = torch.zeros_like(mask_selector, dtype=q.dtype)
            masked_phi.masked_fill_(mask_selector, float("-inf"))
            phi = phi + masked_phi

        A = F.gumbel_softmax(phi * self.tau_matching, tau=self.tau_matching, hard=True, dim=-1)
        x = (A @ v).transpose(1, 2).squeeze(2)

        f = self.norm_selector(x)
        f = self.selector(f)  # shape (B,L,2)

        f = self.gate2(f)  # shape (B,L,2)
        f = f.view(B, H, W, 2).permute(0, 3, 1, 2)
        f = self.avg_pooler(self.repl_pad(f))
        f = f.flatten(2).transpose(1, 2)
        selector_soft = F.gumbel_softmax(f * self.tau_selector, tau=self.tau_selector, hard=False)

        selector_soft = selector_soft[:, :, 0:1]  # shape (B, L, 1)
        sorted_selector_soft, index = torch.sort(selector_soft.squeeze(-1), dim=1, descending=True)
        num_p = f.shape[1]
        index = index[:, :int(num_p * self.quantile)].unsqueeze(2)
        y_hard = torch.zeros_like(selector_soft, memory_format=torch.legacy_contiguous_format).scatter_(1, index, 1.0)
        diff_selector = y_hard - selector_soft.detach() + selector_soft

        selector = diff_selector.bool().squeeze(2)
        return selector, diff_selector, selector_soft