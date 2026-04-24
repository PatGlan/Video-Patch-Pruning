import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from typing import Callable

from mmdet.registry import MODELS
from timm.models.vision_transformer import Mlp, named_apply
from custom_mmdet.apis import MetricLogger


@MODELS.register_module()
class SelectiveModule(nn.Module):
    def __init__(self, emb_dim, hidden_dim, drop=0.,
                 tau=1., inference=False,
                 pool_p=False, compute_pruned_layers=False,
                 use_gate2=True, test_sampler=False,
                 *args, **kwargs):
        #kwargs.pop('type')
        super(SelectiveModule, self).__init__()
        #self.layer_idx = layer_idx
        self.inference = inference
        self.norm = nn.LayerNorm(emb_dim)
        #self.norm = nn.LayerNorm(2*emb_dim)
        self.tau = tau
        #MLP
        #self.mlp = Mlp(2*emb_dim, hidden_dim, 2, act_layer=nn.GELU, drop=drop)
        self.mlp = Mlp(emb_dim, hidden_dim, 2, act_layer=nn.GELU, drop=drop)
        self.gate2 = nn.LogSoftmax(dim=-1)

    def forward(self, x,  H, W, is_training=True):
        B, N, C = x.shape

        f = self.norm(x)
        f = self.mlp(f)  # shape (B,L,1) or (B,L,2)
        f = self.gate2(f)  #shape (B,L,1) or (B,L,2)
        selector = F.gumbel_softmax(f*self.tau, tau=self.tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)

        selector_soft = f.softmax(dim=-1)[:, :, 0:1]  # shape (B, L, 1)
        diff_selector = selector
        selector = selector.bool().squeeze(2)


        return selector, diff_selector, selector_soft
'''
class SelectiveModule(nn.Module):
    def __init__(self, channels, rd_channels, hidden_channels, drop=0.,
                 tau=1.,
                 version=0, inference=False, log_patch_size=False,):
        super(SelectiveModule, self).__init__()
        self.inference = inference
        self.norm = nn.LayerNorm(channels)
        self.tau = tau
        self.version = version  # version 0 includes CLS token when selecting, version 1 excludes CLS while selecting
        assert version in (0, 1)
        if self.version == 0 or self.version == 1:
            self.mlp = Mlp(channels, hidden_channels, 2, act_layer=nn.GELU, drop=drop)
        self.gate2 = nn.LogSoftmax(dim=-1)

        self.log_metrics = False

    def enable_logging(self):
        self.log_metrics = True
        self.metric_logger = MetricLogger(delimiter="  ")

    def log_patches(self, mask):
        assert mask.dtype == torch.bool
        num_selec_patches = mask.int().sum().item()
        self.metric_logger.update(num_selec_patches=num_selec_patches)

        num_total_patches = mask.numel()
        self.metric_logger.update(num_total_patches=num_total_patches)

    def forward(self, x):
        B, L, C = x.shape
        if self.version == 1:
            x = x[:, 1:, :]
        x = self.norm(x)
        x = self.mlp(x)  # shape (B,L,1) or (B,L,2)
        scale = self.gate2(x)  #shape (B,L,1) or (B,L,2)
        if not self.inference:
            selector = F.gumbel_softmax(scale, tau=self.tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)
        else:
            selector = torch.argmin(scale, dim=-1, keepdim=True)
        diff_selector = selector
        if self.version == 1:
            selector = torch.cat((torch.ones(B, 1, 1, device=selector.device), selector), dim=1).bool().squeeze(2)
        else:  # self.version = 0
            selector = selector.bool().squeeze(2)

        return selector, diff_selector
'''
