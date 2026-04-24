# Copyright (c) Shanghai AI Lab. All rights reserved.
import logging
import math
import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from setuptools import find_namespace_packages
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# from mmdet.models.builder import BACKBONES
from mmdet.registry import MODELS
# from ops.modules import MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_

#from .vpp.ms_deform_attn import MSDeformAttn
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
# from projects.VPP.models.modules import SelectiveVisionTransformer
from .selective_vit import SelectiveVisionTransformer
from .adapter_modules import (SpatialPriorModule, InteractionBlockWithSelection,
                                  InteractionBlockWithInitialSelection,
                                  deform_inputs)
from .selective_module import SelectiveModule


_logger = logging.getLogger(__name__)


@MODELS.register_module()
class SelectiveViTAdapter(SelectiveVisionTransformer):
    def __init__(self, pretrain_size=224, num_heads=12, conv_inplane=64, n_points=4, deform_num_heads=6,
                 init_values=0., interaction_indexes=None, with_cffn=True, cffn_ratio=0.25,
                 deform_ratio=1.0, add_vit_feature=True, use_extra_extractor=True,
                 num_classes=80,
                 ratio_sm_sparsity_loss=1.0, ratio_match_module_sparsity_loss=1.0,
                 pruning_method=None, selec_skip_ids=[],
                 selective_module=None, select_module_ids=[3, 6, 9], selective_module_kr=[0.56, 0.448, 0.358],
                 matching_module=None, matching_module_idx=None,
                 prev_x_update_layer=5,
                 *args, **kwargs):

        kwargs.pop('window_attn')
        kwargs.pop('window_size')
        assert kwargs.pop('layer_scale') is False, 'Pytorch Better Transformer does not support layer scale'
        assert kwargs.pop('pretrained') is None, 'use load_from instead of pretrained'
        super().__init__(num_heads=num_heads, *args, **kwargs)

        self.norm = None
        self.head = None
        self.num_classes = num_classes
        self.cls_token = None

        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane,
                                      embed_dim=embed_dim)

        self.pruning_method = pruning_method
        if self.pruning_method == "ivpp":
            self.ratio_sm_sparsity_loss = ratio_sm_sparsity_loss
            self.selective_module_kr = selective_module_kr

            self.num_heads = num_heads
            self.selective_modules = nn.ModuleList([])
            for i in range(len(set(select_module_ids))):
                self.selective_modules.append(MODELS.build(selective_module))

            self.interactions = nn.Sequential(*[
                InteractionBlockWithInitialSelection(dim=embed_dim, num_heads=deform_num_heads, n_points=n_points,
                                                     init_values=init_values, drop_path=self.drop_path_rate,
                                                     norm_layer=self.norm_layer, with_cffn=with_cffn,
                                                     cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                                                     extra_extractor=((True if i == len(
                                                         interaction_indexes) - 1 else False) and use_extra_extractor),
                                                     selec_skip_ids=selec_skip_ids,
                                                     select_module_ids=select_module_ids,
                                                     matching_module_idx=matching_module_idx,
                                                     prev_x_update_layer=prev_x_update_layer,
                                                     ratio_match_module_sparsity_loss=ratio_match_module_sparsity_loss,
                                                     interact_index=interaction_indexes[i],
                                                     matching_module=matching_module,
                                                     )
                for i in range(len(interaction_indexes))
            ])
            self.prev_x = None

        else:
            raise ValueError(f'Unknown pruning method {self.pruning_method}')


        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // 16, self.pretrain_size[1] // 16, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False). \
            reshape(1, -1, H * W).permute(0, 2, 1)
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m.init_weights()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x, need_loss=False, video_emb=None, prev_x=None, selec_init_p=None, selec_gt_p=None,
                loss_name_prefix="",
                x_dense=None, ret_interm_results=False, compute_dense=False):

        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # prepare selection mask
        if self.pruning_method == 'ivpp' and prev_x is None:
            prev_x = {'x_ref': None, 'x_to_map': None, 'selector': None}
            if not need_loss and self.prev_x is not None:  # inference
                prev_x = self.prev_x

        layer_loss = 0.
        num_loss = 0
        curr_selector=None
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c, layer_ratio_loss, has_loss, prev_x, curr_selector = layer(x, c, indexes,
                                          deform_inputs1, deform_inputs2, H, W,
                                          self.blocks, self.selective_modules, prev_x,
                                          self.selective_module_kr, curr_selector,
                                          )
            layer_loss = layer_loss + layer_ratio_loss
            num_loss = num_loss + has_loss


        if self.pruning_method == 'ivpp' and not need_loss:
            self.prev_x = prev_x # in inference, save prev features as for next frame

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        # calc ratio loss
        if need_loss:
            losses = {}
            if self.pruning_method == 'ivpp':
                if not compute_dense and num_loss > 0:
                    layer_loss = layer_loss / num_loss
                    layer_loss = self.ratio_sm_sparsity_loss * layer_loss
                    losses.update({f'backbone.{loss_name_prefix}sparse_loss': layer_loss})
                prev_x
            else:
                assert False, f'Unknown pruning method {self.pruning_method}'

            return [f1, f2, f3, f4], prev_x, losses
        else:
            return [f1, f2, f3, f4]
