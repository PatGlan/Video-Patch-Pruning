import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
import random
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn
from .map_selective_module import Mapping_Selective_Module
from mmdet.registry import MODELS

_logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes, device):
    reference_points_list = []
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]
    return reference_points


def deform_inputs(x):
    bs, c, h, w = x.shape
    spatial_shapes = torch.as_tensor([(h // 8, w // 8),
                                      (h // 16, w // 16),
                                      (h // 32, w // 32)],
                                     dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 16, w // 16)], x.device)
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor([(h // 16, w // 16)], dtype=torch.long, device=x.device)
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    reference_points = get_reference_points([(h // 8, w // 8),
                                             (h // 16, w // 16),
                                             (h // 32, w // 32)], x.device)
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        n = N // 21
        x1 = x[:, 0:16 * n, :].transpose(1, 2).view(B, C, H * 2, W * 2).contiguous()
        x2 = x[:, 16 * n:20 * n, :].transpose(1, 2).view(B, C, H, W).contiguous()
        x3 = x[:, 20 * n:, :].transpose(1, 2).view(B, C, H // 2, W // 2).contiguous()
        x1 = self.dwconv(x1).flatten(2).transpose(1, 2)
        x2 = self.dwconv(x2).flatten(2).transpose(1, 2)
        x3 = self.dwconv(x3).flatten(2).transpose(1, 2)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class Extractor(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 with_cffn=True, cffn_ratio=0.25, drop=0., drop_path=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), with_cp=False):
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        #self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        self.attn = MSDeformAttn(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index, H, W):

        def _inner_forward(query, feat):

            #attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None)
            attn = self.attn(query=self.query_norm(query), reference_points=reference_points, value=self.feat_norm(feat), spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=None)
            query = query + attn

            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query


class Injector(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, n_levels=1, deform_ratio=1.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0., with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        #self.attn = MSDeformAttn(d_model=dim, n_levels=n_levels, n_heads=num_heads, n_points=n_points, ratio=deform_ratio)
        self.attn = MSDeformAttn(embed_dims=dim, num_levels=n_levels, num_heads=num_heads, num_points=n_points, value_proj_ratio=deform_ratio, batch_first=True)
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, query, reference_points, feat, spatial_shapes, level_start_index):

        def _inner_forward(query, feat):

            #attn = self.attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes, level_start_index, None)
            attn = self.attn(query=self.query_norm(query), reference_points=reference_points, value=self.feat_norm(feat), spatial_shapes=spatial_shapes, level_start_index=level_start_index, key_padding_mask=None)
            return query + self.gamma * attn

        if self.with_cp and query.requires_grad:
            query = cp.checkpoint(_inner_forward, query, feat)
        else:
            query = _inner_forward(query, feat)

        return query



class InteractionBlock(nn.Module):
    def __init__(self, dim, num_heads=6, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 drop=0., drop_path=0., with_cffn=True, cffn_ratio=0.25, init_values=0.,
                 deform_ratio=1.0, extra_extractor=False, with_cp=False):
        super().__init__()

        self.injector = Injector(dim=dim, n_levels=3, num_heads=num_heads, init_values=init_values,
                                 n_points=n_points, norm_layer=norm_layer, deform_ratio=deform_ratio,
                                 with_cp=with_cp)
        self.extractor = Extractor(dim=dim, n_levels=1, num_heads=num_heads, n_points=n_points,
                                   norm_layer=norm_layer, deform_ratio=deform_ratio, with_cffn=with_cffn,
                                   cffn_ratio=cffn_ratio, drop=drop, drop_path=drop_path, with_cp=with_cp)
        if extra_extractor:
            self.extra_extractors = nn.Sequential(*[
                Extractor(dim=dim, num_heads=num_heads, n_points=n_points, norm_layer=norm_layer,
                          with_cffn=with_cffn, cffn_ratio=cffn_ratio, deform_ratio=deform_ratio,
                          drop=drop, drop_path=drop_path, with_cp=with_cp)
                for _ in range(2)
            ])
        else:
            self.extra_extractors = None

    def forward(self, x, c, blocks, deform_inputs1, deform_inputs2, H, W):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        for idx, blk in enumerate(blocks):
            x = blk(x, H, W)
        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c


class InteractionBlockWithSelection(InteractionBlock):
    def __init__(self, ratio_per_sample=False, selec_skip_ids=[], selec_model_ids=[], **kwargs):
        super(InteractionBlockWithSelection, self).__init__(**kwargs)
        self.ratio_per_sample = ratio_per_sample
        self.selec_skip_ids = selec_skip_ids
        self.selec_modelId_to_pos = {x: i for i, x in enumerate(selec_model_ids)}

        y = 1

    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio) ** 2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio):
        # n_skip = 12 - len(selective_modules)
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        layer_ratio_loss = 0.
        has_loss = 0
        used_selectors = []
        for i in range(indexes[0], indexes[-1] + 1):
            if i in self.selec_skip_ids:
                x = blks[i](x, x_dense=x)
                used_selectors.append(None)
            else:
                if self.training:
                    selector, diff_selector = selective_modules[self.selec_modelId_to_pos[i]](x)
                    x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - diff_selector) * x
                    layer_ratio_loss += self._ratio_loss(diff_selector, keep_ratio[self.selec_modelId_to_pos[i]])
                    has_loss += 1
                else:
                    if x.shape[0] == 1:
                        selector, _ = selective_modules[self.selec_modelId_to_pos[i]](x)
                        real_indices = torch.argsort(selector.int(), dim=1, descending=True) \
                            [:, :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        selected_x = torch.gather(x, 1, real_indices)
                        selected_x = blks[i](selected_x, x_dense=x, selec_mask=selector)
                        x.scatter_(1, real_indices, selected_x)

                        used_selectors.append(selector.to(torch.float).unsqueeze(-1))
                    else:
                        assert False, "does not suppoert >1 Samples in batch"
                        selector, diff_selector = selective_modules[self.block_to_selec_module[i]](x)
                        l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                        nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                        nt_x = blks[i](nt_x, src_key_padding_mask=None)
                        x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W,
                           )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, layer_ratio_loss, has_loss, used_selectors


class InteractionBlockWithInitialSelection(InteractionBlock):
    def __init__(self, selec_skip_ids=[],
                 select_module_ids=[],
                 prev_x_update_layer=False,
                 interact_index=0,
                 matching_module=dict(),
                 matching_module_idx=1,
                 ratio_match_module_sparsity_loss=10,
                 ratio_per_sample=False,
                 *args, **kwargs):
        super(InteractionBlockWithInitialSelection, self).__init__(*args, **kwargs)
        self.ratio_per_sample = ratio_per_sample
        self.selec_skip_ids = selec_skip_ids
        self.matching_module_idx = matching_module_idx
        self.selec_modelId_to_pos = {x: i for i, x in enumerate(select_module_ids)}
        self.prev_x_update_layer = prev_x_update_layer

        assert type(interact_index) is list and len(interact_index) == 2
        if matching_module_idx >= interact_index[0] and matching_module_idx < interact_index[1]:
            self.ratio_match_module_sparsity_loss = ratio_match_module_sparsity_loss
            self.matching_module = MODELS.build(matching_module)

    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio) ** 2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, prev_x, keep_ratio,
                prev_selector=None):

        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        diff_selector = None
        layer_ratio_loss = 0.
        has_loss = 0
        for i in range(indexes[0], indexes[-1] + 1):
            if i in self.selec_skip_ids:
                x = blks[i](x, x_dense=x)
            else:
                if i == self.matching_module_idx and prev_x['x_to_map'] is not None:
                    selector, diff_selector, selector_soft = self.matching_module(
                        x, prev_x['x_ref'], prev_x['x_to_map'], prev_x['selector'], H, W)

                    if self.training:
                        layer_ratio_loss += self.ratio_match_module_sparsity_loss * self._ratio_loss(selector_soft, 0.5)
                        has_loss += self.ratio_match_module_sparsity_loss

                elif i in self.selec_modelId_to_pos.keys():  # call selective module
                    j = self.selec_modelId_to_pos[i]
                    sm = selective_modules[j]
                    selector, diff_selector, selector_soft = sm(x, H, W)

                    if prev_selector is not None: # avoid reactivation
                        diff_selector = diff_selector * prev_selector
                        selector = diff_selector.bool().squeeze(2)

                elif diff_selector is None:
                    diff_selector = torch.ones_like(x[:, :, 0:1], device=x.device)
                    selector = diff_selector.bool().squeeze(2)

                prev_selector = diff_selector
                if i == self.matching_module_idx:
                    prev_x['x_ref'] = x

                if self.training:
                    x = (diff_selector * blks[i](x, src_key_padding_mask=~selector) + (1 - diff_selector) * x)

                    if i in self.selec_modelId_to_pos.keys():
                        layer_ratio_loss += self._ratio_loss(diff_selector, keep_ratio[j])
                        has_loss += 1
                else:
                    assert x.shape[0] == 1, 'In testmode batch_size must be 1'
                    selector_sort = torch.argsort(selector.int(), dim=1, descending=True)
                    real_indices = selector_sort[:, :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    selected_x = torch.gather(x, 1, real_indices)
                    selected_x = blks[i](selected_x, x_dense=x, selec_mask=selector)
                    x.scatter_(1, real_indices, selected_x)

                if i == self.prev_x_update_layer:
                    prev_x['selector'] = selector
                    prev_x['x_to_map'] = x

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W,
                           )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, layer_ratio_loss, has_loss, prev_x, prev_selector


class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(2 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv4 = nn.Sequential(*[
            nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            nn.SyncBatchNorm(4 * inplanes),
            nn.ReLU(inplace=True)
        ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3, c4


def left_align_tokens2(x: torch.Tensor, mask: torch.Tensor):
    """
        x: tensor of shape B, L, D
        mask: boolean tensor of shape B, L
    """
    # flatten_mask = torch.flatten(mask, start_dim=0, end_dim=1)  # (B*L)
    # flatten_x = torch.flatten(x, start_dim=0, end_dim=1)  # (B*L, D)
    # x.masked_scatter_(mask, flatten_x[flatten_mask])

    l_aligned_mask, indexes = torch.sort(mask.int(), dim=1, descending=True, stable=True)
    l_aligned_mask = l_aligned_mask.bool()  # bool --> int (sort) --> bool, because CUDA does not sort boolean tensor
    l_aligned_x = x[torch.arange(x.shape[0], device=x.device).unsqueeze(1), indexes]

    return l_aligned_x, l_aligned_mask
