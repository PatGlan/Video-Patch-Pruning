import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Mlp, Attention
from timm.models.layers import DropPath
import torch.utils.checkpoint as cp
import random
from mmdet.registry import MODELS
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention as MSDeformAttn

_logger = logging.getLogger(__name__)


class Downsize_Attention(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, tau=8, num_heads=1, qkv_bias=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.num_heads = num_heads
        head_dim = hid_dim // num_heads
        self.scale = head_dim ** -0.5

        # self.q = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        # self.k = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        self.v = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        self.proj = nn.Linear(hid_dim, out_dim)

        # self.act = nn.GELU()
        # self.sampler = nn.Linear(out_dim, 2)
        # self.gate2 = nn.LogSoftmax(dim=-1)
        self.tau = tau

        self.init_mlp = Mlp(in_features=in_dim, hidden_features=hid_dim, out_features=hid_dim, act_layer=nn.GELU)
        # self.init_mlp = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        self.norm = nn.LayerNorm(in_dim)

    def forward(self, x_q, x_k, x_v, mask_selector=None, H=None, W=None):
        B, N, C = x_q.shape

        q = self.init_mlp(self.norm(x_q)).unsqueeze(1)  # (B, H, N, C)
        k = self.init_mlp(self.norm(x_k)).unsqueeze(1)  # (B, H, N, C)
        v = self.v(x_v).reshape(B, N, self.num_heads, self.hid_dim // self.num_heads).permute(0, 2, 1, 3)

        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)

        attn = (q @ k.transpose(-2, -1))  # * self.scale
        attn = attn / (q_norm * k_norm.transpose(-2, -1) + 1e-6)  # cosine similarity

        attn *= self.tau
        if mask_selector is not None:
            mask_selector = (~mask_selector).unsqueeze(1).unsqueeze(1)
            new_attn_mask = torch.zeros_like(mask_selector, dtype=q.dtype)
            new_attn_mask.masked_fill_(mask_selector, float("-inf"))
            attn += new_attn_mask
        # attn = (attn / self.tau).softmax(dim=-1)
        attn = attn.softmax(dim=-1)

        apply_test = False
        if apply_test and random.random() < 1 / 100:
            w_map = torch.linspace(0, 1, W).repeat(H, 1)
            h_map = torch.linspace(0, 1, H).repeat(W, 1).transpose(0, 1)
            wh_map = torch.stack([w_map, h_map], dim=0).flatten(1).transpose(0, 1)

            manh_dist = torch.cdist(wh_map, wh_map, p=1).to(attn.device)
            attn_mask = (~mask_selector)[0, 0, 0]
            attn_0 = attn[0, 0, :, :]
            top_k_values, top_k_indices = torch.topk(attn_0, k=5, dim=1, sorted=True)
            top_k_values_masked = torch.gather(manh_dist, dim=1, index=top_k_indices)

            print(f'top_k_value: {top_k_values_masked.mean().item()}')

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.hid_dim)
        x = self.proj(x)

        return x


class Downsize_Attention_with_selector(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, tau=10, scale_attn=10, mask_pooling_size=4, num_heads=1,
                 qkv_bias=False, quantile=0.7, train_dynamicly=False):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.quantile = quantile
        self.train_dynamicly = train_dynamicly

        self.num_heads = num_heads
        head_dim = hid_dim // num_heads
        self.scale = head_dim ** -0.5

        # self.proj = nn.Linear(hid_dim, 2)

        self.tau = tau
        self.scale_attn = scale_attn

        # self.use_argmax = scale_attn >= 100
        self.use_argmax = True
        self.use_dist_mat = True

        # if not self.use_argmax:
        #    self.v = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        #    self.proj = nn.Linear(hid_dim, out_dim)
        # else:
        #    self.v = nn.Linear(in_dim, out_dim, bias=qkv_bias)

        # TOO OLD self.init_mlp = Mlp(in_features=in_dim, hidden_features=hid_dim, out_features=hid_dim, act_layer=nn.GELU)
        # if not self.use_argmax:
        # self.init_mlp = nn.Linear(in_dim, hid_dim, bias=qkv_bias)
        # self.norm_input = nn.LayerNorm(in_dim)
        if self.use_dist_mat:
            # t00 = time.time()
            # W=30
            # H=17
            # w_map = torch.linspace(0, 1, W).repeat(H, 1)
            # h_map = torch.linspace(0, 1, H).repeat(W, 1).transpose(0, 1)
            # wh_map = torch.stack([w_map, h_map], dim=0).flatten(1).transpose(0, 1).cuda()
            # manh_dist = torch.cdist(wh_map, wh_map, p=1)
            # t01 = time.time()
            # manh_dist_np = manh_dist.cpu().detach().numpy()
            # dist_scale = (2 - manh_dist) / 2
            # dist_scale = dist_scale.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            # dist_scale = torch.log10(dist_scale * 10 + 1e-6)
            # dist_scale_np = dist_scale.cpu().detach().numpy()

            W_dist, H_dist = 60, 34
            w_map02 = torch.linspace(0, 1, W_dist).repeat(H_dist, 1)
            h_map02 = torch.linspace(0, 1, H_dist).repeat(W_dist, 1).transpose(0, 1)
            wh_map02 = torch.stack([w_map02, h_map02], dim=0).flatten(1).transpose(0, 1)
            manh_dist02 = torch.cdist(wh_map02, wh_map02, p=1)
            # manh_dist02_np = manh_dist02.cpu().detach().numpy()
            dist_scale02 = (2 - manh_dist02) / 2
            dist_scale02 = dist_scale02  # .unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            dist_scale02 = torch.log10(dist_scale02 * 10 + 1e-6)  # default()
            dist_scale02 = dist_scale02.view(H_dist, W_dist, H_dist, W_dist)
            self.dist_scale = nn.Parameter(dist_scale02, requires_grad=False)

            # t02 = time.time()
            # manh_dist03 = manh_dist02.view(-1, H2, W2)
            # manh_dist03 = F.interpolate(manh_dist03, size=(W), mode='linear', align_corners=True).transpose(1, 2)
            # manh_dist03 = F.interpolate(manh_dist03, size=(H), mode='linear', align_corners=True).transpose(1, 2)
            # manh_dist03 = manh_dist03.reshape(-1, H * W).transpose(0, 1).view(-1, H2, W2)
            # manh_dist03 = F.interpolate(manh_dist03, size=(W), mode='linear', align_corners=True).transpose(1, 2)
            # manh_dist03 = F.interpolate(manh_dist03, size=(H), mode='linear', align_corners=True).transpose(1, 2)
            # manh_dist03 = manh_dist03.reshape(-1, H * W).transpose(0, 1)

            # interpolate bilinear
            # manh_dist03 = manh_dist02.view(H2, W2, H2, W2)
            # manh_dist03 = F.interpolate(manh_dist03, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            # manh_dist03 = F.interpolate(manh_dist03, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            # manh_dist03 = manh_dist03.view(H * W, H * W)

            # dist_scale03 = dist_scale02.view(H2, W2, H2, W2)
            # dist_scale03 = F.interpolate(dist_scale03, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            # dist_scale03 = F.interpolate(dist_scale03, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            # dist_scale03 = dist_scale03.view(H * W, H * W).unsqueeze(0).unsqueeze(0) # (1, 1, N, N)

            # dist_scale04 = (2 - manh_dist03) / 2
            # dist_scale04 = dist_scale04.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            # dist_scale04 = torch.log10(dist_scale04 * 10 + 1e-6)
            # dist_scale04_np = dist_scale04.cpu().detach().numpy()

            # manh_dist03 = manh_dist02.unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            # manh_dist03 = F.interpolate(manh_dist03, size=(H*W, H*W), mode='area', align_corners=None)
            # manh_dist03 = manh_dist03.squeeze()

            # t03 = time.time()

            # manh_dist03_np = manh_dist03.cpu().detach().numpy()
            # dist_scale03_np = dist_scale03.cpu().detach().numpy()
            # print(f'time_original: {(t01 - t00)*10e3:.2f}, time_interpolated: {(t03 - t02)*10e3:.2f}')
            y = 1

        # self.widener = nn.Conv2d(2, 2, kernel_size=5, stride=1, padding=2, bias=False)
        self.repl_pad = nn.ReplicationPad2d(mask_pooling_size)
        self.avg_pooler = nn.AvgPool2d(int(2 * mask_pooling_size + 1), stride=1)

        self.selector = Mlp(in_features=in_dim, hidden_features=hid_dim, out_features=2, act_layer=nn.GELU)
        self.norm_selector = nn.LayerNorm(in_dim)
        self.gate2 = nn.LogSoftmax(dim=-1)

        # dist_h, dist_w = 32, 32
        # w_map = torch.linspace(0, 1, dist_w).repeat(dist_h, 1)
        # h_map = torch.linspace(0, 1, dist_h).repeat(dist_w, 1).transpose(0, 1)
        # wh_map = torch.stack([w_map, h_map], dim=0).flatten(1).transpose(0, 1)
        # manh_dist = torch.cdist(wh_map, wh_map, p=1)
        # self.manh_dist = nn.Parameter(manh_dist, requires_grad=False)

    def forward(self, x_q, x_k, x_v, mask_selector=None, H=None, W=None):
        B, N, C = x_q.shape

        # q = self.init_mlp(self.norm_input(x_q)).unsqueeze(1)  # (B, H, N, C)
        # k = self.init_mlp(self.norm_input(x_k)).unsqueeze(1)  # (B, H, N, C)

        q = x_q.unsqueeze(1)
        k = x_k.unsqueeze(1)
        v = x_v.unsqueeze(1)

        attn = (q @ k.transpose(-2, -1))  # * self.scale

        if self.use_dist_mat:
            # old
            # self.dist_scale = self.dist_scale.to(attn.device)
            # attn = attn + self.dist_scale

            # new
            dist_scale = self.dist_scale
            dist_scale = F.interpolate(dist_scale, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            dist_scale = F.interpolate(dist_scale, size=(H, W), mode='nearest', align_corners=None).permute(2, 3, 0, 1)
            dist_scale = dist_scale.view(H * W, H * W).unsqueeze(0).unsqueeze(0)  # (1, 1, N, N)
            attn = attn + dist_scale

        if mask_selector is not None:
            mask_selector = (~mask_selector).unsqueeze(1).unsqueeze(1)
            new_attn_mask = torch.zeros_like(mask_selector, dtype=q.dtype)
            new_attn_mask.masked_fill_(mask_selector, float("-inf"))
            attn += new_attn_mask
        tau_match = 10.0
        attn = F.gumbel_softmax(attn * tau_match, tau=tau_match, hard=True, dim=-1)

        x = (attn @ v).transpose(1, 2).squeeze(2)

        f = self.norm_selector(x)
        f = self.selector(f)  # shape (B,L,2)

        f = self.gate2(f)  # shape (B,L,2)
        f = f.view(B, H, W, 2).permute(0, 3, 1, 2)
        f = self.avg_pooler(self.repl_pad(f))
        f = f.flatten(2).transpose(1, 2)
        selector_soft = F.gumbel_softmax(f * self.tau, tau=self.tau, hard=False)

        if self.train_dynamicly:
            dim = -1
            index = selector_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(selector_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index,
                                                                                                            1.0)
            diff_selector = y_hard - selector_soft.detach() + selector_soft
            diff_selector = diff_selector[:, :, 0:1]  # shape (B, L, 1)
            selector_soft = selector_soft[:, :, 0:1]  # shape (B, L, 1)
        else:
            dim = 1
            selector_soft = selector_soft[:, :, 0:1]  # shape (B, L, 1)
            sorted_selector_soft, index = torch.sort(selector_soft.squeeze(-1), dim=1, descending=True)
            num_p = f.shape[1]
            index = index[:, :int(num_p * self.quantile)].unsqueeze(2)
            y_hard = torch.zeros_like(selector_soft, memory_format=torch.legacy_contiguous_format).scatter_(dim, index,
                                                                                                            1.0)
            diff_selector = y_hard - selector_soft.detach() + selector_soft

        selector = diff_selector.bool().squeeze(2)
        return selector, diff_selector, selector_soft


def scale_mask_from_p(mask_p, p_min=0.0, p_max=1.0):
    B, P, C = mask_p.shape
    if C > 1:
        mask_p = mask_p[..., 0:1]
    mask_p_scaled = mask_p * (p_max - p_min) + p_min
    mask_p_scaled = torch.clip(mask_p_scaled, min=0.0, max=1.0)
    return mask_p_scaled


def sample_mask_from_p_min_max(mask_p):
    # V01
    mask_sampled = torch.rand(mask_p.shape, device=mask_p.device) < mask_p

    # V02
    '''
    tau = 1
    dim=-1
    gumbels = (
        -torch.empty_like(mask_p, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    test = (mask_p + 1e-6).log()
    gumbels = ((mask_p + 1e-6).log() + gumbels) / tau  # ~Gumbel(logits,tau)
    #gumbels = (mask_p + (gumbels / 10))  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    #if hard:
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(mask_p, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    mask_sampled = y_hard - y_soft.detach() + y_soft
    #mask_sampled = mask_sampled.bool()
    '''
    # V03
    # noise = torch.rand_like(mask_p)

    y = 1

    # sample from mask_p
    # tau=1.0
    # if not is_inference:
    #    selector = F.gumbel_softmax(mask_p, tau=tau, hard=True)[:, :, 0:1]  # shape (B, L, 1)
    #    diff_selector = selector
    #    selector = diff_selector.bool().squeeze(2)
    # else:
    #    selector = F.gumbel_softmax(mask_p, tau=tau, hard=True)[:, :, 0].bool()  # shape (B, L, 1)
    #    diff_selector = None

    return mask_sampled


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


def easy_gather(x, indices):
    # used by InteractionBlockForEvo
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    assert N_new == N, 'Just a check. indices should be full shape'
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out


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


class InteractionBlockForEvo(InteractionBlock):
    def __init__(self, **kwargs):
        super(InteractionBlockForEvo, self).__init__(**kwargs)

    def forward(self, cls_token, x, c, indexes, blocks, norms, vs, qks, projs,
                deform_inputs1, deform_inputs2, H, W, prune_ratio, tradeoff, cls_attn):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        real_indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], x.shape[1])
        x = torch.cat((cls_token, x), dim=1)
        for index, blk in enumerate(blocks):
            if prune_ratio[indexes[index]] != 1:
                # token selection
                x_patch = x[:, 1:, :]

                B, N, C = x_patch.shape
                N_ = int(N * prune_ratio[indexes[index]])
                indices = torch.argsort(cls_attn, dim=1, descending=True)
                x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
                x_sorted = easy_gather(x_patch, indices)
                x_patch, cls_attn = x_sorted[:, :, :-1], x_sorted[:, :, -1]

                real_indices = easy_gather(real_indices.unsqueeze(-1), indices).squeeze(-1)

                if self.training:
                    x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
                else:
                    x[:, 1:, :] = x_patch
                    x_ = x
                x = x_[:, :N_ + 1]

                # slow updating
                tmp_x = x
                B, N, C = x.shape
                x = norms[index](x)
                v = vs[index](x)
                attn = qks[index](x)

                # with torch.no_grad():
                if self.training:
                    temp_cls_attn = (1 - tradeoff[indexes[index]]) * cls_attn[:, :N_] + tradeoff[
                        indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:],
                        dim=1)
                    cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

                else:
                    cls_attn[:, :N_] = (1 - tradeoff[indexes[index]]) * cls_attn[:, :N_] + tradeoff[
                        indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:],
                        dim=1)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = projs[index](x)
                x = blk.drop_path(x)
                x = x + tmp_x

                x = blk(x)

                # fast updating, only preserving the placeholder tokens presents enough good results on DeiT
                if False and indexes[index] == 11:
                    pass
                else:
                    if self.training:
                        x = torch.cat((x, x_[:, N_ + 1:]), dim=1)
                    else:
                        x_[:, :N_ + 1] = x
                        x = x_

            # normal updating in the beginning four layers
            else:
                tmp_x = x
                B, N, C = x.shape
                x = norms[index](x)
                v = vs[index](x)
                attn = qks[index](x)

                if indexes[index] == 0:
                    cls_attn = torch.sum(attn[:, :, 0, 1:], dim=1)
                else:
                    cls_attn = (1 - tradeoff[indexes[index]]) * cls_attn + tradeoff[indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:], dim=1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = projs[index](x)
                x = blk.drop_path(x)
                x = x + tmp_x

                x = blk(x)
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]

        # restore the original orders of tokens and cls_attn
        inv_indices = torch.argsort(real_indices, dim=1)
        x = torch.cat((x, cls_attn.unsqueeze(-1)), dim=-1)
        x = easy_gather(x, inv_indices)
        x, cls_attn = x[:, :, :-1], x[:, :, -1]

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return cls_token, x, c, cls_attn

    def forward_demo(self, cls_token, x, c, indexes, blocks, norms, vs, qks, projs,
                     deform_inputs1, deform_inputs2, H, W, prune_ratio, tradeoff, cls_attn):
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        real_indices = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], x.shape[1])
        x = torch.cat((cls_token, x), dim=1)
        sele_dict = {}
        for index, blk in enumerate(blocks):
            if prune_ratio[indexes[index]] != 1:
                # token selection
                x_patch = x[:, 1:, :]

                B, N, C = x_patch.shape
                N_ = int(N * prune_ratio[indexes[index]])
                indices = torch.argsort(cls_attn, dim=1, descending=True)
                x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
                x_sorted = easy_gather(x_patch, indices)
                x_patch, cls_attn = x_sorted[:, :, :-1], x_sorted[:, :, -1]

                real_indices = easy_gather(real_indices.unsqueeze(-1), indices).squeeze(-1)
                sele_dict[indexes[index]] = real_indices[:, :N_].unsqueeze(-1)

                if self.training:
                    x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
                else:
                    x[:, 1:, :] = x_patch
                    x_ = x
                x = x_[:, :N_ + 1]

                # slow updating
                tmp_x = x
                B, N, C = x.shape
                x = norms[index](x)
                v = vs[index](x)
                attn = qks[index](x)

                # with torch.no_grad():
                if self.training:
                    temp_cls_attn = (1 - tradeoff[indexes[index]]) * cls_attn[:, :N_] + tradeoff[
                        indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:],
                        dim=1)
                    cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

                else:
                    cls_attn[:, :N_] = (1 - tradeoff[indexes[index]]) * cls_attn[:, :N_] + tradeoff[
                        indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:],
                        dim=1)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = projs[index](x)
                x = blk.drop_path(x)
                x = x + tmp_x

                x = blk(x)

                # fast updating, only preserving the placeholder tokens presents enough good results on DeiT
                if False and indexes[index] == 11:
                    pass
                else:
                    if self.training:
                        x = torch.cat((x, x_[:, N_ + 1:]), dim=1)
                    else:
                        x_[:, :N_ + 1] = x
                        x = x_

            # normal updating in the beginning four layers
            else:
                tmp_x = x
                B, N, C = x.shape
                x = norms[index](x)
                v = vs[index](x)
                attn = qks[index](x)

                if indexes[index] == 0:
                    cls_attn = torch.sum(attn[:, :, 0, 1:], dim=1)
                else:
                    cls_attn = (1 - tradeoff[indexes[index]]) * cls_attn + tradeoff[indexes[index]] * torch.sum(
                        attn[:, :, 0, 1:], dim=1)
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = projs[index](x)
                x = blk.drop_path(x)
                x = x + tmp_x

                x = blk(x)
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]

        # restore the original orders of tokens and cls_attn
        inv_indices = torch.argsort(real_indices, dim=1)
        x = torch.cat((x, cls_attn.unsqueeze(-1)), dim=-1)
        x = easy_gather(x, inv_indices)
        x, cls_attn = x[:, :, :-1], x[:, :, -1]

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return cls_token, x, c, cls_attn, sele_dict


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

    def update_tmp_selector(self, tmp_selector, new_selector):
        if tmp_selector is None:
            return new_selector
        else:
            return torch.logical_or(tmp_selector, new_selector)

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
                        # selector_x = self.update_tmp_selector(selector_x, selector)
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

    def forward_demo(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio):
        # n_skip = 12 - len(selective_modules)
        # assert (blks[0].TransformerEncoderLayer.self_attn.num_heads % 2) == 0
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        sele_dict = {}
        for i in range(indexes[0], indexes[-1] + 1):
            if i in self.selec_skip_ids:
                x = blks[i](x)
            else:
                if self.training:
                    selector, diff_selector = selective_modules[self.selec_modelId_to_pos[i]](x)
                    x = diff_selector * blks[i](x, src_key_padding_mask=~selector) + \
                        (1 - diff_selector) * x
                else:
                    if x.shape[0] == 1:
                        selector, _ = selective_modules[self.selec_modelId_to_pos[i]](x)
                        real_indices = torch.argsort(selector.int(), dim=1, descending=True)[:,
                                       :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                        selected_x = torch.gather(x, 1, real_indices)
                        selected_x = blks[i](selected_x, selec_mask=selector)
                        x.scatter_(1, real_indices, selected_x)
                    else:
                        selector, diff_selector = selective_modules[self.selec_modelId_to_pos[i]](x)
                        l_aligned_x, l_aligned_mask = left_align_tokens2(x, selector)
                        nt_x = torch._nested_tensor_from_mask(l_aligned_x, l_aligned_mask, mask_check=False)
                        nt_x = blks[i](nt_x, src_key_padding_mask=None)
                        x.masked_scatter_(selector.unsqueeze(-1), torch.cat(nt_x.unbind(), 0))

                sele_dict[i] = selector

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W)
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, sele_dict


class InteractionBlockWithDynVitSelection(InteractionBlock):
    def __init__(self, ratio_per_sample=False, selec_skip_ids=[], selec_model_ids=[], **kwargs):
        super(InteractionBlockWithDynVitSelection, self).__init__(**kwargs)
        self.pruning_loc = selec_model_ids
        self.selec_modelId_to_pos = {x: i for i, x in enumerate(selec_model_ids)}

        # embed_dim=kwargs['dim']
        # predictor_list = [PredictorLG(embed_dim) for _ in range(len(self.pruning_loc))]
        # self.score_predictor = nn.ModuleList(predictor_list)
        y = 1

    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio) ** 2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def update_tmp_selector(self, tmp_selector, new_selector):
        if tmp_selector is None:
            return new_selector
        else:
            return torch.logical_or(tmp_selector, new_selector)

    def batch_index_select(self, x, idx):
        if len(x.size()) == 3:
            B, N, C = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B * N, C)
            out = out[idx.reshape(-1)]
            out = out.reshape(B, N_new, C)
            return out
        elif len(x.size()) == 2:
            B, N = x.size()
            N_new = idx.size(1)
            offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
            idx = idx + offset
            out = x.reshape(B * N)[idx.reshape(-1)].reshape(B, N_new)
            return out
        else:
            raise NotImplementedError

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, keep_ratio,
                prev_decision):
        # n_skip = 12 - len(selective_modules)
        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])
        B = x.shape[0]
        init_n = x.shape[1]
        out_pred_prob = []
        # layer_ratio_loss = 0.
        # has_loss = 0
        # p_count = 0
        for i in range(indexes[0], indexes[-1] + 1):
            if i in [0, 1, 2]:
                # selector_mask = torch.ones_like(x[:, :, 0]).to(torch.bool)
                x = blks[i](x, x_dense=x)
            else:
                if self.training:
                    if i in self.pruning_loc:
                        # spatial_x = x[:, 1:] #for cls-token
                        p_count = self.selec_modelId_to_pos[i]
                        pred_score = selective_modules[p_count](x, prev_decision).reshape(B, -1, 2)
                        prev_decision = F.gumbel_softmax(pred_score, hard=True)[:, :, 0:1] * prev_decision
                        out_pred_prob.append(prev_decision.reshape(B, init_n))  # todo check which ones are neccesary
                        # p_count += 1
                    x = prev_decision * blks[i](x, src_key_padding_mask=~(prev_decision.bool().squeeze(2))) + (
                                1 - prev_decision) * x
                else:
                    if i in self.pruning_loc:
                        p_count = self.selec_modelId_to_pos[i]
                        pred_score = selective_modules[p_count](x, prev_decision).reshape(B, -1, 2)
                        score = pred_score[:, :, 0]
                        prev_decision_bool = prev_decision.to(torch.bool)
                        score.masked_fill_(~prev_decision_bool.squeeze(-1), -torch.inf)

                        num_keep_node = int(init_n * keep_ratio[p_count])
                        val, policy = torch.sort(score, dim=1, descending=True)
                        keep_policy = policy[:, :num_keep_node]
                        prune_policy = policy[:, num_keep_node:]
                        prev_decision.scatter_(1, prune_policy.unsqueeze(-1), torch.zeros_like(prev_decision))

                    # apply masks
                    real_indices = keep_policy.unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    selected_x = torch.gather(x, 1, real_indices)
                    selector_mask = prev_decision[:, :, 0]
                    selected_x = blks[i](selected_x, x_dense=x, selec_mask=selector_mask.type(torch.bool))
                    x.scatter_(1, real_indices, selected_x)

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W,
                           )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, prev_decision, out_pred_prob



class InteractionBlockWithInitialSelection(InteractionBlock):
    def __init__(self, ratio_per_sample=False, selec_skip_ids=[], selec_model_ids=[], prev_x_strategy="last",
                 matching_module_idx=1,
                 prev_x_update_layer=False, prev_x_update_attn=False, use_buffered_loss=False, interact_index=0,
                 kappa_buff_size=4, momentum_kappa=0.99, tau_update_layer=8, scale_attn_update_layer=10,
                 mask_pooling_size=4,
                 initBlock_keep_ratio=0.7,
                 loss_scale_matching_module=10, prune_adapter=False, **kwargs):
        emb_dim = kwargs.get('dim', 192)
        super(InteractionBlockWithInitialSelection, self).__init__(**kwargs)
        self.prune_adapter = prune_adapter
        self.ratio_per_sample = ratio_per_sample
        self.selec_skip_ids = selec_skip_ids
        self.matching_module_idx = matching_module_idx
        self.selec_modelId_to_pos = {x: i for i, x in enumerate(selec_model_ids)}

        self.dynamic_matching_module = prev_x_update_attn
        self.prev_x_update_layer = prev_x_update_layer

        # self.first_pruning_layer = selec_model_ids[0]
        if interact_index == 0 and matching_module_idx >= 0:
            self.loss_scale_matching_module = loss_scale_matching_module
            self.matching_module = Downsize_Attention_with_selector(emb_dim, emb_dim // 2, emb_dim,
                                                                    tau=tau_update_layer,
                                                                    scale_attn=scale_attn_update_layer,
                                                                    mask_pooling_size=mask_pooling_size,
                                                                    quantile=initBlock_keep_ratio,
                                                                    train_dynamicly=self.dynamic_matching_module)

        self.use_buffered_loss = use_buffered_loss
        if use_buffered_loss:
            self.kappa_buff_size = kappa_buff_size
            self.momentum_kappa = momentum_kappa
            self.register_buffer(f'running_kappa', torch.ones(1))

    def _buffered_ratio_loss(self, selector: torch.Tensor, ratio, selec_idx):
        p_curr = selector.sum() / (selector.shape[0] * selector.shape[1])

        p_mean = ((self.kappa_buff_size - 1) * self.running_kappa.detach() + p_curr) / self.kappa_buff_size
        loss = self.kappa_buff_size * ((p_mean - ratio) ** 2)

        self.running_kappa = (self.momentum_kappa * self.running_kappa +
                              (1 - self.momentum_kappa) * p_curr.detach())

        return loss

    def _ratio_loss(self, selector: torch.Tensor, ratio=1.):
        if not self.ratio_per_sample:
            return (selector.sum() / (selector.shape[0] * selector.shape[1]) - ratio) ** 2
        else:
            n_tokens = selector.shape[1]
            return ((selector.sum(dim=1) / n_tokens - ratio) ** 2).mean()

    def update_tmp_selector(self, tmp_selector, new_selector):
        assert False
        if tmp_selector is None:
            return new_selector
        else:
            return torch.logical_or(tmp_selector, new_selector)

    def scale_mask_x_to_c(self, x_mask, H, W):
        B = x_mask.shape[0]
        mask_c = []
        mask_c.append(
            x_mask.reshape(B, H, W).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).reshape(B, -1, 1))
        mask_c.append(x_mask)
        mask_c.append(
            torch.max_pool2d(x_mask.reshape(B, 1, H, W), kernel_size=2, stride=2).reshape(B, -1, 1))
        diff_selec_c = torch.cat(mask_c, dim=1)
        return diff_selec_c

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules, prev_x, keep_ratio,
                prev_selector=None, prev_inst=None):

        x = self.injector(query=x, reference_points=deform_inputs1[0],
                          feat=c, spatial_shapes=deform_inputs1[1],
                          level_start_index=deform_inputs1[2])

        diff_selector = None
        selector_soft = None
        layer_ratio_loss = 0.
        has_loss = 0
        selectors = []
        for i in range(indexes[0], indexes[-1] + 1):
            if i in self.selec_skip_ids:
                x = blks[i](x, x_dense=x)
            else:
                '''
                diff_selector = torch.ones_like(x[:, :, 0:1], device=x.device) #todo remove
                selector = diff_selector.bool().squeeze(2)
                selector_soft = diff_selector.softmax(dim=-1)

                '''
                if i == self.matching_module_idx and prev_x['x_11'] is not None:
                    selector, diff_selector, selector_soft = self.matching_module(
                        x, prev_x['x_0'], prev_x['x_11'], prev_x['selector'], H, W)

                    if prev_inst is not None:
                        prev_inst_mask, _ = prev_inst.max(dim=1)
                        prev_inst_mask = F.max_pool2d(prev_inst_mask[None, :], kernel_size=16, stride=16).flatten()

                        diff_selector = torch.clamp(diff_selector + prev_inst_mask[None, :, None], max=1.0)
                        selector = diff_selector.bool().squeeze(2)
                    if self.training:

                        if self.matching_module.train_dynamicly:
                            goal_kr = self.matching_module.quantile
                            layer_ratio_loss += self.loss_scale_matching_module * self._ratio_loss(diff_selector,
                                                                                                   goal_kr)
                        else:
                            layer_ratio_loss += self.loss_scale_matching_module * self._ratio_loss(selector_soft, 0.5)
                        has_loss += self.loss_scale_matching_module

                elif i in self.selec_modelId_to_pos.keys():  # call selective module
                    j = self.selec_modelId_to_pos[i]
                    sm = selective_modules[j]
                    selector, diff_selector, selector_soft = sm(x, H, W)

                    # avoid reavitvation
                    if prev_selector is not None:
                        diff_selector = diff_selector * prev_selector
                        selector = diff_selector.bool().squeeze(2)
                        selector_soft = selector_soft * prev_selector
                elif diff_selector is None:
                    diff_selector = torch.ones_like(x[:, :, 0:1], device=x.device)
                    selector = diff_selector.bool().squeeze(2)
                    selector_soft = diff_selector.softmax(dim=-1)

                    # avoid reavitvation
                    if prev_selector is not None:
                        diff_selector = diff_selector * prev_selector
                        selector = diff_selector.bool().squeeze(2)
                        selector_soft = selector_soft * prev_selector

                prev_selector = diff_selector
                if i == self.matching_module_idx:
                    prev_x['x_0'] = x

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
                    prev_x['diff_selector'] = diff_selector
                    prev_x['x_11'] = x

                selectors.append(selector.to(torch.float).unsqueeze(0).detach())

        c = self.extractor(query=c, reference_points=deform_inputs2[0],
                           feat=x, spatial_shapes=deform_inputs2[1],
                           level_start_index=deform_inputs2[2], H=H, W=W,
                           )
        if self.extra_extractors is not None:
            for extractor in self.extra_extractors:
                c = extractor(query=c, reference_points=deform_inputs2[0],
                              feat=x, spatial_shapes=deform_inputs2[1],
                              level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, layer_ratio_loss, has_loss, selectors, prev_x, prev_selector, selector_soft


@MODELS.register_module()
class InteractionBlockVideoSelection(InteractionBlock):
    def __init__(self,
                 skip_ids=list(), select_module_ids=list(), map_select_module_ids=list(),
                 select_mod_keep_ratios=list(), map_select_mod_keep_ratios=list(), map_select_mod_ref_idx=list(),
                 prune_injector=False, prune_extractor=False, prune_x_by_pool_replacement=False,
                 **kwargs):
        kwargs['num_heads'] = kwargs.pop('deform_num_heads')
        super(InteractionBlockVideoSelection, self).__init__(**kwargs)

        self.skip_ids = skip_ids
        self.prune_injector = prune_injector
        self.prune_extractor = prune_extractor
        self.prune_x_by_pool_replacement = prune_x_by_pool_replacement
        self.select_mod_keep_ratios = select_mod_keep_ratios
        self.map_select_mod_keep_ratios = map_select_mod_keep_ratios

        self.select_module_ids = select_module_ids
        self.selec_moduleId_to_pos = {x: i for i, x in enumerate(select_module_ids)}
        self.map_select_module_ids = map_select_module_ids
        self.map_selec_moduleId_to_pos = {x: i for i, x in enumerate(map_select_module_ids)}
        self.map_select_mod_ref_idx = map_select_mod_ref_idx

    def _batchwise_ratio_loss(self, selector: torch.Tensor, ratio=1.):
        return ((selector.sum() / (selector.shape[0] * selector.shape[1])) - ratio) ** 2

    def _samplewise_ratio_loss(self, selector: torch.Tensor, ratio=1.):
        return ((selector.mean(dim=(1)) - ratio) ** 2).mean()

    def scale_mask_x_to_c(self, x_mask, H, W, to_bool=False):
        B = x_mask.shape[0]
        mask_big = x_mask.reshape(B, H, W).repeat_interleave(2, dim=1).repeat_interleave(2, dim=2).reshape(B, -1, 1)
        mask_med = x_mask
        mask_small = torch.max_pool2d(x_mask.reshape(B, 1, H, W), kernel_size=2, stride=2).reshape(B, -1, 1)

        if to_bool:
            mask_c_b2s_list = [mask_big.bool().squeeze(), mask_med.bool().squeeze(), mask_small.bool().squeeze()]
            mask_c_b2s = torch.cat(mask_c_b2s_list, dim=0)
        else:
            mask_c_b2s_list = [mask_big, mask_med, mask_small]
            mask_c_b2s = torch.cat(mask_c_b2s_list, dim=1)
        return mask_c_b2s, mask_c_b2s_list

    def _get_mask_loss(self, diff_selector, selector_soft, selec_gt_p, H, W):
        bs, num_patches, _ = diff_selector.shape

        gt_mask = F.max_pool2d(selec_gt_p, kernel_size=16, stride=16).detach()
        gt_mask = gt_mask.flatten(2).permute(0, 2, 1)  # [bs, H*W, 1]

        if False:  # if mask is to sparse
            val, idx = torch.sort(selector_soft.mean(dim=2, keepdim=True) * diff_selector, dim=1, descending=True)
            sparsity_condition = torch.linspace(1, 0, steps=num_patches, device=diff_selector.device)
            sparsity_condition = sparsity_condition.unsqueeze(0).expand(bs, -1).unsqueeze(-1)
            min_coverage = diff_selector.mean(dim=(1, 2)) * 0.8
            sparsity_condition = (sparsity_condition >= (1 - min_coverage[:, None, None])).to(torch.float)
            sparsity_condition = sparsity_condition.detach()

            A = torch.gather((1 - diff_selector), 1, idx)
            B = torch.gather(gt_mask, 1, idx)
            mask_loss_FN = (1 - A) * B * sparsity_condition
            mask_loss_TP = selector_soft * gt_mask * sparsity_condition
            mask_loss = torch.pow(mask_loss_FN, 2).mean() - torch.pow(mask_loss_TP, 2).mean()
        else:
            mask_loss_FN = (1 - diff_selector) * gt_mask
            # mask_loss_TP = selector_soft * gt_mask
            mask_loss = torch.pow(mask_loss_FN, 2).mean()  # - torch.pow(mask_loss_TP, 2).mean()

        return mask_loss

    def _get_selector_mask(self, x, prev_x, prev_selector, map_selective_modules, selective_modules, layer_idx, H, W,
                           interm_losses, selec_gt_p=None):
        selector, diff_selector, selector_soft = None, None, None
        reduce_feature_size = False

        # apply map-selective module
        # if layer_idx in self.map_select_module_ids and prev_x['x_11'] is not None:
        if layer_idx in self.map_select_module_ids and prev_x['x'][11] is not None:  # MAP-SM
            j = self.map_selec_moduleId_to_pos[layer_idx]
            ref_sim_x = prev_x['x'][layer_idx]
            ref_map_x = prev_x['x'][self.map_select_mod_ref_idx[j]]
            ref_map_selec = prev_x['selector'][self.map_select_mod_ref_idx[j]]
            keep_rate = self.map_select_mod_keep_ratios[j]

            selector, diff_selector, selector_soft = map_selective_modules[j](
                x, ref_sim_x, ref_map_x, ref_map_selec, prev_selector, keep_rate, H, W)

            if self.training:
                # sparsity loss
                if prev_selector is not None:
                    selector_soft = selector_soft * prev_selector
                    num_patches = prev_selector.sum(dim=1)
                    ratio_loss = ((selector_soft.sum(dim=1) / num_patches.detach()) - 0.5) ** 2
                    ratio_loss = ratio_loss.sum().unsqueeze(0)
                else:
                    ratio_loss = self._samplewise_ratio_loss(selector_soft, 0.5).unsqueeze(0)
                ratio_loss_scale = map_selective_modules[j].ratio_loss_scale
                interm_losses['loss_map_sm'] = interm_losses.get('loss_map_sm', []) + [ratio_loss_scale * ratio_loss]

                # mask _loss
                if selec_gt_p is not None:
                    mask_loss = self._get_mask_loss(diff_selector, selector_soft, selec_gt_p, H, W)
                    mask_loss_scale = map_selective_modules[j].mask_loss_scale
                    interm_losses['loss_fg_map_sm'] = interm_losses.get('loss_fg_map_sm', []) + [
                        mask_loss_scale * mask_loss.unsqueeze(0)]
            reduce_feature_size = True
        # apply selective module
        elif layer_idx in self.select_module_ids:  # call selective module
            j = self.selec_moduleId_to_pos[layer_idx]
            keep_rate = self.select_mod_keep_ratios[j]
            selector, diff_selector, selector_soft = selective_modules[j](x, keep_rate, H, W, prev_selector)

            if self.training:

                if selective_modules[j].prune_dynamically:
                    # loss_scale = selective_modules[j].loss_scale
                    ratio_loss = self._batchwise_ratio_loss(diff_selector, self.select_mod_keep_ratios[j]).unsqueeze(0)
                    # interm_losses['loss_sm'].append(loss_scale * ratio_loss)
                else:
                    if prev_selector is not None:
                        selector_soft = selector_soft * prev_selector
                        num_patches = prev_selector.sum(dim=1)
                        ratio_loss = ((selector_soft.sum(dim=1) / num_patches.detach()) - 0.5) ** 2
                        ratio_loss = ratio_loss.sum().unsqueeze(0)
                    else:
                        ratio_loss = self._samplewise_ratio_loss(selector_soft, 0.5).unsqueeze(0)
                ratio_loss_scale = selective_modules[j].ratio_loss_scale
                interm_losses['loss_sm'] = interm_losses.get('loss_sm', []) + [ratio_loss_scale * ratio_loss]

                # mask _loss
                if selec_gt_p is not None:
                    mask_loss = self._get_mask_loss(diff_selector, selector_soft, selec_gt_p, H, W)
                    mask_loss_scale = selective_modules[j].mask_loss_scale
                    interm_losses['loss_fg_sm'] = interm_losses.get('loss_fg_sm', []) + [
                        mask_loss_scale * mask_loss.unsqueeze(0)]
            reduce_feature_size = True
        # apply without pruning
        elif diff_selector is None:
            diff_selector = torch.ones_like(x[:, :, 0:1], device=x.device)
            selector = diff_selector.bool().squeeze(2)
            selector_soft = diff_selector.softmax(dim=-1)

        # avoid reavitvation
        if prev_selector is not None:
            diff_selector = diff_selector * prev_selector
            selector = diff_selector.bool().squeeze(2)
            selector_soft = selector_soft * prev_selector

        # if prev_selector is not None and selector_soft.shape[2] > 1:
        #    test01 = selector_soft[prev_selector.squeeze(2).to(torch.bool)].mean(dim=0)
        #    print(test01)
        #    y=1

        return selector, diff_selector, selector_soft, interm_losses, reduce_feature_size

    def _apply_pool_pruning(self, x, c, diff_selector_x, diff_selector_list_c, spatial_shapes_c, H, W):
        bs, num_p, emb_dim = x.shape

        x_ = (x * diff_selector_x).reshape(bs, H, W, emb_dim).permute(0, 3, 1, 2)  # B,C,H,W
        pool_x_ = F.max_pool2d(x_, kernel_size=5, stride=1, dilation=1, padding=2)
        pool_x_ = pool_x_.permute(0, 2, 3, 1).reshape(bs, num_p, emb_dim)
        x = x * diff_selector_x + pool_x_ * (1 - diff_selector_x)

        '''
        c_list = c.split([H_ * W_ for H_, W_ in spatial_shapes_c], dim=1)
        new_c_ = []
        for level, (H_, W_) in enumerate(spatial_shapes_c):
            diff_selec_c_ = diff_selector_list_c[level]
            c_ = c_list[level]
            c_l_ = (c_ * diff_selec_c_).reshape(bs, H_, W_, emb_dim).permute(0, 3, 1, 2)  # B,C,H,W

            if level == 2:
                pool_c_l_ = F.pad(c_l_, (2, 2, 2, 2), mode='constant')
                pool_c_l_ = F.max_pool2d(pool_c_l_, kernel_size=5, stride=1, dilation=2, padding=2)
            else:
                pool_c_l_ = F.max_pool2d(c_l_, kernel_size=5, stride=1, dilation=1, padding=2)
            pool_c_l_ = pool_c_l_.permute(0, 2, 3, 1).reshape(bs, -1, emb_dim)
            c_ = c_ * diff_selec_c_ + pool_c_l_ * (1 - diff_selec_c_)
            new_c_.append(c_)
        c = torch.cat(new_c_, dim=1)
        '''

        return x, c

    def forward(self, x, c, indexes, deform_inputs1, deform_inputs2, H, W, blks, selective_modules,
                map_selective_modules,
                prev_x, selec_gt_p=None, prev_selector=None, interm_losses=None, compute_dense=False):

        if self.prune_injector and prev_selector is not None and not compute_dense:
            if self.training:
                diff_selec_x = prev_selector  # .bool().squeeze()
                diff_selec_c, diff_selec_c_list = self.scale_mask_x_to_c(prev_selector, H, W)
                x = self.injector(query=x, reference_points=deform_inputs1[0],
                                  feat=c, spatial_shapes=deform_inputs1[1],
                                  level_start_index=deform_inputs1[2],
                                  mask_query=diff_selec_x, mask_value=diff_selec_c)
            else:
                # prev_selector[0,0:100,:] = 0.0  # for test purpose
                diff_selec_x = prev_selector.bool().squeeze()
                diff_selec_c, diff_selec_c_list = self.scale_mask_x_to_c(prev_selector, H, W, to_bool=True)
                x_ = x[:, diff_selec_x]
                c_ = c[:, diff_selec_c]
                ref_point_ = deform_inputs1[0][:, diff_selec_x]
                x_ = self.injector(query=x_, reference_points=ref_point_,
                                   feat=c_, spatial_shapes=deform_inputs1[1],
                                   level_start_index=deform_inputs1[2],
                                   mask_query=diff_selec_x, mask_value=diff_selec_c_list)
                x[:, diff_selec_x] = x_
        else:
            x = self.injector(query=x, reference_points=deform_inputs1[0],
                              feat=c, spatial_shapes=deform_inputs1[1],
                              level_start_index=deform_inputs1[2])

        selector_log = []
        selector_soft_log = []
        for i in range(indexes[0], indexes[-1] + 1):
            if i in self.skip_ids or compute_dense:
                x = blks[i](x, x_dense=x)
                prev_x['x'][i] = x
                prev_x['selector'][i] = None
                prev_x['diff_selector'][i] = None

                logging_fake_mask = torch.ones_like(x[:, :, 0:1]).to(torch.float).permute(0, 2, 1)
                selector_log.append(logging_fake_mask)
                selector_soft_log.append(logging_fake_mask)
            else:
                selector, diff_selector, selector_soft, interm_losses, reduce_feature_size = self._get_selector_mask(x,
                                                                                                                     prev_x,
                                                                                                                     prev_selector,
                                                                                                                     map_selective_modules,
                                                                                                                     selective_modules,
                                                                                                                     i,
                                                                                                                     H,
                                                                                                                     W,
                                                                                                                     interm_losses,
                                                                                                                     selec_gt_p=selec_gt_p)

                prev_selector = diff_selector

                if self.prune_x_by_pool_replacement and reduce_feature_size:
                    bs, num_p, emb_dim = x.shape
                    diff_selec_c, diff_selec_c_list = self.scale_mask_x_to_c(prev_selector, H, W)
                    x, c = self._apply_pool_pruning(x, c, diff_selector, diff_selec_c_list, deform_inputs1[1], H, W)

                # if i == self.matching_module_idx:
                #    prev_x['x_0'] = x

                if self.training:
                    x = (diff_selector * blks[i](x, src_key_padding_mask=~selector) + (1 - diff_selector) * x)
                else:
                    assert x.shape[0] == 1, 'In testmode batch_size must be 1'
                    selector_sort = torch.argsort(selector.int(), dim=1, descending=True)
                    real_indices = selector_sort[:, :selector.sum(1)].unsqueeze(-1).expand(-1, -1, x.shape[-1])
                    selected_x = torch.gather(x, 1, real_indices)
                    selected_x = blks[i](selected_x, x_dense=x, selec_mask=selector)
                    x.scatter_(1, real_indices, selected_x)

                prev_x['x'][i] = x
                prev_x['selector'][i] = selector
                prev_x['diff_selector'][i] = diff_selector

                selector_log.append(diff_selector.permute(0, 2, 1).detach())  # [bs, 1, n_patch]
                # selector_soft_log.append(selector_soft.mean(dim=2, keepdim=True).permute(0, 2, 1).detach())    #[bs, n_patch, 1]
                selector_soft_log.append(selector_soft[:, :, 0:1].permute(0, 2, 1).detach())  # [bs, n_patch, 1]

        if self.prune_extractor and prev_selector is not None and not compute_dense:
            if self.training:
                diff_selec_x = prev_selector  # .bool().squeeze()
                diff_selec_c, diff_selec_c_list = self.scale_mask_x_to_c(prev_selector, H, W)
                c = self.extractor(query=c, reference_points=deform_inputs2[0],
                                   feat=x, spatial_shapes=deform_inputs2[1],
                                   level_start_index=deform_inputs2[2], H=H, W=W,
                                   mask_query=diff_selec_c, mask_value=diff_selec_x,
                                   )
                if self.extra_extractors is not None:
                    for extractor in self.extra_extractors:
                        c = extractor(query=c, reference_points=deform_inputs2[0],
                                      feat=x, spatial_shapes=deform_inputs2[1],
                                      level_start_index=deform_inputs2[2], H=H, W=W,
                                      mask_query=diff_selec_c, mask_value=diff_selec_x,
                                      )
            else:
                diff_selec_x = prev_selector.bool().squeeze()
                diff_selec_c, diff_selec_c_list = self.scale_mask_x_to_c(prev_selector, H, W, to_bool=True)
                c_ = c[:, diff_selec_c]
                x_ = x[:, diff_selec_x]
                ref_point_ = deform_inputs2[0][:, diff_selec_c]
                c_ = self.extractor(query=c_, reference_points=ref_point_,
                                    feat=x_, spatial_shapes=deform_inputs2[1],
                                    level_start_index=deform_inputs2[2], H=H, W=W,
                                    mask_query=diff_selec_c, mask_value=[diff_selec_x],
                                    )
                if self.extra_extractors is not None:
                    for extractor in self.extra_extractors:
                        c_ = extractor(query=c_, reference_points=ref_point_,
                                       feat=x_, spatial_shapes=deform_inputs2[1],
                                       level_start_index=deform_inputs2[2], H=H, W=W,
                                       mask_query=diff_selec_c, mask_value=[diff_selec_x],
                                       )
                c[:, diff_selec_c] = c_


        else:
            c = self.extractor(query=c, reference_points=deform_inputs2[0],
                               feat=x, spatial_shapes=deform_inputs2[1],
                               level_start_index=deform_inputs2[2], H=H, W=W,
                               )
            if self.extra_extractors is not None:
                for extractor in self.extra_extractors:
                    c = extractor(query=c, reference_points=deform_inputs2[0],
                                  feat=x, spatial_shapes=deform_inputs2[1],
                                  level_start_index=deform_inputs2[2], H=H, W=W)
        return x, c, interm_losses, selector_log, prev_x, prev_selector, selector_soft_log


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
