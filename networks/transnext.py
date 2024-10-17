import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
import pkg_resources


def is_installed(package_name):
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except pkg_resources.DistributionNotFound:
        return False


if is_installed('swattention'):
    print('swattention package found, loading CUDA version of Aggregated Attention')
    from networks.attention_cuda import AggregatedAttention
else:
    print('swattention package not found, loading PyTorch native version of Aggregated Attention')
    from networks.attention_native import AggregatedAttention

import torch
import torch.nn as nn
import einops
from timm.models.layers import trunc_normal_


class SlideAttention(nn.Module):

    def __init__(
            self, dim, num_heads, ka, qkv_bias=True, qk_scale=None, attn_drop=0.,
            proj_drop=0., dim_reduction=4, rpb=True, padding_mode='zeros',
            share_dwc_kernel=True, share_qkv=False):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        if share_qkv:
            self.qkv_scale = 1
        else:
            self.qkv_scale = 3
        self.rpb = rpb
        self.share_dwc_kernel = share_dwc_kernel
        self.padding_mode = padding_mode
        self.share_qkv = share_qkv
        self.ka = ka
        self.dim_reduction = dim_reduction
        self.qkv = nn.Linear(dim, dim * self.qkv_scale // self.dim_reduction, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim // self.dim_reduction, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dep_conv = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                  self.ka * self.ka * dim // self.dim_reduction // self.num_heads, kernel_size=self.ka,
                                  bias=True, groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                  padding_mode=padding_mode)
        self.dep_conv1 = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                   self.ka * self.ka * dim // self.dim_reduction // self.num_heads, kernel_size=self.ka,
                                   bias=True, groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                   padding_mode=padding_mode)
        if not share_dwc_kernel:
            self.dep_conv2 = nn.Conv2d(dim // self.dim_reduction // self.num_heads,
                                       self.ka * self.ka * dim // self.dim_reduction // self.num_heads,
                                       kernel_size=self.ka, bias=True,
                                       groups=dim // self.dim_reduction // self.num_heads, padding=self.ka // 2,
                                       padding_mode=padding_mode)

        self.reset_parameters()

        # define a parameter table of relative position bias
        if self.rpb:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(1, self.num_heads, 1, self.ka * self.ka, 1, 1))
            trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=3)

    def reset_parameters(self):
        # shift initialization for group convolution
        kernel = torch.zeros(self.ka * self.ka, self.ka, self.ka)
        for i in range(self.ka * self.ka):
            kernel[i, i // self.ka, i % self.ka] = 1.
        kernel = kernel.unsqueeze(1).repeat(self.dim // self.dim_reduction // self.num_heads, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x,H22, W22, relative_pos_index, relative_coords_table):
        # print("xxxxxxxxxxx111111111", x.shape)
        B1, L1, C1 = x.shape
        H1 = int(L1**0.5)
        x = x.reshape(B1, C1, H1, H1)

        x = einops.rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape

        # print("xxxxxxxxxxx22222222", x.shape)
        qkv = self.qkv(x)

        f_conv = qkv.permute(0, 3, 1, 2).reshape(B * self.num_heads,
                                                 self.qkv_scale * C // self.dim_reduction // self.num_heads, H, W)

        if self.qkv_scale == 3:
            q = (f_conv[:, :C // self.dim_reduction // self.num_heads, :, :] * self.scale).reshape(B, self.num_heads,
                                                                                                   C // self.dim_reduction // self.num_heads,
                                                                                                   1, H, W)
            k = f_conv[:, C // self.dim_reduction // self.num_heads:2 * C // self.dim_reduction // self.num_heads, :,
                :]  # B*self.nhead, C//self.nhead, H, W
            v = f_conv[:, 2 * C // self.dim_reduction // self.num_heads:, :, :]
        elif self.qkv_scale == 1:
            q = (f_conv * self.scale).reshape(B, self.num_heads, C // self.dim_reduction // self.num_heads, 1, H, W)
            k = v = f_conv

        if self.share_dwc_kernel:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv1(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
        else:
            k = (self.dep_conv(k) + self.dep_conv1(k)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)
            v = (self.dep_conv(v) + self.dep_conv2(v)).reshape(B, self.num_heads,
                                                               C // self.dim_reduction // self.num_heads,
                                                               self.ka * self.ka, H, W)

        if self.rpb:
            k = k + self.relative_position_bias_table
        attn = (q * k).sum(2, keepdim=True)  # B, self.nhead, 1, k^2, H, W

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn * v).sum(3).reshape(B, C // self.dim_reduction, H, W).permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')
        x = x.reshape(B1, L1, C1)

        # print("xxxxxxxxxxx",x.shape)
        return x#, None, None





class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_size = pretrain_size or query_size
    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)
    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)
    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw)
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw)

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / (pretrain_size[0] - 1) * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / (pretrain_size[1] - 1) * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
        torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8, dtype=torch.float32))

    return idx_map, relative_coords_table


class Attention(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1))  # Initialize softplus(temperature) to 1/0.24.
        # Generate sequnce length scale
        self.register_buffer("seq_length_scale", torch.as_tensor(np.log(input_resolution[0] * input_resolution[1])),
                             persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # mlp to generate continuous relative position bias
        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        # Use MLP to generate continuous relative positional bias
        rel_bias = self.cpb_fc2(self.cpb_act(self.cpb_fc1(relative_coords_table))).transpose(0, 1)[:,
                   relative_pos_index.view(-1)].view(-1, N, N)

        # Calculate attention map using sequence length scaled cosine attention and query embedding
        attn = ((F.normalize(q, dim=-1) + self.query_embedding) * F.softplus(
            self.temperature) * self.seq_length_scale) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, fixed_pool_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if sr_ratio == 1:
            self.attn = SlideAttention(
                dim,
                num_heads=num_heads,
                ka=3,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop)
        else:
            self.attn = AggregatedAttention(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                fixed_pool_size=fixed_pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvolutionalGLU(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, relative_pos_index, relative_coords_table))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()

        patch_size = to_2tuple(patch_size)

        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W




class TransNeXt(nn.Module):
    '''
    The parameter "img size" is primarily utilized for generating relative spatial coordinates,
    which are used to compute continuous relative positional biases. As this TransNeXt implementation does not support multi-scale inputs,
    it is recommended to set the "img size" parameter to a value that is exactly the same as the resolution of the inference images.
    It is not advisable to set the "img size" parameter to a value exceeding 800x800.
    The "pretrain size" refers to the "img size" used during the initial pre-training phase,
    which is used to scale the relative spatial coordinates for better extrapolation by the MLP.
    For models trained on ImageNet-1K at a resolution of 224x224,
    as well as downstream task models fine-tuned based on these pre-trained weights,
    the "pretrain size" parameter should be set to 224x224.
    '''

    def __init__(self, img_size=224, pretrain_size=None, window_size=[3, 3, 3, None],
                 patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], num_stages=4, fixed_pool_size=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        pretrain_size = pretrain_size or img_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            # Generate relative positional coordinate table and index for each stage to compute continuous relative positional bias.
            relative_pos_index, relative_coords_table = get_relative_position_cpb(
                query_size=to_2tuple(img_size // (2 ** (i + 2))),
                key_size=to_2tuple(img_size // ((2 ** (i + 2)) * sr_ratios[i])) if (
                        fixed_pool_size is None or sr_ratios[i] == 1) else to_2tuple(fixed_pool_size),
                pretrain_size=to_2tuple(pretrain_size // (2 ** (i + 2))))

            self.register_buffer(f"relative_pos_index{i + 1}", relative_pos_index, persistent=False)
            self.register_buffer(f"relative_coords_table{i + 1}", relative_coords_table, persistent=False)

            patch_embed = OverlapPatchEmbed(patch_size=patch_size * 2 - 1 if i == 0 else 3,
                                            stride=patch_size if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], input_resolution=to_2tuple(img_size // (2 ** (i + 2))), window_size=window_size[i],
                num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], fixed_pool_size=fixed_pool_size)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        for n, m in self.named_modules():
            self._init_weights(m, n)

    def _init_weights(self, m: nn.Module, name: str = ''):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'query_embedding', 'relative_pos_bias_local', 'cpb', 'temperature'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        downsample = []
        downsample.append(x)
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            relative_pos_index = getattr(self, f"relative_pos_index{i + 1}")
            relative_coords_table = getattr(self, f"relative_coords_table{i + 1}")
            for blk in block:
                x = blk(x, H, W, relative_pos_index, relative_coords_table)
            x = norm(x)

            if i != (self.num_stages):
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            downsample.append(x)

        return x, downsample

    def forward(self, x):
        x, downsample = self.forward_features(x)

        return x, downsample



@register_model
def transnext_base(pretrained=False, num_classes=3, **kwargs):
    model = TransNeXt(window_size=[3, 3, 3, None],
                      patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[4, 8, 16, 32],
                      mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[5, 5, 5, 5], sr_ratios=[8, 4, 2, 1],
                      num_classes=num_classes,
                      **kwargs)
    model.default_cfg = _cfg()

    return model
