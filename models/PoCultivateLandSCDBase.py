import torch.nn as nn
# from modeling.sseg.uperhead import UperNetHead
import torch.nn.functional as F
from models.sseg.uperhead import UperNetHead
# from models.utils.MultiScaleAttention import MultiScaleAttention
from functools import partial
import torch
from torch.optim import lr_scheduler
# from functools import partial
# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import torch
import torch.nn as nn
from collections import OrderedDict
import torch.utils.checkpoint as checkpoint
from timm.models.layers import trunc_normal_, DropPath
# from mmcv.runner import _load_checkpoint
# from mmcv.cnn import constant_init, trunc_normal_init
# from mmseg.utils import get_root_logger
# from mmseg.models.builder import BACKBONES
import torch.nn.functional as F

from models.ops_dcnv3 import modules as dcnv3


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')

class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_last')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class DownsampleLayer(nn.Module):
    r""" Downsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """

    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.conv = nn.Conv2d(channels,
                              2 * channels,
                              kernel_size=3,
                              stride=2,
                              padding=1,
                              bias=False)
        self.norm = build_norm_layer(2 * channels, norm_layer,
                                     'channels_first', 'channels_last')

    def forward(self, x):
        x = self.conv(x.permute(0, 3, 1, 2))
        x = self.norm(x)  # [B, H // 8, W // 8, 256]
        return x

class UpsampleLayer(nn.Module):
    r""" Upsample layer of InternImage
    Args:
        channels (int): number of input channels
        norm_layer (str): normalization layer
    """
    def __init__(self, channels, norm_layer='LN'):
        super().__init__()
        self.transposed_conv = nn.ConvTranspose2d(in_channels=channels, out_channels=channels // 2, kernel_size=2, stride=2, padding=0)
        self.norm = build_norm_layer(channels // 2, norm_layer,
                                     'channels_first', 'channels_last')
        self.act = nn.LeakyReLU(0.02, inplace=True)

    def forward(self, x):
        x = self.act(self.transposed_conv(x.permute(0, 3, 1, 2)))
        x = self.norm(x)  # [B, H // 8, W // 8, 256]
        return x

class MLPLayer(nn.Module):
    r""" MLP layer of InternImage
    Args:
        in_features (int): number of input features
        hidden_features (int): number of hidden features
        out_features (int): number of output features
        act_layer (str): activation layer
        drop (float): dropout rate
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer='GELU',
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_act_layer(act_layer)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class InternImageLayer(nn.Module):
    r""" Basic layer of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 groups,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 layer_scale=None,
                 offset_scale=1.0,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False):#,
                 # use_dcn_v4_op=False):  # for InternImage-H/G
        super().__init__()
        self.channels = channels
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(channels, 'LN')
        self.post_norm = post_norm
        self.dcn = core_op(
            channels=channels,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=groups,
            offset_scale=offset_scale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
            center_feature_scale=center_feature_scale)#,
            #use_dcn_v4_op=use_dcn_v4_op)  # for InternImage-H/G
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.norm2 = build_norm_layer(channels, 'LN')
        # self.mlp = MLPLayer(in_features=channels,
        #                     hidden_features=int(channels * mlp_ratio),
        #                     act_layer=act_layer,
        #                     drop=drop)
        self.layer_scale = layer_scale is not None
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(channels),
                                       requires_grad=True)
        self.res_post_norm = res_post_norm
        if res_post_norm:
            self.res_post_norm1 = build_norm_layer(channels, 'LN')
            self.res_post_norm2 = build_norm_layer(channels, 'LN')

    def forward(self, x):

        def _inner_forward(x):
            if not self.layer_scale:
                if self.post_norm:
                    x = x + self.drop_path(self.norm1(self.dcn(x)))
                    # x = x + self.drop_path(self.norm2(self.mlp(x)))
                elif self.res_post_norm:  # for InternImage-H/G
                    x = x + self.drop_path(self.res_post_norm1(self.dcn(self.norm1(x))))
                    # x = x + self.drop_path(self.res_post_norm2(self.mlp(self.norm2(x))))
                else:
                    x = x + self.drop_path(self.dcn(self.norm1(x)))
                    # x = x + self.drop_path(self.mlp(self.norm2(x)))
                return x
            if self.post_norm:
                x = x + self.drop_path(self.gamma1 * self.norm1(self.dcn(x)))
                # x = x + self.drop_path(self.gamma2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma1 * self.dcn(self.norm1(x)))
                # x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
            return x

        if self.with_cp and x.requires_grad:
            x = checkpoint.checkpoint(_inner_forward, x)
        else:  # [2, 64, 64, 128]
            x = _inner_forward(x)  #
        return x


class InternImageBlock(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 downsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False):#,  # for InternImage-H/G
                 #use_dcn_v4_op=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale#,  # for InternImage-H/G
                #use_dcn_v4_op=use_dcn_v4_op
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:  # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.downsample = DownsampleLayer(
            channels=channels, norm_layer=norm_layer) if downsample else None

    def forward(self, x, return_wo_downsample=False):  # x:[B, H // 4, W // 4, 128]
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # x:[B, H // 4, W // 4, 128]
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)  # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)  # x:[B, H // 4, W // 4, 128]
        if return_wo_downsample:
            x_ = x  # x:[B, H // 4, W // 4, 128]
        if self.downsample is not None:
            x = self.downsample(x)  # x:[B, H // 8, W // 8, 256]

        if return_wo_downsample:
            return x, x_
        return x

class InternImageBlock_up(nn.Module):
    r""" Block of InternImage
    Args:
        core_op (nn.Module): core operation of InternImage
        channels (int): number of input channels
        depths (list): Depth of each block.
        groups (list): Groups of each block.
        mlp_ratio (float): ratio of mlp hidden features to input channels
        drop (float): dropout rate
        drop_path (float): drop path rate
        act_layer (str): activation layer
        norm_layer (str): normalization layer
        post_norm (bool): whether to use post normalization
        layer_scale (float): layer scale
        offset_scale (float): offset scale
        with_cp (bool): whether to use checkpoint
    """

    def __init__(self,
                 core_op,
                 channels,
                 depth,
                 groups,
                 upsample=True,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer='GELU',
                 norm_layer='LN',
                 post_norm=False,
                 offset_scale=1.0,
                 layer_scale=None,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False):#,  # for InternImage-H/G
                 #use_dcn_v4_op=False):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.post_norm = post_norm
        self.center_feature_scale = center_feature_scale

        self.blocks = nn.ModuleList([
            InternImageLayer(
                core_op=core_op,
                channels=channels,
                groups=groups,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale#,  # for InternImage-H/G
                #use_dcn_v4_op=use_dcn_v4_op
            ) for i in range(depth)
        ])
        if not self.post_norm or center_feature_scale:
            self.norm = build_norm_layer(channels, 'LN')
        self.post_norm_block_ids = post_norm_block_ids
        if post_norm_block_ids is not None:  # for InternImage-H/G
            self.post_norms = nn.ModuleList(
                [build_norm_layer(channels, 'LN', eps=1e-6) for _ in post_norm_block_ids]
            )
        self.upsample = UpsampleLayer(
            channels=channels, norm_layer=norm_layer) if upsample else None

    def forward(self, x, return_wo_upsample=False):  # x:[B, H // 4, W // 4, 128]
        for i, blk in enumerate(self.blocks):
            x = blk(x)  # x:[B, H // 4, W // 4, 128]
            if (self.post_norm_block_ids is not None) and (i in self.post_norm_block_ids):
                index = self.post_norm_block_ids.index(i)
                x = self.post_norms[index](x)  # for InternImage-H/G
        if not self.post_norm or self.center_feature_scale:
            x = self.norm(x)  # x:[B, H // 4, W // 4, 128]
        if return_wo_upsample:
            x_ = x  # x:[B, H // 4, W // 4, 128]
        if self.upsample is not None:
            x = self.upsample(x)  # x:[B, H // 8, W // 8, 256]

        if return_wo_upsample:
            return x, x_
        return x

# @BACKBONES.register_module()
class InternImageSTUNet(nn.Module):
    r""" InternImage
        A PyTorch impl of : `InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        core_op (str): Core operator. Default: 'DCNv3'
        channels (int): Number of the first stage. Default: 64
        depths (list): Depth of each block. Default: [3, 4, 18, 5]
        groups (list): Groups of each block. Default: [3, 6, 12, 24]
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        drop_rate (float): Probability of an element to be zeroed. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        act_layer (str): Activation layer. Default: 'GELU'
        norm_layer (str): Normalization layer. Default: 'LN'
        layer_scale (bool): Whether to use layer scale. Default: False
        cls_scale (bool): Whether to use class scale. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
        dw_kernel_size (int): Size of the dwconv. Default: None
        level2_post_norm (bool): Whether to use level2 post norm. Default: False
        level2_post_norm_block_ids (list): Indexes of post norm blocks. Default: None
        res_post_norm (bool): Whether to use res post norm. Default: False
        center_feature_scale (bool): Whether to use center feature scale. Default: False
    """

    def __init__(self,
                 core_op='DCNv3',
                 channels=128,
                 depths=[3, 4, 18, 5],
                 groups=[4, 8, 16, 32],  # [3, 6, 12, 24],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.2,
                 drop_path_type='linear',
                 act_layer='GELU',
                 norm_layer='LN',
                 layer_scale=None,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False,
                 dw_kernel_size=None,  # for InternImage-H/G
                 level2_post_norm=False,  # for InternImage-H/G
                 level2_post_norm_block_ids=None,  # for InternImage-H/G
                 res_post_norm=False,  # for InternImage-H/G
                 center_feature_scale=False,  # for InternImage-H/G
                 num_class=5,
                 #use_dcn_v4_op=False,
                 out_indices=(0, 1, 2, 3),
                 num_expert=1,
                 init_cfg=None,
                 **kwargs):
        super().__init__()
        self.core_op = core_op
        self.num_levels = len(depths)
        self.depths = depths
        self.channels = channels
        self.num_features = int(channels * 2 ** (self.num_levels - 1))
        self.post_norm = post_norm
        self.mlp_ratio = mlp_ratio
        self.init_cfg = init_cfg
        self.out_indices = out_indices
        self.level2_post_norm_block_ids = level2_post_norm_block_ids
        self.num_expert = num_expert

        in_chans = 3
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=channels,
                                     act_layer=act_layer,
                                     norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        if drop_path_type == 'uniform':
            for i in range(len(dpr)):
                dpr[i] = drop_path_rate

        self.levels = nn.ModuleList()
        for i in range(self.num_levels):
            # concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            # int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()

            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                    i == 2) else None  # for InternImage-H/G
            level = InternImageBlock(
                core_op=getattr(dcnv3, core_op),
                channels=int(channels * 2 ** i),
                depth=depths[i],
                groups=groups[i],
                mlp_ratio=self.mlp_ratio,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                act_layer=act_layer,
                norm_layer=norm_layer,
                post_norm=post_norm,
                downsample=(i < self.num_levels - 1),
                layer_scale=layer_scale,
                offset_scale=offset_scale,
                with_cp=with_cp,
                dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                post_norm_block_ids=post_norm_block_ids,  # for InternImage-H/G
                res_post_norm=res_post_norm,  # for InternImage-H/G
                center_feature_scale=center_feature_scale,  # for InternImage-H/G
                #use_dcn_v4_op=use_dcn_v4_op,
            )
            self.levels.append(level)

        self.levels_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i in range(self.num_levels):
            concat_conv = nn.Conv2d(int(channels * 2 ** (4-i)) * 2, int(channels * 2 ** (4-i)), kernel_size=1, stride=1, padding=0)

            post_norm_block_ids = level2_post_norm_block_ids if level2_post_norm and (
                    i == 2) else None  # for InternImage-H/G
            if i ==0 :
                level_up = InternImageBlock_up(
                    core_op=getattr(dcnv3, core_op),
                    channels=int(channels * 2 ** (self.num_levels - 1 - i)),
                    depth=depths[i],
                    groups=groups[i],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    post_norm=post_norm,
                    upsample=None,
                    layer_scale=layer_scale,
                    offset_scale=offset_scale,
                    with_cp=with_cp,
                    dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                    post_norm_block_ids=post_norm_block_ids,  # for InternImage-H/G
                    res_post_norm=res_post_norm,  # for InternImage-H/G
                    center_feature_scale=center_feature_scale,  # for InternImage-H/G
                    #use_dcn_v4_op=use_dcn_v4_op,
                )
            else:
                level_up = InternImageBlock_up(
                    core_op=getattr(dcnv3, core_op),
                    channels=int(channels * 2 ** (4 - i)),# int(channels * 2 ** (self.num_levels- 1 - i)),
                    depth=depths[i],
                    groups=groups[i],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    post_norm=post_norm,
                    upsample=True,
                    layer_scale=layer_scale,
                    offset_scale=offset_scale,
                    with_cp=with_cp,
                    dw_kernel_size=dw_kernel_size,  # for InternImage-H/G
                    post_norm_block_ids=post_norm_block_ids,  # for InternImage-H/G
                    res_post_norm=res_post_norm,  # for InternImage-H/G
                    center_feature_scale=center_feature_scale,  # for InternImage-H/G
                    #use_dcn_v4_op=use_dcn_v4_op,
                )
            self.levels_up.append(level_up)
            self.concat_back_dim.append(concat_conv)

        # self.conv_compress = nn.Conv2d(in_channels=channels*8, out_channels=channels*2, kernel_size=1, stride=1)

        self.num_layers = len(depths)
        # self.MultiScaleAttention = MultiScaleAttention(patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        # )

        self.MultiScaleAttention_layer = nn.ModuleList()

        # self.MultiScaleAttention = MultiScaleAttention(patch_size=4, embed_dims=[channels, channels*2, channels*4, channels*8], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        #                                             norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,)
        # for i in range(self.num_levels):
        #     self.MultiScaleAttention = MultiScaleAttention(patch_size=4, embed_dims=[channels, channels*2, channels*4, channels*8], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1])
        #     self.MultiScaleAttention_layer.append(self.MultiScaleAttention)

        if num_expert == 1:
            self.classifyhead = nn.Sequential(
                nn.Conv2d(in_channels=channels, out_channels=num_class, kernel_size=1, stride=1),
                nn.GELU(),
                nn.BatchNorm2d(num_class)
            )
        else:
            self.classifyhead_list = nn.ModuleList(
                    [nn.Sequential(
                    nn.Conv2d(in_channels=channels, out_channels=num_class, kernel_size=1, stride=1),
                    nn.GELU(),
                    nn.BatchNorm2d(num_class)
                ) for _ in
                     range(num_expert)])

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)  # [B, 64, 64, 128]

        seq_out = []
        for level_idx, level in enumerate(self.levels):
            # x, x_ = level(x, return_wo_downsample=True)  # x: [B, 32, 32, 256] x_:[B, 64, 64, 128] x: [B, 16, 16, 512] x_:[B, 32, 32, 256] x:[B, 8, 8, 1024] x_:[B, 16, 16, 512] x:[B, 8, 8, 1024] x_:[B, 8, 8, 1024]
            x, x_downsample = level(x, return_wo_downsample=True)
            if level_idx in self.out_indices:
                seq_out.append(x_downsample.permute(0, 3, 1, 2).contiguous())
                # seq_out.append(x_.permute(0, 3, 1, 2).contiguous())
        return x, seq_out

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.levels_up):
            if inx == 0:
                x, _ = layer_up(x, return_wo_upsample=True)  # [16, 8, 8, 1024]
                # x = self.conv_compress(x.permute(0, 3, 1, 2))
                x = self.MultiScaleAttention_layer[inx](x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                # print('x shape is', x.shape)  # [2, 8, 8, 1024]
                # print('x_downsample[4 - inx] shape is', x_downsample[4 - inx].shape)
                # print('inx is ', inx)

                x_down = self.MultiScaleAttention(x_downsample[4 - inx])
                x_down = x_downsample[4-inx]
                # x_down = self.MultiScaleAttention_layer[inx](x_downsample[4 - inx])
                x = torch.cat([x, x_down.permute(0, 2, 3, 1)], 3)  # 断在这  # [2, 8, 8, 2048]
                x = self.concat_back_dim[inx](x.permute(0, 3, 1, 2))  # [16, 1024, 8, 8] [16, 512, 16, 16] [16, 256, 32, 32]
                x, _ = layer_up(x.permute(0, 2, 3, 1), return_wo_upsample=True)  # [16, 16, 16, 512] [16, 32, 32, 245] [16, 64, 64, 128]

        # x = self.norm_up(x)  # B L C [2, 4096, 96]

        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)  # x: [B, H // 32, W // 32, 1024], X_downsample: [B, 128, 64, 64] [B, 256, 32, 32] [B, 512, 16, 16] [B, 1024, 8, 8]
        x = self.forward_up_features(x, x_downsample)
        x = x.permute(0, 3, 1, 2)
        outputs = []
        if self.num_expert == 1:
            x = self.classifyhead(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
            return  x
        else:
            for i in range(self.num_expert):
                x_ = self.classifyhead_list[i](x)
                x_ = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=True)
                x_ = F.interpolate(x_, scale_factor=2, mode='bilinear', align_corners=True)
                outputs.append(x_)
            final_out = torch.stack(outputs, dim=1).mean(dim=1)
            return {
                "output": final_out,
                # "feat": torch.stack(self.feat, dim=1),
                "logits": torch.stack(outputs, dim=0)
            }

def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        args (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler

# class InternImageSTUNet(nn.Module):
#     def __init__(self, n_class=6, inchannels=[128, 256, 512, 1024],
#                  _channels=256, num_expert=3):
#         super().__init__()
#         self.encoder = InternImage()
#         self.in_channels = inchannels
#         self.n_class = n_class
#         self.channels = _channels
#         # self.decoder = UperNetHead(num_classes = n_class, in_channels=in_channels, channels=channels)
#         self.num_expert = num_expert
#         self.MultiScaleAttention = MultiScaleAttention(patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
#         )
#
#     def forward(self, x):
#         H, W = x.size(2), x.size(3)
#         x_list = self.encoder(x)  # [2, 128,64, 64] [2, 256,32, 32]  [2, 512,16, 16] [2, 1024,8, 8]
#
#         # return x
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_test = torch.randn(2, 3, 256, 256).to(device)
    # model = UpsampleLayer(channels=256)
    model = InternImageSTUNet().to(device)
    # from thop import profile
    # FLops, params = profile(model, inputs=(tensor_test,))
    # output = model(tensor_test)
    # print(model(tensor_test)[0].shape)
    # print('Flops: % .4fG' % (FLops / 1000000000))
    # print('param参数量: %.4fM' % (params / 1000000))
    # print(model(tensor_test).shape)

    # from  thop import profile
    # Flops, params = profile(model, inputs=(tensor_test,)) # macs
    # print('Flops: % .4fG'%(Flops / 1000000000))# 计算量  8.54386 GFlops
    # print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值

    # import numpy as np
    #
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 300
    # timings = np.zeros((repetitions, 1))
    # # GPU-WARM-UP
    # for _ in range(10):
    #     _ = model(tensor_test)
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = model(tensor_test)
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # mean_fps = 1000. / mean_syn
    # print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
    #                                                                                      std_syn=std_syn,
    #                                                                                      mean_fps=mean_fps))
    # print(mean_syn)

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #
    # iterations = 10
    # # GPU预热
    # for _ in range(50):
    #     _ = model(tensor_test)
    #
    # # 测速
    # times = torch.zeros(iterations)  # 存储每轮iteration的时间
    # with torch.no_grad():
    #     for iter in range(iterations):
    #         starter.record()
    #         _ = model(tensor_test)
    #         ender.record()
    #         # 同步GPU时间
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)  # 计算时间
    #         times[iter] = curr_time
    #         # print(curr_time)
    #
    # mean_time = times.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
    #
    # flops, params = profile(model, (tensor_test,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))