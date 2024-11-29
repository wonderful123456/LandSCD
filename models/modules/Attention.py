import torch
import torch.nn as nn
from sympy.physics.vector import outer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from models.modules.dctcam_advanced import MultiSpectralAttentionLayer

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape   # B: 2 C: 512 N: 4096
        x = x.transpose(1, 2).view(B, C, H, W)  # [2, 512, 64, 64]
        x = self.dwconv(x)  # [2, 512, 64, 64]
        x = x.flatten(2).transpose(1, 2)  # [2, 4096, 512] [2, 1024, 1024] [2, 256, 1024]

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)   # [2, 4096, 512]
        x = self.act(x + self.dwconv(x, H, W))  # [2, 4096, 512]
        x = self.drop(x)
        x = self.fc2(x)  # [2, 4096, 64] [2, 1024, 128] [2, 256, 256]
        x = self.drop(x)
        return x

class SpatialMultiAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads   # 2
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_s = nn.Linear(dim, dim // 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)


        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.act = nn.GELU()
            if sr_ratio==8:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==4:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm2 = nn.LayerNorm(dim)
            if sr_ratio==2:
                self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
                self.norm1 = nn.LayerNorm(dim)
                self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
                self.norm2 = nn.LayerNorm(dim)
            self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
            self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
            self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
        else:
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        length = 3
        if len(x.shape) == 4:
            length = 4
            x = x.view(-1, x.shape[1], H * W).permute(0, 2, 1)
        # H, W = int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))
        B, N, C = x.shape  # B: 2 C: 128 N: 4096
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # --[B, num_heads, N, C // num_heads]
        if self.sr_ratio > 1:# and self.sr_ratio != 8:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [2, 64, 64, 64]
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [2, N // sr_ratio ** 2, C]
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [2, N // (sr_ratio // 2) ** 2, C]
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, num_heads//2, N // sr_ratio ** 2, C // num_heads]
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, num_heads//2, N // (sr_ratio // 2) ** 2, C // num_heads]
            k1, v1 = kv1[0], kv1[1]  # B head N C  k1: [2, 1, 64, 32] v1: [2, 1, 64, 32]
            k2, v2 = kv2[0], kv2[1]  # k2: [2, 1, 256, 32] v1: [2, 1, 256, 32]
            q_ = q[:, :self.num_heads//2]  # --[B, self.num_heads//2, N, C // num_heads]
            k_ = k1.transpose(-2, -1)  # --[B,, num_heads//2, N // (sr_ratio // 2) ** 2, C // num_heads]
            attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale  # --[2, 4, 256, 64]
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).contiguous().reshape(B, -1, C//2).
                                    transpose(1, 2).contiguous().view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2).contiguous()  # v1: [B, num_heads//2, N // sr_ratio ** 2, C // num_heads]
            # v1_transposed = v1.transpose(1, 2).contiguous()
            # v1_reshaped = v1_transposed.reshape(B, -1, C // 2)
            #
            # v1 = v1 + self.local_conv1(v1_reshaped).transpose(1, 2).contiguous().reshape(B, C // 2, H // self.sr_ratio,
            #                                                                              W // self.sr_ratio). \
            #     reshape(B, C // 2, -1).reshape(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)

            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)   # [2, 4096, 32]
            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).   # v2: [2, 1, 256, 32]
                                    transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)  # [2, 4096, 32]  [2, 1024, 64] [2, 256, 128]

            x = torch.cat([x1,x2], dim=-1)  # [2, 4096, 64]
        elif self.sr_ratio == 8:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // sr_ratio) * (W // sr_ratio), C]
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C]
            kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # [2, B, num_heads // 2, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
            kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                    4)  # [2, B, num_heads // 2, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C // num_heads]
            k1, v1 = kv1[0], kv1[1]  # B head N C  k1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads] v1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
            k2, v2 = kv2[0], kv2[1]  # k2: [2, 1, 256, 64] v1: [2, 1, 256, 64]
            q = self.q_s(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)  # [B, num_heads // 2, N, C // num_heads]
            attn1 = (q @ k1.transpose(-2, -1)) * self.scale  # [B, num_heads // 2, N, C // num_heads]
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)  # [2, 1, 4096, 64]
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
                                       transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1,
                                                                                                    -2)  # v1: [2, 1, 64, 64]
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]
            attn2 = (q @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).  # v2: [2, 1, 256, 32]
                                       transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
                view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)  # [2, 1, 256, 64]
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]

            x = torch.cat([x1, x2], dim=-1)  # [2, 4096, 64]
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, 2, 16, 64, 32]
            k, v = kv[0], kv[1]  # [2, 16, 64, 64]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # [2, 16, 64, 32]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
                                        transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)  # [2, 64, 512]

        x = self.proj(x)
        x = self.proj_drop(x)  # [2, 4096, 128]

        if length == 4:
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x

# class Block(nn.Module):
#
#     def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
#                  drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
#                  focusing_factor=3, kernel_size=5, ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.sr_ratio = sr_ratio
#         self.attn = SpatialMultiAttention(
#             dim,
#             num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#             attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
#         # self.attn = MultiScaleLinearAttention(
#         #         dim, num_patches,
#         #         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
#         #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
#         #     focusing_factor=focusing_factor, kernel_size=kernel_size)
#         # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#
#         self.apply(self._init_weights)
#
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.Conv2d):
#             fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#             fan_out //= m.groups
#             m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
#             if m.bias is not None:
#                 m.bias.data.zero_()
#
#     def forward(self, x, H, W):
#         # x = torch.concat([x, y], dim=1)
#         x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # [2, 64, 512]   [2, 4096, 64]
#         x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   # [2, 64, 512] [2, 4096, 64] [2, 1024, 128] [2, 256, 256]
#
#         return x

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class ChannelMultiAttention(nn.Module):
    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16], aug=False):
        super(ChannelMultiAttention, self).__init__()
        self.conv_1 = conv(inplans, planes // 4, kernel_size=conv_kernels[0], padding=conv_kernels[0] // 2,
                           stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes // 4, kernel_size=conv_kernels[1], padding=conv_kernels[1] // 2,
                           stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes // 4, kernel_size=conv_kernels[2], padding=conv_kernels[2] // 2,
                           stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes // 4, kernel_size=conv_kernels[3], padding=conv_kernels[3] // 2,
                           stride=stride, groups=conv_groups[3])
        # self.se = SEWeightModule(planes // 4)
        self.dctcam = MultiSpectralAttentionLayer(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)
        self.aug = aug

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.dctcam(x1)#self.se(x1)
        x2_se = self.dctcam(x2)#self.se(x2)
        x3_se = self.dctcam(x3)#self.se(x3)
        x4_se = self.dctcam(x4)#self.se(x4)

        if self.aug == True:
            x_se = torch.max_pool2d(torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1), kernel_size=x1_se.shape[2])
            attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
            attention_vectors = self.softmax(attention_vectors)
            feats_weight = feats * attention_vectors
            for i in range(4):
                x_se_weight_fp = feats_weight[:, i, :, :]
                if i == 0:
                    out = x_se_weight_fp
                else:
                    out = torch.cat((x_se_weight_fp, out), 1)
        else:
            out = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        return out

class ChannelMultiAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.05, act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, is_change=False):
        super().__init__()

        self.mlp2 = Mlp(in_features=dim * 2, hidden_features=dim * 2, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm3 = norm_layer(dim * 2)

        self.norm1 = norm_layer(dim)
        self.sr_ratio = sr_ratio
        # self.attn_spatial = SpatialMultiAttention(
        #     dim,
        #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.attn_channel = ChannelMultiAttention(dim, dim, conv_groups=[1, 3, 6, 12])
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.isChange = is_change

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x = torch.concat([x, y], dim=1)
        C = x.shape[1]
        if self.isChange:
            x = self.drop_path(self.mlp2(self.norm3(x.view(-1, H*W, C)), H, W))
            x = x.view(-1, int(C / 2) ,H, W)#.view(-1, x.size(1), x.size(2), x.size(3))

        # x_spatial = x + self.drop_path(self.attn_spatial(self.norm1(x), H, W))  # [2, 64, 512]   [2, 4096, 64]
        # x_spatial = x + self.drop_path(self.mlp(self.norm2(x_spatial), H, W))   # [2, 64, 512] [2, 4096, 64] [2, 1024, 128] [2, 256, 256]
        # x_channel = self.drop_path(self.attn_channel(x.view(-1, x.shape[2], H, W)).view(-1, H * W, x.shape[2])) + x
        x_channel = self.drop_path(self.attn_channel(x)) + x
        # x_channel = self.drop_path(self.mlp(self.norm2(x_channel), H, W)) + x

        # if self.isChange:
        #     return self.drop_path(self.mlp2(self.norm3(x_spatial), H, W)) + self.drop_path(self.mlp2(self.norm3(x_channel), H, W))

        # return x_spatial + x_channel
        return x_channel

import torch.nn.functional as F

# class ImprovedSpatialAttention(nn.Module):
#     def __init__(self, in_channels, kernel_size=7):
#         super(ImprovedSpatialAttention, self).__init__()
#         self.kernel_size = kernel_size
#         self.conv1 = nn.Conv2d(4, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=False)
#         self.gaussian_kernel = self.create_gaussian_kernel(kernel_size)
#
#     def create_gaussian_kernel(self, kernel_size):
#         """生成高斯核"""
#         sigma = 1.0
#         kernel_range = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
#         kernel_1d = (1 / (sigma * torch.sqrt(torch.tensor(2.0 * torch.pi)))) * \
#                     torch.exp(-(kernel_range ** 2) / (2 * sigma ** 2))
#         kernel_1d = kernel_1d / kernel_1d.sum()  # 归一化
#         kernel_2d = kernel_1d.unsqueeze(0).mm(kernel_1d.unsqueeze(1))
#         return kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size, kernel_size)
#
#     def gaussian_blur(self, x):
#         """应用高斯模糊"""
#         # padding = self.kernel_size // 2
#         return F.conv2d(x, self.gaussian_kernel.to(x.device), groups=x.size(1))
#         # return F.conv2d(x, self.gaussian_kernel.to(x.device), padding=padding, groups=x.size(1))
#
#     def forward(self, x, H, W):
#         # x: 输入特征图，形状为 (batch_size, channels, height, width)
#
#         avg_out = torch.mean(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)  # (batch_size, 1, height, width)
#
#         # 获取不同尺度特征，并调整大小
#         scale1 = F.interpolate(avg_out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
#         scale2 = F.interpolate(max_out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
#
#         # 拼接
#         concat = torch.cat([avg_out, max_out, scale1, scale2], dim=1)  # (batch_size, 4, height, width)
#
#         # 空间注意力计算
#         attention = self.conv1(concat)  # (batch_size, 1, height, width)
#         attention = torch.sigmoid(attention)  # 激活函数
#
#         # 确保注意力图与输入特征图的尺寸一致
#         attention = F.interpolate(attention, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
#
#         # 应用高斯模糊处理
#         # attention = self.gaussian_blur(attention)
#
#         # 应用注意力权重
#         out = x * attention  # (batch_size, channels, height, width)
#         return out

class GaussianBlur(nn.Module):
    def __init__(self, kernel_size=3, sigma=1.0):
        super(GaussianBlur, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2

        # Create Gaussian kernel
        self.register_buffer('gaussian_kernel', self.create_gaussian_kernel())

    def create_gaussian_kernel(self):
        # Create a Gaussian kernel
        x = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
        y = torch.exp(-0.5 * (x**2 / self.sigma**2))
        kernel = y / y.sum()
        # Create 2D Gaussian kernel
        kernel_2d = kernel.unsqueeze(0) * kernel.unsqueeze(1)  # Outer product to create 2D kernel
        return kernel_2d.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        # Apply Gaussian blur
        return F.conv2d(x, self.gaussian_kernel, padding=self.padding, groups=x.shape[1])

class ImprovedSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(ImprovedSpatialAttention, self).__init__()
        self.gaussian_blur = GaussianBlur(kernel_size=3, sigma=1.0)  # 高斯模糊层
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

    def forward(self, x, H, W):
        # 应用高斯模糊
        # blurred_x = self.gaussian_blur(x)

        # 使用空间注意力机制
        # avg_out = F.adaptive_avg_pool2d(x, (1, 1))
        max_out = F.adaptive_max_pool2d(x, (1, 1))
        out = self.conv1(max_out)  #self.conv1(avg_out) + self.conv1(max_out)
        out = torch.sigmoid(self.conv2(out))
        return x * out

# 示例使用
# if __name__ == "__main__":
#     # 假设输入特征图大小为 (batch_size=2, channels=64, height=64, width=64)
#     input_tensor = torch.rand(2, 64, 64, 64)
#     spatial_attention = ImprovedSpatialAttention(in_channels=64)
#     output_tensor = spatial_attention(input_tensor)
#     print(output_tensor.shape)  # 输出形状应与输入相同



# if __name__ == '__main__':
#     img = torch.randn(2, 384, 64, 64).to('cuda')
#     model = ChannelMultiAttentionBlock(dim=384, num_heads=8).to('cuda')
#     print(model(img, 64, 64).shape)

# if __name__ == '__main__':
    from functools import partial
    # img = torch.randn(2, 128, 64, 64)
    # input_list = []
    # # for i in range(4):
    # #     img = torch.randn(2, 128*(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)))
    # #     input_list.append(img)
    # model = SpatialMultiAttention(dim=128, num_heads=8, sr_ratio=8)
    # print(model(img, 64, 64).shape)
    # tensor_test = torch.randn(2, 64*(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)))
#     tensor_test = torch.randn(1, 4096, 96)
#     # print(input_list[1].shape)
#     model = Attention(96,
#             num_heads=2, qkv_bias=False, qk_scale=None,
#             attn_drop=0.1, proj_drop=0.1, sr_ratio=8)
#     H, W = int(math.sqrt(tensor_test.shape[1])), int(math.sqrt(tensor_test.shape[1]))
#     print(model(tensor_test, H, W).shape)