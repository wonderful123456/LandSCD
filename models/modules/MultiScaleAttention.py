import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

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

# class FocusedLinearAttention(nn.Module):
#     def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
#                  focusing_factor=3, kernel_size=5):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#
#         self.q = nn.Conv2d(dim, dim, kernel_size=1)
#         # self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         # self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#         self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
#         self.attn_drop = nn.Dropout(attn_drop)
#
#         # self.proj = nn.Conv2d(dim, dim, stride=1)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
#             self.norm = nn.LayerNorm(dim)
#
#         self.focusing_factor = focusing_factor
#         self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
#                              groups=head_dim, padding=kernel_size // 2)
#         self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
#         self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
#         print('Linear Attention sr_ratio{} f{} kernel{}'.
#               format(sr_ratio, focusing_factor, kernel_size))
#
#     def forward(self, x, H, W):
#         B, N, C = x.shape
#         q = self.q(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, N, C)
#
#         if self.sr_ratio > 1:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
#             x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
#             x_ = self.norm(x_)
#             kv = self.kv(x_.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, N, 2 * C).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
#         else:
#             kv = self.kv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, N, 2 * C).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
#         k, v = kv[0], kv[1]
#
#         k = k + self.positional_encoding
#         focusing_factor = self.focusing_factor
#         kernel_function = nn.ReLU()
#         scale = nn.Softplus()(self.scale)
#         q = kernel_function(q) + 1e-6
#         k = kernel_function(k) + 1e-6
#         q = q / scale
#         k = k / scale
#         q_norm = q.norm(dim=-1, keepdim=True)
#         k_norm = k.norm(dim=-1, keepdim=True)
#         q = q ** focusing_factor
#         k = k ** focusing_factor
#         q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
#         k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
#
#         q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#         v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
#
#         z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
#         kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
#         x = q @ kv * z
#
#         if self.sr_ratio > 1:
#             v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
#         H = W = int(N ** 0.5)
#         x = x.transpose(1, 2).reshape(B, N, C)
#         v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
#         x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#
#         return x

# 多尺度边缘增强线性注意力
'''
    params:
        dim: 序列维度， 即d
        num_patches: patch的维度，即N
        num_heads: 注意力头数
        sr_ratio: 缩放率，用于多尺度注意力
        focusing_factor: 聚焦因子，参考聚焦线性注意力
        kernel_size: 注意力卷积核
'''
class MultiScaleLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 focusing_factor=3, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.sr_ratio = sr_ratio

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度数

        # 大尺度patch的q
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        # 小尺度边缘patch的q
        self.q_s = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)

        self.act = act_layer()

        self.attn_drop = nn.Dropout(attn_drop)  # 注意力的dropout层，防止注意力过拟合
        self.proj = nn.Linear(dim, dim)  # 线性投射层
        self.pro_drop = nn.Dropout(proj_drop)  # 线性投射层的dropout层，防止线性投射过拟合

        if self.sr_ratio > 1:
            self.sr1 = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm1 = nn.LayerNorm(dim)
            self.sr2 = nn.Conv2d(dim, dim, kernel_size=self.sr_ratio // 2, stride=self.sr_ratio // 2)
            self.norm2 = nn.LayerNorm(dim)

            self.kv1 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
            self.kv2 = nn.Conv2d(dim, 2 * dim, kernel_size=1)
            self.positional_encoding1 = nn.Parameter(
                torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))  # 位置编码
            self.positional_encoding2 = nn.Parameter(
                torch.zeros(size=(1, 4 * num_patches // (sr_ratio * sr_ratio), dim)))  # 位置编码
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
            self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.positional_encoding1 = nn.Parameter(
                torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))  # 位置编码
            self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        # 聚焦因子，需要调整
        self.focusing_factor = focusing_factor
        # 深度卷积，防止特征均质化
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))

    def LinearAttenrion(self, q, k, v, B, N, C, idx=1, sr_choose=0):
        if sr_choose == 0:
            q = q.reshape(B, N, C)
        else:
            q = q.reshape(B, N // self.sr_ratio ** 2, C)

        if idx == 1:
            k = k + self.positional_encoding1
        elif idx == 2:
            k = k + self.positional_encoding2

        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm

        if sr_choose == 0:
            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        else:
            q = q.reshape(B, N // self.sr_ratio ** 2, self.num_heads, -1).permute(0, 2, 1, 3)

        if idx == 1:
            k = k.reshape(B, N // (self.sr_ratio ** 2), self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, N // (self.sr_ratio ** 2), self.num_heads, -1).permute(0, 2, 1, 3)
        elif idx == 2:
            k = k.reshape(B, 4 * N // (self.sr_ratio ** 2), self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(B, 4 * N // (self.sr_ratio ** 2), self.num_heads, -1).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
        x = q @ kv * z

        if idx == 1:
            v = v.reshape(B, N // (self.sr_ratio ** 2), -1)
        elif idx == 2:
            v = v.reshape(B, 4 * N // (self.sr_ratio ** 2), -1)

        v = nn.functional.interpolate(v.permute(0, 2, 1), size=N, mode='linear').permute(0, 2, 1)

        H = W = int(N ** 0.5)
        if sr_choose == 0:
            x = x.transpose(1, 2).reshape(B, N, C)
            v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
            x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        else:
            if idx == 1:
                x = x.transpose(1, 2).reshape(B, N // self.sr_ratio ** 2, C)
                x = x.reshape(B, N // (self.sr_ratio ** 2), -1)
            elif idx == 2:
                x = x.transpose(1, 2).reshape(B, N // self.sr_ratio ** 2, C)
                # x = x.reshape(B, 4 * N // (self.sr_ratio ** 2), -1)
            x = nn.functional.interpolate(x.permute(0, 2, 1), size=N, mode='linear').permute(0, 2, 1)
            v = v.reshape(B * self.num_heads, H, W, -1).permute(0, 3, 1, 2)
            x = x + self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B, N, self.num_heads,
                                                                                          C // self.num_heads).permute(
            0, 2, 1, 3)

        if self.sr_ratio > 1 and self.sr_ratio != 8:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1.reshape(B, H // self.sr_ratio, W // self.sr_ratio, C).permute(0, 3, 1, 2)).reshape(B,
                                                                                                                  H // self.sr_ratio,
                                                                                                                  W // self.sr_ratio,
                                                                                                                  2 * C).reshape(
                B, -1, 2, C).permute(2, 0, 1, 3)
            kv2 = self.kv2(
                x_2.reshape(B, 2 * H // self.sr_ratio, 2 * W // self.sr_ratio, C).permute(0, 3, 1, 2)).reshape(B,
                                                                                                               2 * H // self.sr_ratio,
                                                                                                               2 * W // self.sr_ratio,
                                                                                                               2 * C).reshape(
                B, -1, 2, C).permute(2, 0, 1, 3)
            k1, v1 = kv1[0], kv1[1]
            k2, v2 = kv2[0], kv2[1]

            x_attn1 = self.LinearAttenrion(q, k1, v1, B, N, C, idx=1, sr_choose=0)
            x_attn2 = self.LinearAttenrion(q, k2, v2, B, N, C, idx=2, sr_choose=0)

            x = torch.cat([x_attn1, x_attn2], dim=-1)  # [2, 4096, 64]
            x = self.proj_drop(self.proj(x))
        elif self.sr_ratio == 8:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))
            x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))
            kv1 = self.kv1(x_1.reshape(B, H // self.sr_ratio, W // self.sr_ratio, C).permute(0, 3, 1, 2)).reshape(B,
                                                                                                                  H // self.sr_ratio,
                                                                                                                  W // self.sr_ratio,
                                                                                                                  2 * C).reshape(
                B, -1, 2, C).permute(2, 0, 1, 3)
            kv2 = self.kv2(
                x_2.reshape(B, 2 * H // self.sr_ratio, 2 * W // self.sr_ratio, C).permute(0, 3, 1, 2)).reshape(B,
                                                                                                               2 * H // self.sr_ratio,
                                                                                                               2 * W // self.sr_ratio,
                                                                                                               2 * C).reshape(
                B, -1, 2, C).permute(2, 0, 1, 3)
            k1, v1 = kv1[0], kv1[1]
            k2, v2 = kv2[0], kv2[1]

            q_s = self.q_s(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B,
                                                                                                  N // self.sr_ratio ** 2,
                                                                                                  self.num_heads,
                                                                                                  C // self.num_heads).permute(
                0, 2, 1, 3)
            x_attn1 = self.LinearAttenrion(q_s, k1, v1, B, N, C, idx=1, sr_choose=1)
            x_attn2 = self.LinearAttenrion(q_s, k2, v2, B, N, C, idx=2, sr_choose=1)

            x = torch.cat([x_attn1, x_attn2], dim=-1)  # [2, 4096, 64]
            x = self.proj_drop(self.proj(x))

        else:
            kv = self.kv(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).reshape(B,
                                                                                                N // self.sr_ratio ** 2,
                                                                                                2 * C).reshape(B, -1, 2,
                                                                                                               C).permute(
                2, 0, 1, 3)
            k, v = kv[0], kv[1]  # [2, 16, 64, 64]

            x = self.LinearAttenrion(q, k, v, B, N, C, idx=1)
            x = self.proj_drop(self.proj(x))

        return x

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
#         super().__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
#
#         self.dim = dim
#         self.num_heads = num_heads   # 2
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#
#         self.q = nn.Linear(dim, dim, bias=qkv_bias)
#         self.q_s = nn.Linear(dim, dim // 2, bias=qkv_bias)
#
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#
#         self.sr_ratio = sr_ratio
#         if sr_ratio > 1:
#             self.act = nn.GELU()
#             if sr_ratio==8:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
#                 self.norm2 = nn.LayerNorm(dim)
#             if sr_ratio==4:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
#                 self.norm2 = nn.LayerNorm(dim)
#             if sr_ratio==2:
#                 self.sr1 = nn.Conv2d(dim, dim, kernel_size=2, stride=2)
#                 self.norm1 = nn.LayerNorm(dim)
#                 self.sr2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
#                 self.norm2 = nn.LayerNorm(dim)
#             self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
#             self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
#             self.local_conv1 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
#             self.local_conv2 = nn.Conv2d(dim//2, dim//2, kernel_size=3, padding=1, stride=1, groups=dim//2)
#         else:
#             self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
#             self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
#         self.apply(self._init_weights)
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
#         B, N, C = x.shape  # B: 2 C: 128 N: 4096
#         q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)   # --[B, num_heads, N, C // num_heads]
#         if self.sr_ratio > 1:# and self.sr_ratio != 8:
#             x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [2, 64, 64, 64]
#             x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [2, N // sr_ratio ** 2, C]
#             x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [2, N // (sr_ratio // 2) ** 2, C]
#             kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, num_heads//2, N // sr_ratio ** 2, C // num_heads]
#             kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, B, num_heads//2, N // (sr_ratio // 2) ** 2, C // num_heads]
#             k1, v1 = kv1[0], kv1[1]  # B head N C  k1: [2, 1, 64, 32] v1: [2, 1, 64, 32]
#             k2, v2 = kv2[0], kv2[1]  # k2: [2, 1, 256, 32] v1: [2, 1, 256, 32]
#             q_ = q[:, :self.num_heads//2]  # --[B, self.num_heads//2, N, C // num_heads]
#             k_ = k1.transpose(-2, -1)  # --[B,, num_heads//2, N // (sr_ratio // 2) ** 2, C // num_heads]
#             attn1 = (q[:, :self.num_heads//2] @ k1.transpose(-2, -1)) * self.scale  # --[2, 4, 256, 64]
#             attn1 = attn1.softmax(dim=-1)
#             attn1 = self.attn_drop(attn1)
#             v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
#                                     transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
#                 view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)  # v1: [B, num_heads//2, N // sr_ratio ** 2, C // num_heads]
#             x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)   # [2, 4096, 32]
#             attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
#             attn2 = attn2.softmax(dim=-1)
#             attn2 = self.attn_drop(attn2)
#             v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).   # v2: [2, 1, 256, 32]
#                                     transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
#                 view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
#             x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)  # [2, 4096, 32]  [2, 1024, 64] [2, 256, 128]
#
#             x = torch.cat([x1,x2], dim=-1)  # [2, 4096, 64]
#         # elif self.sr_ratio == 8:
#         #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
#         #     x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // sr_ratio) * (W // sr_ratio), C]
#         #     x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C]
#         #     kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
#         #                                                                                             4)  # [2, B, num_heads // 2, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
#         #     kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
#         #                                                                                             4)  # [2, B, num_heads // 2, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C // num_heads]
#         #     k1, v1 = kv1[0], kv1[1]  # B head N C  k1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads] v1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
#         #     k2, v2 = kv2[0], kv2[1]  # k2: [2, 1, 256, 64] v1: [2, 1, 256, 64]
#         #     q = self.q_s(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)  # [B, num_heads // 2, N, C // num_heads]
#         #     attn1 = (q @ k1.transpose(-2, -1)) * self.scale  # [B, num_heads // 2, N, C // num_heads]
#         #     attn1 = attn1.softmax(dim=-1)
#         #     attn1 = self.attn_drop(attn1)  # [2, 1, 4096, 64]
#         #     v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
#         #                                transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
#         #         view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1,
#         #                                                                                             -2)  # v1: [2, 1, 64, 64]
#         #     x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]
#         #     attn2 = (q @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
#         #     attn2 = attn2.softmax(dim=-1)
#         #     attn2 = self.attn_drop(attn2)
#         #     v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).  # v2: [2, 1, 256, 32]
#         #                                transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
#         #         view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)  # [2, 1, 256, 64]
#         #     x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]
#         #
#         #     x = torch.cat([x1, x2], dim=-1)  # [2, 4096, 64]
#         else:
#             kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # [2, 2, 16, 64, 32]
#             k, v = kv[0], kv[1]  # [2, 16, 64, 64]
#
#             attn = (q @ k.transpose(-2, -1)) * self.scale  # [2, 16, 64, 32]
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#
#             x = (attn @ v).transpose(1, 2).reshape(B, N, C) + self.local_conv(v.transpose(1, 2).reshape(B, N, C).
#                                         transpose(1, 2).view(B,C, H, W)).view(B, C, N).transpose(1, 2)  # [2, 64, 512]
#
#         x = self.proj(x)
#         x = self.proj_drop(x)  # [2, 4096, 128]
#
#         return x

# new Attention modify by ljc
class Attention(nn.Module):
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
            v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C//2).
                                    transpose(1, 2).view(B,C//2, H//self.sr_ratio, W//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)  # v1: [B, num_heads//2, N // sr_ratio ** 2, C // num_heads]
            x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)   # [2, 4096, 32]
            attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C//2).   # v2: [2, 1, 256, 32]
                                    transpose(1, 2).view(B, C//2, H*2//self.sr_ratio, W*2//self.sr_ratio)).\
                view(B, C//2, -1).view(B, self.num_heads//2, C // self.num_heads, -1).transpose(-1, -2)
            x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C//2)  # [2, 4096, 32]  [2, 1024, 64] [2, 256, 128]

            x = torch.cat([x1,x2], dim=-1)  # [2, 4096, 64]
        # elif self.sr_ratio == 8:
        #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)  # [B, C, H, W]
        #     x_1 = self.act(self.norm1(self.sr1(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // sr_ratio) * (W // sr_ratio), C]
        #     x_2 = self.act(self.norm2(self.sr2(x_).reshape(B, C, -1).permute(0, 2, 1)))  # [B, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C]
        #     kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
        #                                                                                             4)  # [2, B, num_heads // 2, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
        #     kv2 = self.kv2(x_2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1,
        #                                                                                             4)  # [2, B, num_heads // 2, (H // (sr_ratio // 2)) * (W // (sr_ratio // 2)), C // num_heads]
        #     k1, v1 = kv1[0], kv1[1]  # B head N C  k1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads] v1: [B, 1, (H // sr_ratio) * (W // sr_ratio), C // num_heads]
        #     k2, v2 = kv2[0], kv2[1]  # k2: [2, 1, 256, 64] v1: [2, 1, 256, 64]
        #     q = self.q_s(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3)  # [B, num_heads // 2, N, C // num_heads]
        #     attn1 = (q @ k1.transpose(-2, -1)) * self.scale  # [B, num_heads // 2, N, C // num_heads]
        #     attn1 = attn1.softmax(dim=-1)
        #     attn1 = self.attn_drop(attn1)  # [2, 1, 4096, 64]
        #     v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).
        #                                transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)). \
        #         view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1,
        #                                                                                             -2)  # v1: [2, 1, 64, 64]
        #     x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]
        #     attn2 = (q @ k2.transpose(-2, -1)) * self.scale  # [2, 1, 4096, 256]
        #     attn2 = attn2.softmax(dim=-1)
        #     attn2 = self.attn_drop(attn2)
        #     v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).  # v2: [2, 1, 256, 32]
        #                                transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)). \
        #         view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)  # [2, 1, 256, 64]
        #     x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)  # [2, 4096, 64]
        #
        #     x = torch.cat([x1, x2], dim=-1)  # [2, 4096, 64]
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

        return x

class Block(nn.Module):

    def __init__(self, dim, num_patches, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 focusing_factor=3, kernel_size=5, ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.sr_ratio = sr_ratio
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # self.attn = MultiScaleLinearAttention(
        #         dim, num_patches,
        #         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
        #     focusing_factor=focusing_factor, kernel_size=kernel_size)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

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
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))  # [2, 64, 512]   [2, 4096, 64]
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))   # [2, 64, 512] [2, 4096, 64] [2, 1024, 128] [2, 256, 256]

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

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

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        stem = [nn.Conv2d(3, 64, 7, 2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(True)]
        for i in range(num):
            stem.append(nn.Conv2d(64, 64, 3, 1, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(64))
            stem.append(nn.ReLU(True))
        stem.append(nn.Conv2d(64, 64, kernel_size=2, stride=2))
        self.conv = nn.Sequential(*stem)
        self.norm = nn.LayerNorm(64)
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
    def forward(self, x):
        x = self.conv(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H,W

# 传入tensor list
# class MultiScaleAttention(nn.Module):
#     def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
#                  num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
#                  attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
#                  depths=[3, 4, 6, 3], sr_ratios=[4, 2, 1], num_stages=4, num_conv=0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.depths = depths
#         self.num_stages = num_stages
#
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#         cur = 0
#
#
#
#         for i in range(num_stages):
#
#             block = nn.ModuleList([Block(
#                 dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale,
#                 drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
#                 sr_ratio=sr_ratios[i])
#                 for j in range(depths[i])])
#             norm = norm_layer(embed_dims[i])
#             cur += depths[i]
#
#             # setattr(self, f"patch_embed{i + 1}", patch_embed)
#             setattr(self, f"block{i + 1}", block)
#             setattr(self, f"norm{i + 1}", norm)
#
#             # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
#
#     def forward_features(self, x):
#         B = x[0].shape[0]  # B: 2
#
#         for i in range(self.num_stages):
#             # patch_embed = getattr(self, f"patch_embed{i + 1}")
#             block = getattr(self, f"block{i + 1}")
#             norm = getattr(self, f"norm{i + 1}")
#             # x, H, W = patch_embed(x)  # [2, 4096, 64]
#             H, W = x[i].shape[2], x[i].shape[3]
#             x_ = x[i].reshape(x[i].shape[0], x[i].shape[1], H*W).permute(0, 2, 1)
#             for blk in block:
#                 x_ = blk(x_, H, W)  #  [2, 4096, 64] [2, 64, 512]
#             x_ = norm(x_)
#             if i != self.num_stages - 1:
#                 x_ = x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
#
#         return x_.permute(0, 2, 1).reshape(x_.shape[0], x_.shape[2], 8, 8)#x_.mean(dim=1)
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         # x = self.head(x)
#
#         return x



# 传入tensor
class MultiScaleAttention(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=64,
                 num_heads=8, mlp_ratios=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], la_sr_ratios='8421', num_stages=4,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # num_patches = [64, 256, 1024, 4096]
        num_patches = [4096, 1024,  256,   64]


        for i in range(num_stages):

            block = nn.ModuleList([Block(
                dim=embed_dims, num_patches=num_patches, num_heads=num_heads, mlp_ratio=mlp_ratios,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i],
                focusing_factor=focusing_factor, kernel_size=kernel_size)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims)
            cur += depths[i]

            # setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

            # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]  # B: 2  [2, 512, 8, 8]

        # patch_embed = getattr(self, f"patch_embed{i + 1}")
        i = int(math.log(x.shape[1] // 128, 2))
        block = getattr(self, f"block{i + 1}")
        norm = getattr(self, f"norm{i + 1}")
        # x, H, W = patch_embed(x)  # [2, 4096, 64]
        H, W = x.shape[2], x.shape[3]
        x_ = x.reshape(x.shape[0], x.shape[1], H*W).permute(0, 2, 1)
        for blk in block:
            x_ = blk(x_, H, W)  #  [2, 4096, 64] [2, 64, 512]
        x_ = norm(x_)
        x_ = x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [2, 512, 8, 8]

        return x_#x_.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

if __name__ == '__main__':
    from functools import partial
    # img = torch.randn(2, 3, 256, 256)
    input_list = []
    for i in range(4):
        img = torch.randn(2, 128*(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)))
        input_list.append(img)
    # tensor_test = torch.randn(2, 64*(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)))
    tensor_test = torch.randn(2, 128, 64, 64)
    # print(input_list[1].shape)
    model = MultiScaleAttention(patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 24, 2], sr_ratios=[8, 4, 2, 1], num_conv=2,
        )
    # print(model(tensor_test).shape)

    # from  thop import profile
    # Flops, params = profile(model, inputs=(tensor_test,)) # macs
    # print('Flops: % .4fG'%(Flops / 1000000000))# 计算量  8.54386 GFlops
    # print('params参数量: % .4fM'% (params / 1000000)) #参数量：等价与上面的summary输出的Total params值
    #
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

    import numpy as np

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(tensor_test)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(tensor_test)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn
    print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn,
                                                                                         std_syn=std_syn,
                                                                                         mean_fps=mean_fps))
    print(mean_syn)



    # img_size = 256
    # patch_size = 16
    # in_chans = 3
    # num_classes = 1000
    # embed_dims = [64, 128, 256, 512]
    # num_heads = [1, 2, 4, 8]
    # mlp_ratios = [4, 4, 4, 4]
    # qkv_bias = False
    # qk_scale = None
    # drop_rate = 0.
    # attn_drop_rate = 0.
    # drop_path_rate = 0.
    # norm_layer = nn.LayerNorm
    # depths = [3, 4, 6, 3]
    # sr_ratios = [8, 4, 2, 1]
    # num_stages = 4
    # num_conv = 0
    # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
    # cur = 0
    # for i in range(num_stages):
    #     dim = embed_dims[i]
    #     num_heads = num_heads[i]
    #     mlp_ratio = mlp_ratios[i]
    #     qkv_bias = qkv_bias
    #     qk_scale = qk_scale
    #     drop = drop_rate
    #     attn_drop = attn_drop_rate
    #     # drop_path = [dpr[cur + j] for j in range(depths[i]]
    #     # norm_layer = norm_layer
    #
    #     model = Attention(dim,
    #         num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
    #         attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratios[i])