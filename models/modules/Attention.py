import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

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

        return x

if __name__ == '__main__':
    from functools import partial
    # img = torch.randn(2, 3, 256, 256)
    input_list = []
    for i in range(4):
        img = torch.randn(2, 128*(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)), 128//(int)(math.pow(2, i)))
        input_list.append(img)
    # tensor_test = torch.randn(2, 64*(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)), 64//(int)(math.pow(2, i)))
    tensor_test = torch.randn(1, 4096, 96)
    # print(input_list[1].shape)
    model = Attention(96,
            num_heads=2, qkv_bias=False, qk_scale=None,
            attn_drop=0.1, proj_drop=0.1, sr_ratio=8)
    H, W = int(math.sqrt(tensor_test.shape[1])), int(math.sqrt(tensor_test.shape[1]))
    print(model(tensor_test, H, W).shape)