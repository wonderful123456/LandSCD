#!/usr/bin/env python
# coding=utf-8
# -*- coding:utf-8 -*-
#@Time : 2022/10/6 15:11
#@Author: sunrise
#@File : IMSCSANetV7.py
#@Todo : improved muti-scale context self-attention model
# DANet+改进的注意力机制+多尺度ASPP+多个pca
import torch
from torch import nn
from torch.nn import Parameter, Conv2d, Softmax
import torch.nn.functional as F

from models.MSCSANet import resnet50
from models.aspp import build_aspp
from models.sync_batchnorm import SynchronizedBatchNorm2d

'''
注意力基础上加上了CNN卷积
'''
def delu_feature_map(x, param = 10):
    x1 = F.relu(x)
    x2 = x - x1
    x2 = torch.exp(param*x2) - 1
    return param*x1 + x2 + 1
'''
空间注意力
'''
class ILMSPAM_Module(nn.Module):
    def __init__(self, in_places, scale=8, eps=1e-10):
        super(ILMSPAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.feature_map = delu_feature_map
        self.eps = eps
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys, y ,z 是x的上下文特征图
        batch_size, chnnels, width, height = x.shape
        Q = self.query_conv(x).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height) # b, c, n
        Q = self.feature_map(Q).permute(-3, -1, -2) # b, n, c
        K = self.feature_map(K) #b,c,n
        KV = torch.einsum("bmn, bcn->bmc", K, V)
        norm = 1 / torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps)
        weight_value = torch.einsum("bnm, bmc, bn->bcn", Q, KV, norm)
        # attention = weight_value.view(batch_size, chnnels, height, width) + self.dconv3(x) + self.dconv1(x)
        attention = weight_value.view(batch_size, chnnels, height, width)

        return (x + self.gamma * attention).contiguous()

'''
通道注意力
'''
class ILMSCAM_Module(nn.Module):
    def __init__(self, in_places):
        super(ILMSCAM_Module, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        batch_size, chnnels, width, height = x.shape
        proj_query = x.view(batch_size, chnnels, -1)
        proj_key = x.view(batch_size, chnnels, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)  # 矩阵乘法
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention1 = self.softmax(energy_new)
        proj_value = x.view(batch_size, chnnels, -1)
        # out = torch.bmm(attention1, proj_value).view(batch_size, chnnels, height, width) + self.dconv3(x) + self.dconv1(x)
        out = torch.bmm(attention1, proj_value).view(batch_size, chnnels, height, width)
        out = self.gamma * out + x
        return out


class ContextAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., eps = 1e-10):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.eps = eps
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.feature_map = delu_feature_map
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.lk = nn.Linear(conv_dim, dim, bias=qkv_bias)
        self.attn_local_drop = nn.Dropout(attn_drop)
        self.attn_global_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop1 = nn.Dropout(proj_drop)
        self.proj_drop2 = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, N, C = x.shape
        _, _, conv_C = y.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        # b, head, N, c/head
        l_k = y.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.feature_map(q)
        k = self.feature_map(k)
        l_k = self.feature_map(l_k)
        kv = (k.transpose(-2, -1) @ v) * self.scale # b, head, c, c
        l_kv = (l_k.transpose(-2, -1) @ v) * self.scale # b, head, c, c
        norm = 1 / torch.einsum("bhnc, bhc->bhn", q, torch.sum(k, dim=-2)+self.eps)
        attn_local = torch.einsum("bhnc, bhcc, bhn->bhnc", q, kv, norm)
        attn_global = torch.einsum("bhnc, bhcc, bhn->bhnc", q, l_kv, norm)

        attn_local = self.attn_local_drop(attn_local)
        attn_global = self.attn_global_drop(attn_global)
        x = attn_local.transpose(1, 2).reshape(B, N, C)
        y = attn_global.transpose(1, 2).reshape(B, N, C)
        # x = torch.cat((x, y), dim=-1)
        x = self.proj_drop1(self.proj1(x))
        y = self.proj_drop2(self.proj2(y))
        return x, y
'''
多头注意力
'''

class _MHead(nn.Module):
    def __init__(self, in_channels):
        super(_MHead, self).__init__()
        self.pam = ILMSPAM_Module(in_channels)
        self.cam = ILMSCAM_Module(in_channels)

    def forward(self, x):
        return self.pam(x) + self.cam(x)
'''
resnet block 
'''
class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out


class IMSCSANetV7(ResNet50):
    def __init__(self, n_classes, backbone = "resnet50", output_stride=16):
        super(IMSCSANetV7, self).__init__(n_classes)
        #         self.head = _DAHead(2048, nclass, aux, **kwargs)
        # self.head = _MDAHead(256, n_classes, aux, **kwargs)
        self.aux = True
        self.aspp = ASPP(backbone, output_stride, nn.BatchNorm2d)
        self.up1 = UpHead(1024, 256, bilinear=True)
        self.up2 = UpHead(512, 128, bilinear=True)
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(128, n_classes, 1)
        )
        self.__setattr__('exclusive', ['head'])

    def forward(self, x):
        size = x.size()[2:]  #512 * 512
        feature_map, _ = self.base_forward(x) # 256*128*128   512*64*64   1024*32*32  2048*32*32
        x = self.aspp(feature_map[3])
        x = self.up1(x, feature_map[1])
        x = self.up2(x, feature_map[0])
        x = self.out(x)
        output = F.interpolate(x, size, mode='bilinear', align_corners=True)
        return output

class UpHead(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.head = _MHead(out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.head(self.conv(x))

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 512, 1, bias=False)
        self.bn1 = BatchNorm(512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# if __name__ == "__main__":
#     print(1024 // 4)
#     model = IMSCSANetV7(16)
#     x = torch.randn(4, 3, 512, 512)
#     y = torch.randn(1, 3, 128, 128)
#     z = torch.randn(1, 1024, 32, 32)
#     w = torch.randn(1, 1024, 32, 32)
#     h = torch.randn(1, 1024, 32, 32)
#     # model = ResNet50()
#     # out = torch.cat(w, z, h), dim=1)
#     # out,_ = model.base_forward(x)
#     # model = MSCSABlock(64)
#     out = model(x)
#     print(out.size())
#
#     i = 1
if __name__ == "__main__":
    mode = Attention(56)
    x = torch.randn((1, 1024, 56))
    y = torch.randn((1, 1024, 56))
    out = mode(x, y)


