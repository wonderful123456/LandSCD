import math

# from build import model_cfg
# from models.backbone.swin_transfomer_v2_GF import SwinTransformerV2
from models.backbone.swin_transformer_v2 import SwinTransformerV2
from models.sseg.uperhead import UperNetHead

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class CatMerging(nn.Module):
    def __init__(self, dim, conv_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(dim + conv_dim, dim, bias=False)
        self.norm = norm_layer(dim + conv_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# base model
class SwinTransformerUperNet(nn.Module):
    def __init__(self, pretrain_img_size=256,embed_dim=96, num_classes=6, in_chans=3, patch_size=8, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.num_layers = len(depths)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        self.backbone = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8
        )
        self.backbone2 = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8
        )
        self.embed_dim = embed_dim

        self.a_seg_decode_head = UperNetHead(
            in_channels=[self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
            channels=self.embed_dim * 4,
            num_classes=num_classes,
        )
        self.b_seg_decode_head = UperNetHead(
            in_channels=[self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
            channels=self.embed_dim * 4,
            num_classes=num_classes,
        )
        self.res1 = self._make_layer(ResBlock, self.embed_dim * 16, self.embed_dim * 8, 6, stride=1)
        self.CD = nn.Sequential(nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1), nn.BatchNorm2d(self.embed_dim * 4), nn.ReLU(), nn.Conv2d(self.embed_dim * 4, 2, kernel_size=1))

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def CD_forward(self, x1, x2):
        b,c,h,w = x1.size()
        x = torch.cat([x1,x2], 1)
        x = self.res1(x)
        change = self.CD(x)
        return change

    def forward(self, x1, x2):
        H, W = x1.shape[2], x1.shape[3]
        x1, x1_list = self.backbone.forward_intermediates(x1)
        x2, x2_list = self.backbone2.forward_intermediates(x2)
        x1_seg = self.a_seg_decode_head(x1_list)
        x2_seg = self.b_seg_decode_head(x2_list)
        change = self.CD_forward(x1.transpose(1, 3).transpose(2, 3), x2.transpose(1, 3).transpose(2, 3))

        return F.interpolate(x1_seg, (H, W), mode='bilinear', align_corners=True), F.interpolate(x2_seg, (H, W), mode='bilinear', align_corners=True), \
                F.interpolate(change, (H, W), mode='bilinear', align_corners=True)


if __name__ == '__main__':

    img = torch.randn(2, 3, 256, 256).to('cuda')
    img_B = torch.randn(2, 3, 256, 256).to('cuda')
    model = SwinTransformerUperNet().to('cuda')
    print(model(img, img_B)[1].shape)