import math

from models.backbone.swin_transformer_v2 import SwinTransformerV2
from models.sseg.uperhead import UperNetHead
from models.modules.Attention import DualMultiAttentionBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

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

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# base model
class SwinTransformerUperNetV2(nn.Module):
    def __init__(self, pretrain_img_size=256, embed_dim=96, num_classes=6, in_chans=3, patch_size=8, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.num_layers = len(depths)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        self.backbone = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8
        )
        # self.backbone2 = SwinTransformerV2(
        #     img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        #     embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8
        # )
        self.embed_dim = embed_dim

        self.num_heads = [1, 2, 4, 8]

        # self.fuse_module = nn.ModuleList(CatMerging(num_features[i], num_features[i]) for i in range(self.num_layers))
        self.fuse_module = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i], is_change=True) for i in range(self.num_layers)])

        self.depth = [2, 2, 6, 2]
        self.depthCH = [2, 2, 4, 2]

        self.segA_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])
        self.segB_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])

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
        self.change_decode_head = UperNetHead(
            in_channels=[self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
            channels=self.embed_dim * 4,
            num_classes=2,
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
        x1, x1_list = self.backbone.forward_intermediates(x1)  # 共享权重
        x2, x2_list = self.backbone.forward_intermediates(x2)

        for i in range(4):  # 假设有4层
            x1_layer = x1_list[i].view(-1, x1_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
            x2_layer = x2_list[i].view(-1, x2_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
            for j in range(self.depth[i]):  # 遍历当前层的深度
                processed = self.segA_encoder[i](x1_layer, x1_list[i].shape[2], x1_list[i].shape[2])
                x1_list[i] = processed.view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])
                processed2 = self.segB_encoder[i](x2_layer, x2_list[i].shape[2], x2_list[i].shape[2])
                x2_list[i] = processed2.view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3])

        # x1_list = [self.segA_encoder[i](x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #                                 x1_list[i].shape[2], x1_list[i].shape[2]).view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3]) for i in range(4)]
        # x2_list = [self.segB_encoder[i](x2_list[i].view(-1, x2_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #                                 x2_list[i].shape[2], x2_list[i].shape[2]).view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3]) for i in range(4)]

        x1_seg = self.a_seg_decode_head(x1_list)
        x2_seg = self.b_seg_decode_head(x2_list)
        # change = self.CD_forward(x1.transpose(1, 3).transpose(2, 3), x2.transpose(1, 3).transpose(2, 3))
        change = [self.fuse_module[i](
            torch.cat((x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
                      x2_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i)), dim=-1), x1_list[i].shape[2], x1_list[i].shape[2])
                  .view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])  for i in range(4)]
        change = self.change_decode_head(change)

        return F.interpolate(x1_seg, (H, W), mode='bilinear', align_corners=True), F.interpolate(x2_seg, (H, W), mode='bilinear', align_corners=True), \
                F.interpolate(change, (H, W), mode='bilinear', align_corners=True)

if __name__ == '__main__':
    device = torch.device("cuda")
    img = torch.randn(2, 3, 256, 256).to('cuda')
    img_B = torch.randn(2, 3, 256, 256).to('cuda')
    models = SwinTransformerUperNetV2().to('cuda')
    print(models(img, img_B)[2].shape)

    from thop import profile

    # input = torch.randn(16, 3, 256, 256).to(device)
    # input_B = torch.randn(16, 3, 256, 256).to(device)
    # flops, params = profile(models, inputs=(input,input_B))
    # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
    #                                                   round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50