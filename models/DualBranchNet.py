import math

from models.backbone.swin_transformer_v2 import SwinTransformerV2
from models.sseg.uperhead import UperNetHead
from models.modules.Attention import ChannelMultiAttentionBlock, ImprovedSpatialAttention
from models.backbone.resnet import ResNet, Bottleneck
from models.backbone.iswin_transformerv3 import ISwinTransformerV3

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
class DualBranchNetBase(nn.Module):
    def __init__(self, pretrain_img_size=256, embed_dim=96, num_classes=6, in_chans=3, patch_size=4, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], num_stages=4, output_stride=16, use_attens=0):
        super().__init__()
        self.num_layers = len(depths)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # self.backbone1 = SwinTransformerV2(
        #     img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        #     embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8, drop_rate=0.1, proj_drop_rate=0.1,
        #     attn_drop_rates=[0.1, 0.1, 0.2, 0.2], drop_path_rate=0.1
        # )
        # self.backbone2 = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, nn.BatchNorm2d, layer_name=50, pretrained=False, isChangeVer=True)
        self.backbone = ISwinTransformerV3(pretrain_img_size=pretrain_img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.3,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            out_indices=(0, 1, 2, 3),
            frozen_stages=-1,
            use_checkpoint=False,
            use_attens=use_attens,
            layer_name="tiny")

        self.embed_dim = embed_dim

        self.num_heads = [1, 2, 4, 8]

        # self.fuse_module_A = nn.ModuleList(CatMerging(num_features[i], num_features[i]) for i in range(self.num_layers))
        # self.fuse_module_B = nn.ModuleList(CatMerging(num_features[i], num_features[i]) for i in range(self.num_layers))


        self.depth = [2, 2, 4, 1]
        self.depthCH = [1, 1, 1, 1]

        # self.segA_encoder_blocks = []
        # self.segB_encoder_blocks = []

        self.num_stages = num_stages

        # self.segA_encoder_block = DualMultiAttentionBlock(num_features[0], self.num_heads[0])
        # self.segB_encoder_block = DualMultiAttentionBlock(num_features[0], self.num_heads[0])

        # for i in range(self.num_stages):
        #     if 0 <= i and i <= 2:
        #         self.segA_encoder_block = nn.ModuleList(
        #             [ImprovedSpatialAttention(num_features[i]) for j in range(self.depth[i])])
        #         self.segB_encoder_block = nn.ModuleList(
        #             [ImprovedSpatialAttention(num_features[i]) for j in range(self.depth[i])])
        #     elif i > 2:
        #         self.segA_encoder_block = nn.ModuleList(
        #             [ChannelMultiAttentionBlock(num_features[i], self.num_heads[i]) for j in range(self.depth[i])])
        #         self.segB_encoder_block = nn.ModuleList(
        #             [ChannelMultiAttentionBlock(num_features[i], self.num_heads[i]) for j in range(self.depth[i])])
        #
        #     if i != 0:
        #         setattr(self, f"SegA_block{i + 1}", self.segA_encoder_block)
        #         setattr(self, f"SegB_block{i + 1}", self.segB_encoder_block)

            # self.segA_encoder_blocks.append(self.segA_encoder)
            # self.segB_encoder_blocks.append(self.segB_encoder)

        # self.segA_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])
        # self.segB_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])
        # self.fuse_module = nn.ModuleList([ChannelMultiAttentionBlock(num_features[i + 1], self.num_heads[i + 1], is_change=True) for i in range(3)])

        # self.fuse_module = nn.ModuleList(CatMerging(num_features[i + 1], num_features[i + 1]) for i in range(self.num_layers - 1))

        self.fuse_module = nn.ModuleList(
            CatMerging(num_features[i], num_features[i]) for i in range(self.num_layers))

        self.conv_compress_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2 * num_features[i], num_features[i], kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                nn.BatchNorm2d(num_features[i]),
                nn.GELU()
            ) for i in range(self.num_layers)
        ])

        # self.a_seg_decode_head = UperNetHead(
        #     in_channels=[self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
        #     channels=self.embed_dim * 4,
        #     num_classes=num_classes,
        # )
        # self.b_seg_decode_head = UperNetHead(
        #     in_channels=[self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
        #     channels=self.embed_dim * 4,
        #     num_classes=num_classes,
        # )
        # self.change_decode_head = UperNetHead(
        #     in_channels=[self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8],
        #     channels=self.embed_dim * 8,
        #     num_classes=2,
        # )

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

        # self.res1 = self._make_layer(ResBlock, self.embed_dim * 16, self.embed_dim * 8, 6, stride=1)
        # self.CD = nn.Sequential(nn.Conv2d(self.embed_dim * 8, self.embed_dim * 4, kernel_size=1), nn.BatchNorm2d(self.embed_dim * 4), nn.ReLU(), nn.Conv2d(self.embed_dim * 4, 2, kernel_size=1))

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

        x1_o, x2_o = torch.cat([x1, x1], dim=1), torch.cat([x2, x2], dim=1)

        x1_list = self.backbone(x1_o)
        x2_list = self.backbone(x2_o)


        # x1, x1_list = self.backbone1.forward_intermediates(x1)  # 共享权重
        # x2, x2_list = self.backbone1.forward_intermediates(x2)

        # x1_list[3] = F.interpolate(x1_list[3], scale_factor=2, mode='bilinear', align_corners=False)
        # x2_list[3] = F.interpolate(x2_list[3], scale_factor=2, mode='bilinear', align_corners=False)
        #
        # x1_conv, x1_list_conv = self.backbone2(x1_o)
        # x2_conv, x2_list_conv = self.backbone2(x2_o)
        #
        # x1_list = [self.conv_compress_list[i](torch.cat([x1_list[i], x1_list_conv[i]], dim=1)) for i in range(4)]
        # x2_list = [self.conv_compress_list[i](torch.cat([x2_list[i], x2_list_conv[i]], dim=1)) for i in range(4)]

        # x1_list_new = []
        # x2_list_new = []

        # for i in range(self.num_stages):  # 假设有4层
        #     if i != 0:
        #         x1_layer = x1_list[i]#.view(-1, x1_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
        #         x2_layer = x2_list[i]#.view(-1, x2_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
        #
        #         block_A = getattr(self, f"SegA_block{i + 1}")
        #         block_B = getattr(self, f"SegB_block{i + 1}")
        #         for blk in block_A:  # 遍历当前层的深度
        #             processed = blk(x1_layer, x1_list[i].shape[2], x1_list[i].shape[2])
        #             x1_list[i] = processed#.view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])
        #         for blk in block_B:
        #             processed2 = blk(x2_layer, x2_list[i].shape[2], x2_list[i].shape[2])
        #             x2_list[i] = processed2#.view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3])
        #         x1_list_new.append(x1_list[i])
        #         x2_list_new.append(x2_list[i])

        x1_list_new = x1_list
        x2_list_new = x2_list

        # x1_layer = x1_list[0].view(-1, x1_list[0].shape[2] ** 2, self.embed_dim * 2 ** 0)
        # x2_layer = x2_list[0].view(-1, x2_list[0].shape[2] ** 2, self.embed_dim * 2 ** 0)
        # processed1 = self.segA_encoder_block(x1_layer, x1_list[0].shape[2], x1_list[0].shape[2])
        # x1_list[0] = processed1.view(-1, self.embed_dim * 2 ** 0, x1_list[0].shape[2], x1_list[0].shape[3])
        # processed2 = self.segB_encoder_block(x2_layer, x2_list[0].shape[2], x2_list[0].shape[2])
        # x2_list[0] = processed2.view(-1, self.embed_dim * 2 ** 0, x2_list[0].shape[2], x2_list[0].shape[3])

        # x1_list = [self.segA_encoder[i](x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #                                 x1_list[i].shape[2], x1_list[i].shape[2]).view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3]) for i in range(4)]
        # x2_list = [self.segB_encoder[i](x2_list[i].view(-1, x2_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #                                 x2_list[i].shape[2], x2_list[i].shape[2]).view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3]) for i in range(4)]

        x1_seg = self.a_seg_decode_head(x1_list_new)
        x2_seg = self.b_seg_decode_head(x2_list_new)
        # change = self.CD_forward(x1.transpose(1, 3).transpose(2, 3), x2.transpose(1, 3).transpose(2, 3))
        # change = [self.fuse_module[i](
        #     torch.cat((x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #               x2_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i)), dim=-1), x1_list[i].shape[2], x1_list[i].shape[2])
        #           .view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])  for i in range(4)]
        # change = [self.fuse_module[i](
        #     torch.cat((x1_list_new[i], x2_list_new[i]), dim=1), x1_list_new[i].shape[2], x1_list_new[i].shape[2])  for i in range(3)]

        # change = [self.fuse_module[i](
        #     torch.cat((x1_list_new[i].view(-1, x1_list_new[i].shape[2] ** 2 ,self.embed_dim * 2 ** (i + 1)),
        #               x2_list_new[i].view(-1, x2_list_new[i].shape[2] ** 2 ,self.embed_dim * 2 ** (i + 1))), dim=-1))
        #           .view(-1, self.embed_dim * 2 ** (i + 1), x1_list_new[i].shape[2], x1_list_new[i].shape[3]) for i in range(3)]

        change = [self.fuse_module[i](
            torch.cat((x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
                      x2_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i)), dim=-1))
                  .view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3]) for i in range(4)]

        change = self.change_decode_head(change)

        return F.interpolate(x1_seg, (H, W), mode='bilinear', align_corners=True), F.interpolate(x2_seg, (H, W), mode='bilinear', align_corners=True), \
                F.interpolate(change, (H, W), mode='bilinear', align_corners=True)

if __name__ == '__main__':
    device = torch.device("cuda")
    img = torch.randn(2, 3, 256, 256).to('cuda')
    img_B = torch.randn(2, 3, 256, 256).to('cuda')
    models = DualBranchNetBase().to('cuda')
    print(models(img, img_B)[1].shape)

    # from thop import profile
    #
    # input = torch.randn(16, 3, 256, 256).to(device)
    # input_B = torch.randn(16, 3, 256, 256).to(device)
    # flops, params = profile(models, inputs=(input,input_B))
    # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
    #                                                   round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50

    # import numpy as np
    # dummy_input = torch.randn(16, 3, 256, 256, dtype=torch.float).to(device)
    # dummy_input_B = torch.randn(16, 3, 256, 256, dtype=torch.float).to(device)
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # repetitions = 50
    # timings = np.zeros((repetitions, 1))
    # # GPU-WARM-UP
    # for _ in range(10):
    #     _ = models(dummy_input, dummy_input_B)
    # # MEASURE PERFORMANCE
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         starter.record()
    #         _ = models(dummy_input, dummy_input_B)
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