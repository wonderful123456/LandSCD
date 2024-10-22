import math

from models.backbone.swin_transformer_v2 import SwinTransformerV2
from models.sseg.uperhead import UperNetHead
from models.modules.Attention import ChannelMultiAttentionBlock, SpatialMultiAttention

from models.modules.MixFFN import MixFFN
from models.modules.FDAF import FDAF

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer

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
class SwinTransformerUperNetV4(nn.Module):
    def __init__(self, pretrain_img_size=256, embed_dim=96, num_classes=6, in_chans=3, patch_size=8, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24], num_stages=4):
        super().__init__()
        self.num_layers = len(depths)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        self.backbone1 = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8, drop_rate=0.1, proj_drop_rate=0.1,
            attn_drop_rates=[0.1, 0.1, 0.2, 0.2], drop_path_rate=0.1
        )
        self.backbone2 = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=8, drop_rate=0.1, proj_drop_rate=0.1,
            attn_drop_rate=[0.1, 0.1, 0.2, 0.2], drop_path_rate=0.1
        )
        self.embed_dim = embed_dim

        self.num_heads = [1, 2, 4, 8]

        # self.fuse_module = nn.ModuleList(CatMerging(num_features[i], num_features[i]) for i in range(self.num_layers))


        self.depth = [1, 1, 1, 1]
        self.depthCH = [2, 2, 4, 2]

        # self.segA_encoder_blocks = []
        # self.segB_encoder_blocks = []

        self.num_stages = num_stages

        self.sr_ratio = [8, 4, 2, 1]

        # self.segA_encoder_block = DualMultiAttentionBlock(num_features[0], self.num_heads[0])
        # self.segB_encoder_block = DualMultiAttentionBlock(num_features[0], self.num_heads[0])
        self.segA_encoder_block = nn.ModuleList()
        self.segB_encoder_block = nn.ModuleList()


        for i in range(self.num_stages):
            # if i == 1:
            #     self.segA_encoder_block.append(SpatialMultiAttention(num_features[i],
            #                                                     num_heads=self.num_heads[i], qkv_bias=False,
            #                                                     qk_scale=None,attn_drop=0.1, proj_drop=0.1,
            #                                                     sr_ratio=self.sr_ratio[i]))
            #     self.segB_encoder_block.append(SpatialMultiAttention(num_features[i],
            #                                                          num_heads=self.num_heads[i], qkv_bias=False,
            #                                                          qk_scale=None, attn_drop=0.1, proj_drop=0.1,
            #                                                          sr_ratio=self.sr_ratio[i]))
            # else:
            self.segA_encoder_block.append(ChannelMultiAttentionBlock(num_features[i], self.num_heads[i]))
            self.segB_encoder_block.append(ChannelMultiAttentionBlock(num_features[i], self.num_heads[i]))

            # setattr(self, f"SegA_block{i + 1}", self.segA_encoder_block)
            # setattr(self, f"SegB_block{i + 1}", self.segB_encoder_block)
            # self.segA_encoder_blocks.append(self.segA_encoder)
            # self.segB_encoder_blocks.append(self.segB_encoder)

        # self.segA_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])
        # self.segB_encoder = nn.ModuleList([DualMultiAttentionBlock(num_features[i], self.num_heads[i]) for i in range(self.num_layers)])
        # self.fuse_module = nn.ModuleList([ChannelMultiAttentionBlock(num_features[i], self.num_heads[i], is_change=True) for i in range(self.num_layers)])

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
        self.conv_compress_A = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * 2 ** i, int(embed_dim * 2 ** i), kernel_size=1),
                nn.GELU()
            ) for i in range(self.num_stages)
        ])
        self.conv_compress_B = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim * 2 ** i, int(embed_dim * 2 ** i), kernel_size=1),
                nn.GELU()
            ) for i in range(self.num_stages)
        ])
        num_inputs = 4
        self.norm_cfg = dict(type='IN')
        self.fusion_conv_A = ConvModule(
            in_channels=int(embed_dim) * int(1 * (1 - 2 ** 4) / (1 - 2)),
            out_channels=embed_dim * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.fusion_conv_B = ConvModule(
            in_channels=int(embed_dim) * int(1 * (1 - 2 ** 4) / (1 - 2)),
            out_channels=embed_dim * 4,
            kernel_size=1,
            norm_cfg=self.norm_cfg)
        self.neck_layer = nn.ModuleList([FDAF(in_channels=embed_dim * 2 ** i) for i in range(4)])
        self.discriminator = nn.ModuleList([MixFFN(
            embed_dims=embed_dim * 2 ** (i + 1),
            feedforward_channels=embed_dim * 2 ** i,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU')) for i in range(4)])
        self.out_channels = 2
        self.conv_ch_cls = nn.Conv2d(embed_dim * 8, self.out_channels, kernel_size=1)
        self.change_decode_head = UperNetHead(
            in_channels=[self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8, self.embed_dim * 16],
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
        x1, x1_list = self.backbone1.forward_intermediates(x1)  # 共享权重
        x2, x2_list = self.backbone2.forward_intermediates(x2)

        for i in range(self.num_stages):  # 假设有4层
            # if i == 1:
            #     x1_layer = x1_list[i].view(-1, x1_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
            #     x2_layer = x2_list[i].view(-1, x2_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
            # else:
            x1_layer = x1_list[i]#.view(-1, x1_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)
            x2_layer = x2_list[i]#.view(-1, x2_list[i].shape[2] ** 2, self.embed_dim * 2 ** i)

            # block_A = getattr(self, f"SegA_block{i + 1}")
            # block_B = getattr(self, f"SegB_block{i + 1}")
            # for blk in block_A:
            #     processed = blk(x1_layer, x1_list[i].shape[2], x1_list[i].shape[2])

            processed = self.segA_encoder_block[i](x1_layer, x1_list[i].shape[2], x1_list[i].shape[2])
            # if i == 1:
            #     x1_list[
            #         i] = processed.view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])
            # else:
            x1_list[i] = processed#.view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])
            # for blk in block_B:
            #     processed2 = blk(x2_layer, x2_list[i].shape[2], x2_list[i].shape[2])
            processed2 = self.segB_encoder_block[i](x2_layer, x2_list[i].shape[2], x2_list[i].shape[2])
            # if i == 1:
            #     x2_list[
            #         i] = processed2.view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3])
            # else:
            x2_list[i] = processed2#.view(-1, self.embed_dim * 2 ** i, x2_list[i].shape[2], x2_list[i].shape[3])
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

        x1_seg = self.a_seg_decode_head(x1_list)
        x2_seg = self.b_seg_decode_head(x2_list)
        # change = self.CD_forward(x1.transpose(1, 3).transpose(2, 3), x2.transpose(1, 3).transpose(2, 3))
        # change = [self.fuse_module[i](
        #     torch.cat((x1_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i),
        #               x2_list[i].view(-1, x1_list[i].shape[2] ** 2 ,self.embed_dim * 2 ** i)), dim=-1), x1_list[i].shape[2], x1_list[i].shape[2])
        #           .view(-1, self.embed_dim * 2 ** i, x1_list[i].shape[2], x1_list[i].shape[3])  for i in range(4)]


        # change = [self.conv_compress[i](torch.cat((x1_list[i], x2_list[i]), dim=1))  for i in range(4)]
        seg_A_list = [self.conv_compress_A[i](x1_list[i])  for i in range(4)]
        seg_B_list = [self.conv_compress_B[i](x2_list[i])  for i in range(4)]
        # 获取第一个张量的大小
        target_size = seg_A_list[0].size()[2:]  # 获取高度和宽度
        # 对每个张量进行双线性插值上采样
        seg_A_upsampled = [F.interpolate(c, size=target_size, mode='bilinear', align_corners=False) for c in seg_A_list]
        seg_B_upsampled = [F.interpolate(c, size=target_size, mode='bilinear', align_corners=False) for c in seg_B_list]
        # seg_A_fusion = self.fusion_conv_A(torch.cat(seg_A_upsampled, dim=1))
        # seg_B_fusion = self.fusion_conv_B(torch.cat(seg_B_upsampled, dim=1))
        # change = self.neck_layer(seg_A_fusion, seg_B_fusion, 'concat')
        change = [self.neck_layer[i](seg_A_upsampled[i], seg_B_upsampled[i], 'concat') for i in range(4)]
        change = [self.discriminator[i](change[i]) for i in range(4)]
        # change = self.conv_ch_cls(change)

        change = self.change_decode_head(change)

        return F.interpolate(x1_seg, (H, W), mode='bilinear', align_corners=True), F.interpolate(x2_seg, (H, W), mode='bilinear', align_corners=True), \
                F.interpolate(change, (H, W), mode='bilinear', align_corners=True)

if __name__ == '__main__':
    device = torch.device("cuda")
    img = torch.randn(2, 3, 256, 256).to('cuda')
    img_B = torch.randn(2, 3, 256, 256).to('cuda')
    models = SwinTransformerUperNetV4().to('cuda')
    print(models(img, img_B)[2].shape)

    from thop import profile

    input = torch.randn(16, 3, 256, 256).to(device)
    input_B = torch.randn(16, 3, 256, 256).to(device)
    flops, params = profile(models, inputs=(input,input_B))
    print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
                                                      round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50