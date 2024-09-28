import math

import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

from models.backbone.iswin_transformerv3 import *
from models.sseg.uperhead import UperNetHead

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
class RegularLandSCDV1(nn.Module):
    def __init__(self, pretrain_img_size=256, num_classes=6, in_chans = 6, use_attens=1):
        super(RegularLandSCDV1, self).__init__()
        self.backbone = ISwinTransformerV3(
            pretrain_img_size=pretrain_img_size,
            patch_size=4,
            in_chans=in_chans,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 16],
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

        self.conv_channel_compress = nn.ModuleList()
        for i in range(4):
            self.conv_channel_compress.append(conv1x1(256 * 2 ** i, 128 * 2 ** i))

        self.attn_conv_encoder = nn.ModuleList()
        conv_dim = [128, 256, 512, 1024]
        num_heads = [2, 4, 8, 16]
        sr_ratio = [8, 4, 2, 1]
        for i in range(4):
            self.attn_conv_encoder.append(Attention(128 * 2 ** i,
                        num_heads=num_heads[i], qkv_bias=True, qk_scale=None,
                        attn_drop=0.01, proj_drop=0.005, sr_ratio=sr_ratio[i]))

        self.attn_swin_encoder = nn.ModuleList()
        for i in range(4):
            self.attn_swin_encoder.append(Attention(96 * 2 ** i,
                        num_heads=num_heads[i], qkv_bias=True, qk_scale=None,
                        attn_drop=0.01, proj_drop=0.005, sr_ratio=sr_ratio[i]))


        self.cd_decode_head = UperNetHead(
            in_channels=[96, 192, 384, 768],
            channels=384,
            num_classes=2,
        )
        self.a_seg_decode_head = UperNetHead(
            in_channels=[128, 256, 512, 1024],
            channels=512,
            num_classes=num_classes,
        )
        self.b_seg_decode_head = UperNetHead(
            in_channels=[96, 192, 384, 768],
            channels=384,
            num_classes=num_classes,
        )

    def forward(self, input):
        size = input.size()[2:]
        x, a_encoder, b_encoder = self.backbone(input)
        main_ = self.cd_decode_head(x)

        a_encoder = [self.conv_channel_compress[i](a_encoder[i]) for i in range(4)]

        a_encoder = [self.attn_conv_encoder[i](a_encoder[i], a_encoder[i].shape[2], a_encoder[i].shape[3]) for i in range(4)]
        b_encoder = [self.attn_swin_encoder[i](b_encoder[i], b_encoder[i].shape[2], b_encoder[i].shape[3]) for i in range(4)]

        a_seg_out = self.a_seg_decode_head(a_encoder)
        b_seg_out = self.b_seg_decode_head(b_encoder)

        # aux_ = self.auxiliary_head(x)
        # return main_,aux_ # 主分类器，辅助分类器
        main_ = F.interpolate(main_, size, mode='bilinear', align_corners=True)
        a_seg_out = F.interpolate(a_seg_out, size, mode='bilinear', align_corners=True)
        b_seg_out = F.interpolate(b_seg_out, size, mode='bilinear', align_corners=True)
        return {'BCD': main_, 'seg_A': a_seg_out, 'seg_B': b_seg_out}  # 主分类器，辅助分类器

if __name__ == '__main__':
    import torch
    # from torchstat import stat
    device = torch.device("cuda")
    # img = torch.randn(2, 6, 256, 256)#.to(device)
    model = RegularLandSCDV1(pretrain_img_size=256).to(device)
    # print(model(img)[0].shape)
    # print(model(img)['BCD'].shape)
    # print(model(img)['seg_A'].shape)
    # print(model(img)['seg_B'].shape)
    # batchsize % 2==0
    # images = torch.rand(size=(2, 6, 256, 256)).to(device)
    # images = images.to(device, dtype=torch.float32)
    # models.to(device)
    # ret1 = models(images).to(device)
    # print(ret1.size())
    # print(models)

    from thop import profile

    input = torch.randn(16, 6, 256, 256)
    flops, params = profile(model, inputs=(input,))
    print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
                                                      round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50

    dummy_input = torch.randn(16, 6, 256, 256, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 50
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
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

    # stat(models, (3, 256, 256))