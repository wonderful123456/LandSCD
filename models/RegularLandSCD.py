import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
from utils.misc import initialize_weights

from models.backbone.iswin_transformerv3 import *
from models.sseg.uperhead import UperNetHead

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ISwinUperNetV5(nn.Module):
    def __init__(self, pretrain_img_size=224, num_classes=6, in_chans = 3, use_attens=1):
        super(ISwinUperNetV5, self).__init__()
        self.backbone = ISwinTransformerV3(
            pretrain_img_size=pretrain_img_size,
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
        self.decode_head = UperNetHead(
            in_channels=[96, 192, 384, 768],
            channels=512,
            num_classes=num_classes,
        )
        # self.auxiliary_head = FCNHead(
        #     in_channels=512,
        #     in_index=2,
        #     channels=256,
        #     num_convs=1,
        #     concat_input=False,
        #     dropout_ratio=0.1,
        #     num_classes=7,
        #     norm_cfg=dict(type='BN', requires_grad=True),
        #     align_corners=False,
        # )

    def forward(self, input):
        size = input.size()[2:]
        x, a_encoder, b_encoder = self.backbone(input)
        main_ = self.decode_head(x)
        # aux_ = self.auxiliary_head(x)
        # return main_,aux_ # 主分类器，辅助分类器
        main_ = F.interpolate(main_, size, mode='bilinear', align_corners=True)
        return main_  # 主分类器，辅助分类器

if __name__ == '__main__':
    import torch
    # from torchstat import stat
    device = torch.device("cuda")
    model = ISwinUperNetV5(pretrain_img_size=256).to(device)
    # batchsize % 2==0
    # images = torch.rand(size=(2, 6, 256, 256)).to(device)
    # images = images.to(device, dtype=torch.float32)
    # models.to(device)
    # ret1 = models(images).to(device)
    # print(ret1.size())
    # print(models)

    # from thop import profile

    # input = torch.randn(16, 6, 256, 256)
    # flops, params = profile(models, inputs=(input,))
    # print('the flops is {}G,the params is {}M'.format(round(flops / (10 ** 9), 2),
    #                                                   round(params / (10 ** 6), 2)))  # 4111514624.0 25557032.0 res50

    dummy_input = torch.randn(4, 6, 256, 256, dtype=torch.float).to(device)
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