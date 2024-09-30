from models.backbone.swin_transfomer_v2 import SwinTransformerV2
from models.sseg.uperhead import UperNetHead

import torch
import torch.nn as nn

class SwinTransformerUperNetSeg(nn.Module):
    def __init__(self, pretrain_img_size=256,embed_dim=96, num_classes=6, in_chans=6, patch_size=4, depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24]):
        super().__init__()
        self.num_layers = len(depths)
        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features
        self.backbone = SwinTransformerV2(
            img_size=pretrain_img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=8, mlp_ratio=4., qkv_bias=True,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
            use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]
        )
        self.decoder = UperNetHead(num_classes, in_channels=num_features, channels=384)

    def forward(self, x):
        _, x = self.backbone.forward_features(x)


