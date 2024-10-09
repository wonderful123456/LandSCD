import argparse
import torch.nn as nn
from functools import partial

from models.BiSRNet import BiSRNet
from models.RegularLandSCD import ISwinUperNetV5
from models.RegularLandSCDV1 import RegularLandSCDV1
from models.RegularLandSCDV2 import RegularLandSCDV2
from models.SwinTransformerV2UperNetSeg import SwinTransformerUperNet

# model_cfg = dict(
#     type='FarSeg',
#     params=dict(
#         resnet_encoder=dict(
#             resnet_type='resnet50',
#             include_conv5=True,
#             batchnorm_trainable=True,
#             pretrained=False,
#             freeze_at=0,
#             # 8, 16 or 32
#             output_stride=32,
#             with_cp=(False, False, False, False),
#             stem3_3x3=False,
#         ),
#         # fpn=dict(
#         #     in_channels_list=(256, 512, 1024, 2048),
#         #     out_channels=256,
#         #     conv_block=fpn.default_conv_block,
#         #     top_blocks=None,
#         # ),
#         scene_relation=dict(
#             in_channels=512,
#             channel_list=(256, 512, 1024, 2048),
#             out_channels=256,
#             scale_aware_proj=True,
#         ),
#         decoder=dict(
#             in_channels=256,
#             out_channels=128,
#             in_feat_output_strides=(4, 8, 16, 32),
#             out_feat_output_stride=4,
#             norm_fn=nn.BatchNorm2d,
#             num_groups_gn=None
#         ),
#         num_classes=self.n_class,
#     )
# )


model_cfg = dict(
    type='FarSeg',
    params=dict(
        resnet_encoder=dict(
            resnet_type='resnet50',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 8, 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            stem3_3x3=False,
        ),
        # fpn=dict(
        #     in_channels_list=(256, 512, 1024, 2048),
        #     out_channels=256,
        #     conv_block=fpn.default_conv_block,
        #     top_blocks=None,
        # ),
        scene_relation=dict(
            in_channels=512,
            channel_list=(256, 512, 1024, 2048),
            out_channels=256,
            scale_aware_proj=True,
        ),
        decoder=dict(
            in_channels=256,
            out_channels=128,
            in_feat_output_strides=(4, 8, 16, 32),
            out_feat_output_stride=4,
            norm_fn=nn.BatchNorm2d,
            num_groups_gn=None
        ),
        num_classes=5,
    )
)


# model_cfg = dict(
#     type='FarSeg',
#     params=dict(
#         resnet_encoder=dict(
#             resnet_type='resnet50',
#             include_conv5=True,
#             batchnorm_trainable=True,
#             pretrained=False,
#             freeze_at=0,
#             # 8, 16 or 32
#             output_stride=32,
#             with_cp=(False, False, False, False),
#             stem3_3x3=False,
#         ),
#         # fpn=dict(
#         #     in_channels_list=(256, 512, 1024, 2048),
#         #     out_channels=256,
#         #     conv_block=fpn.default_conv_block,
#         #     top_blocks=None,
#         # ),
#         scene_relation=dict(
#             in_channels=192,
#             channel_list=(96, 192, 384, 768),
#             out_channels=192,
#             scale_aware_proj=True,
#         ),
#         decoder=dict(
#             in_channels=256,
#             out_channels=128,
#             in_feat_output_strides=(4, 8, 16, 32),
#             out_feat_output_stride=4,
#             norm_fn=nn.BatchNorm2d,
#             num_groups_gn=None
#         ),
#         num_classes=self.n_class,
#     )
# )

class Builder(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.models = {
            'BiSRNet': BiSRNet,
            'RegularLandSCDV1': RegularLandSCDV1,
            'RegularLandSCDV2': RegularLandSCDV2,
            'SwinTransformerUperNet': SwinTransformerUperNet,
            'ISwinUperNetV5': ISwinUperNetV5#partial(ISwinUperNetV5, layer_name='tiny')
        }

    def build_model(self):
        if self.args.train_model == None or self.args.train_model not in self.models:
            raise NotImplementedError
        # print(self.args.GA_Stages)
        model = self.models[self.args.train_model]
        if model in (BiSRNet, ):
            return model(num_classes=self.args.n_class)
        elif model in (RegularLandSCDV1, RegularLandSCDV2, ISwinUperNetV5):
            return model(pretrain_img_size=self.args.img_size, num_classes=self.args.n_class, in_chans=self.args.num_channel)
        elif model in (SwinTransformerUperNet, ):
            return model()
        else:
            return model(num_classes=self.args.n_class,
                         backbone=self.args.backbone,
                         output_stride=self.args.out_stride,
                         sync_bn=self.args.sync_bn,
                         freeze_bn=self.args.freeze_bn)
