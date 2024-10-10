import argparse
import torch.nn as nn
from functools import partial

from models.BiSRNet import BiSRNet
from models.RegularLandSCD import ISwinUperNetV5
from models.RegularLandSCDV1 import RegularLandSCDV1
from models.RegularLandSCDV2 import RegularLandSCDV2
from models.SwinTransformerV2UperNetSeg import SwinTransformerUperNet
from models.SwinTransformerV2UperNetSegV1 import SwinTransformerUperNetBase

class Builder(object):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.models = {
            'BiSRNet': BiSRNet,
            'RegularLandSCDV1': RegularLandSCDV1,
            'RegularLandSCDV2': RegularLandSCDV2,
            'SwinTransformerUperNet': SwinTransformerUperNet,
            'ISwinUperNetV5': ISwinUperNetV5, #partial(ISwinUperNetV5, layer_name='tiny')
            'SwinTransformerUperNetBase':SwinTransformerUperNetBase
        }

    def build_model(self):
        if self.args.train_model == None or self.args.train_model not in self.models:
            raise NotImplementedError
        # print(self.args.GA_Stages)
        model = self.models[self.args.train_model]
        if model in (BiSRNet, ):
            return model(num_classes=self.args.n_class)
        elif model in (RegularLandSCDV1, RegularLandSCDV2, ISwinUperNetV5, SwinTransformerUperNetBase):
            return model(pretrain_img_size=self.args.img_size, num_classes=self.args.n_class, in_chans=self.args.num_channel)
        elif model in (SwinTransformerUperNet, ):
            return model()
        else:
            return model(num_classes=self.args.n_class,
                         backbone=self.args.backbone,
                         output_stride=self.args.out_stride,
                         sync_bn=self.args.sync_bn,
                         freeze_bn=self.args.freeze_bn)
