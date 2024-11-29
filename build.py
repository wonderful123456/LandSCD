import argparse
import torch.nn as nn
from functools import partial

from models.BiSRNet import BiSRNet
from models.RegularLandSCD import ISwinUperNetV5
from models.RegularLandSCDV1 import RegularLandSCDV1
from models.RegularLandSCDV2 import RegularLandSCDV2
from models.SwinTransformerV2UperNetSeg import SwinTransformerUperNet
from models.SwinTransformerV2UperNetSegV1 import SwinTransformerUperNetBase
from models.SwinTransformerV2UperNetSegV2 import SwinTransformerUperNetV2
from models.SwinTransformerV2UperNetSegV4 import SwinTransformerUperNetV4
from models.SCanNet import SCanNet
from models.MTSCD_Net import DeepLabV3Plus
from models.DualBranchNet import DualBranchNetBase
from models.SwinV2UNet import SwinUNet
from models.SwinV2UNetUpernet import SwinUNetUperNet
from models.SwinV2UNetDCTCAM import SwinUNetDCTCAM
from models.HGINet import HGINet
from models.HRSCD4 import HRSCD4
from models.ConvNeXt import ConvNeXt
from models.vit_seg_modeling import VisionTransformer, CONFIGS
# from models.SwinTransformerV2UperNetSegV3 import SwinTransformerUperNetV3

# TransUNet config settings
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Synapse', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=6, help='output channel of network')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=256, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
args = parser.parse_args()

config_vit = CONFIGS[args.vit_name]
config_vit.n_skip=3
config_vit.vit_name='ViT-B_16'
config_vit.classifier="seg"
config_vit.n_classes = args.num_classes
config_vit.n_skip = args.n_skip
if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))


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
            'SwinTransformerUperNetBase':SwinTransformerUperNetBase,
            'SwinTransformerUperNetV2': SwinTransformerUperNetV2,
            'SwinTransformerUperNetV4': SwinTransformerUperNetV4,
            'SCanNet': SCanNet,
            'MTSCDNet': DeepLabV3Plus,
            'HGINet':HGINet,
            'DualBranchNetBase': DualBranchNetBase,
            'SwinUNet': SwinUNet,
            'SwinUNetUperNet':SwinUNetUperNet,
            'SwinUNetDCTCAM': SwinUNetDCTCAM,
            'HRSCD4': HRSCD4,
            'ConvNeXt': ConvNeXt,
            'TransUNet': VisionTransformer,
            # 'SwinTransformerUperNetV3': SwinTransformerUperNetV3
        }

    def build_model(self):
        if self.args.train_model == None or self.args.train_model not in self.models:
            raise NotImplementedError
        # print(self.args.GA_Stages)
        model = self.models[self.args.train_model]
        if model in (BiSRNet, ):
            return model(num_classes=self.args.n_class)
        elif model in (RegularLandSCDV1, RegularLandSCDV2, ISwinUperNetV5, SwinTransformerUperNetBase,
                       SwinTransformerUperNetV2, SwinTransformerUperNetV4):
            return model(pretrain_img_size=self.args.img_size, num_classes=self.args.n_class, in_chans=self.args.num_channel)
        elif model in (SwinTransformerUperNet, ):
            return model()
        elif model in (SCanNet, ):
            return model(in_channels=self.args.num_channel, num_classes=self.args.n_class, input_size=self.args.img_size )
        elif model in (DeepLabV3Plus,):
            return model('swin_small', False, 6, False)
        elif model in (DualBranchNetBase,):
            return model(pretrain_img_size=self.args.img_size, in_chans=self.args.num_channel, num_classes=self.args.n_class)
        elif model in (SwinUNet, SwinUNetUperNet, SwinUNetDCTCAM):
            return model()
        elif model in (HGINet, ):
            return model(channel=64, num_classes=self.args.n_class)
        elif model in (HRSCD4,):
            return model(input_nbr=3, label_nbr=6, wsl=True)
        elif model in (ConvNeXt,):
            return model(num_classes=self.args.n_class)
        elif model in (VisionTransformer,):
            return model(config_vit, img_size=self.args.img_size, num_classes=self.args.n_class)
        # elif model in (SwinTransformerUperNetV3,):
        #     return model(pretrain_img_size=self.args.img_size, num_classes=self.args.n_class, in_chans=self.args.num_channel, num_expert=self.args.num_expert)
        else:
            return model(num_classes=self.args.n_class,
                         backbone=self.args.backbone,
                         output_stride=self.args.out_stride,
                         sync_bn=self.args.sync_bn,
                         freeze_bn=self.args.freeze_bn)
