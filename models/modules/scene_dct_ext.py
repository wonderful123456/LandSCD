import torch
from torch import nn
import torch.nn.functional as F

class SceneRelation(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_list,
                 out_channels,
                 scale_aware_proj=True):
        super(SceneRelation, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        if scale_aware_proj:
            self.scene_encoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.ReLU(True),
                    nn.Conv2d(out_channels, out_channels, 1),
                ) for _ in range(len(channel_list))]
            )
        else:
            # 2mlp
            self.scene_encoder = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.content_encoders = nn.ModuleList()
        self.feature_reencoders = nn.ModuleList()
        for c in channel_list:
            self.content_encoders.append(
                nn.Sequential(
                    nn.Conv2d(c//4, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )
            self.feature_reencoders.append(
                nn.Sequential(
                    nn.Conv2d(c//4, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True)
                )
            )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features: list):
        content_feats = [c_en(p_feat) for c_en, p_feat in zip(self.content_encoders, features)]  # 同fpn_feat_list的维度
        if self.scale_aware_proj:
            scene_feats = [op(scene_feature) for op in self.scene_encoder]  # [2, 256, 1, 1]
            relations = [self.normalizer((sf * cf).sum(dim=1, keepdim=True)) for sf, cf in # 0:[2,1,64,64],1:[2,1,32,32]
                         zip(scene_feats, content_feats)]  # 2:[2,1,16,16],3:[2,1,8,8]
        # else:
        #     scene_feat = self.scene_encoder(scene_feature)
        #     relations = [self.normalizer((scene_feat * cf).sum(dim=1, keepdim=True)) for cf in content_feats]

        p_feats = [op(p_feat) for op, p_feat in zip(self.feature_reencoders, features)]  # 同fpn_feat_list的维度
        # [2, 256, 64,64] [2, 256, 32,32] [2, 256, 16,16] [2, 256, 8, 8]

        refined_feats = [r * p for r, p in zip(relations, p_feats)]  # 同fpn_feat_list的维度
        # [2, 256, 64,64] [2, 256, 32,32] [2, 256, 16,16] [2, 256, 8, 8]
        return refined_feats