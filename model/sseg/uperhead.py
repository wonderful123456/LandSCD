###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################
import torch
from torch.nn.functional import interpolate
import torch.nn as nn
from torch.nn import functional as F

class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # bilinear interpolate options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        # print('x shape is ', x.shape)
        feat1 = F.interpolate(self.conv1(self.pool1(x)), (h, w), mode = 'bilinear', align_corners = True)
        feat2 = F.interpolate(self.conv2(self.pool2(x)), (h, w), mode = 'bilinear', align_corners = True)
        feat3 = F.interpolate(self.conv3(self.pool3(x)), (h, w), mode = 'bilinear', align_corners = True)
        feat4 = F.interpolate(self.conv4(self.pool4(x)), (h, w), mode = 'bilinear', align_corners = True)
        return torch.cat((x, feat1, feat2, feat3, feat4), 1)

class FCFPNHead(nn.Module):
    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 512, 1024, 2048],
                 fpn_dim=256, up_kwargs=None):
        super(FCFPNHead, self).__init__()
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        fpn_lateral = []
        for fpn_inchannel in fpn_inchannels[:-1]:
            fpn_lateral.append(nn.Sequential(
                nn.Conv2d(fpn_inchannel, fpn_dim, kernel_size=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_lateral = nn.ModuleList(fpn_lateral)
        fpn_out = []
        for _ in range(len(fpn_inchannels) - 1):
            fpn_out.append(nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True),
            ))
        self.fpn_out = nn.ModuleList(fpn_out)
        self.c4conv = nn.Sequential(nn.Conv2d(fpn_inchannels[-1], fpn_dim, 3, padding=1, bias=False),
                                    norm_layer(fpn_dim),
                                    nn.ReLU())
        inter_channels = len(fpn_inchannels) * fpn_dim
        self.conv5 = nn.Sequential(nn.Conv2d(inter_channels, 512, 3, padding=1, bias=False),
                                   norm_layer(512),
                                   nn.ReLU(),
                                   nn.Dropout(0.1, False),
                                   nn.Conv2d(512, out_channels, 1))

    def forward(self, inputs):
        c4 = inputs[-1]
        #se_pred = False
        if hasattr(self, 'extramodule'):
            #if self.extramodule.se_loss:
            #    se_pred = True
            #    feat, se_out = self.extramodule(feat)
            #else:
            c4 = self.extramodule(c4)
        feat = self.c4conv(c4)
        c1_size = inputs[0].size()[2:]
        feat_up = interpolate(feat, c1_size, mode = 'bilinear', align_corners = True)
        fpn_features = [feat_up]
        # c4, c3, c2, c1
        for i in reversed(range(len(inputs) - 1)):
            feat_i = self.fpn_lateral[i](inputs[i])
            feat = interpolate(feat, feat_i.size()[2:], mode = 'bilinear', align_corners = True)
            feat = feat + feat_i
            # upsample to the same size with c1
            feat_up = interpolate(self.fpn_out[i](feat), c1_size, mode = 'bilinear', align_corners = True)
            fpn_features.append(feat_up)
        fpn_features = torch.cat(fpn_features, 1)
        #if se_pred:
        #    return (self.conv5(fpn_features), se_out)
        out = self.conv5(fpn_features)
        return out

class UperNetHead(FCFPNHead):
    def __init__(self, num_classes, norm_layer=nn.BatchNorm2d, in_channels=[256, 512, 1024, 2048],
                 channels=256, up_kwargs=None):
        in_channels[-1] = in_channels[-1] * 2
        super(UperNetHead, self).__init__(num_classes, norm_layer, in_channels,
                                          channels, up_kwargs)
        self.extramodule = PyramidPooling(in_channels[-1] // 2, norm_layer, up_kwargs)


if __name__ == "__main__":
    model = UperNetHead(out_channels=16)
    print(model)



