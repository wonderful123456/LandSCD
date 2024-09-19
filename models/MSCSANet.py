import math
import torch.utils.model_zoo as model_zoo

import torch.nn as nn
import torch
import torch.nn.functional as F
from models.ms_attention import LmsPAM_Module, LmsCAM_Module, LmsCAM_Context_Module, LmsPAM_Context_Module, PAM_Module, CAM_Module, MsPAM_Module
# import modeling.ms_attention
__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',

}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, rate=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=rate, dilation=rate, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        rates = [1, 2, 4]
        self.layer4 = self._make_deeplabv3_layer(block, 512, layers[3], rates=rates, stride=1)  # stride 2 => stride 1
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_deeplabv3_layer(self, block, planes, blocks, rates, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, rate=rates[i]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 models.
    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(ResNet50, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def base_forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out
    
"""Dual Attention Network"""
class MsResNet50(nn.Module):
    def __init__(self, pretrained=True):
        """Declare all needed layers."""
        super(MsResNet50, self).__init__()
        self.model = resnet50(pretrained=pretrained)
        self.relu = self.model.relu  # Place a hook

        layers_cfg = [4, 5, 6, 7]
        self.blocks = []
        for i, num_this_layer in enumerate(layers_cfg):
            self.blocks.append(list(self.model.children())[num_this_layer])

    def forward(self, x):
        feature_map = []
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            feature_map.append(x)

        out = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], -1)

        return feature_map, out

class DANet(ResNet50):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass, aux=True, **kwargs):
        super(DANet, self).__init__(nclass)
        self.head = _DAHead(2048, nclass, aux, **kwargs)
        self.aux = True
        self.__setattr__('exclusive', ['head'])

    def forward(self, x):
        size = x.size()[2:]
        feature_map,_ = self.base_forward(x)
        c3,c4 = feature_map[2],feature_map[3]

        outputs = []
        x = self.head(c4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            #print('x[1]:{}'.format(x[1].shape))
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs
    
    
class MSCSANet(ResNet50):
    def __init__(self, nclass, aux=True, **kwargs):
        super(MSCSANet, self).__init__(nclass)
#         self.head = _DAHead(2048, nclass, aux, **kwargs)
        self.head = _MDAHead(2048, nclass, aux, **kwargs)
        self.backbone1 = ResNet50()
        self.backbone2 = ResNet50()
        self.aux = True
        self.__setattr__('exclusive', ['head'])

    def forward(self, inputs):
        x = inputs[:,0:3,:,:]
        y = inputs[:, 3:6, :, :]
        z = inputs[:, 6:9 :, :]
        size = x.size()[2:]
        feature_map,_ = self.base_forward(x)
        feature_map1,_ = self.backbone1.base_forward(y)
        feature_map2,_ = self.backbone2.base_forward(z)
        
        c3,c4 = feature_map[2],feature_map[3]
        d3, d4 = feature_map1[2],feature_map1[3]
        e3, e4 = feature_map2[2],feature_map2[3]

        x = self.head(c4,d4,e4)
        output = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        return output

class PSMDANet(ResNet50):

    def __init__(self, nclass, aux=True, **kwargs):
        super(PSMDANet, self).__init__(nclass)
#         self.head = _DAHead(2048, nclass, aux, **kwargs)
        self.head = _PSMDAHead(2048, nclass, aux, **kwargs)
        self.backbone1 = ResNet50()
        self.backbone2 = ResNet50()
        self.aux = True
        self.__setattr__('exclusive', ['head'])

    def forward(self, inputs):
        x = inputs[:,0:3,:,:]
        y = inputs[:, 3:6, :, :]
        z = inputs[:, 6:9 : , :]
        size = x.size()[2:]
        feature_map,_ = self.base_forward(x)
        feature_map1,_ = self.backbone1.base_forward(y)
        feature_map2,_ = self.backbone2.base_forward(z)
        
        c3,c4 = feature_map[2],feature_map[3]
        d3, d4 = feature_map1[2],feature_map1[3]
        e3, e4 = feature_map2[2],feature_map2[3]
        outputs = []
        x = self.head(c4,d4,e4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            #print('x[1]:{}'.format(x[1].shape))
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs
   
    
class MDANet_Context(ResNet50):
    r"""Pyramid Scene Parsing Network
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.
    Reference:
        Jun Fu, Jing Liu, Haijie Tian, Yong Li, Yongjun Bao, Zhiwei Fang,and Hanqing Lu.
        "Dual Attention Network for Scene Segmentation." *CVPR*, 2019
    """

    def __init__(self, nclass,context = 'L', aux=True, **kwargs):
        super(MDANet_Context, self).__init__(nclass)
#         self.head = _DAHead(2048, nclass, aux, **kwargs)
        self.head = _MDAHead_Context(2048, nclass, aux, **kwargs)
        self.backbone1 = ResNet50()
        self.backbone2 = ResNet50()
        self.aux = True
        if context == 'L':
            self.low = 3
            self.high = 6
        else:
            self.low = 6
            self.high = 9
        self.__setattr__('exclusive', ['head'])

    def forward(self, inputs):
        x = inputs[:,0:3,:,:]
        y = inputs[:, self.low : self.high, :, :]
        size = x.size()[2:]
        feature_map,_ = self.base_forward(x)
        feature_map1,_ = self.backbone1.base_forward(y)
        
        c3,c4 = feature_map[2],feature_map[3]
        d3, d4 = feature_map1[2],feature_map1[3]
        outputs = []
        x = self.head(c4,d4)
        x0 = F.interpolate(x[0], size, mode='bilinear', align_corners=True)
        outputs.append(x0)

        if self.aux:
            #print('x[1]:{}'.format(x[1].shape))
            x1 = F.interpolate(x[1], size, mode='bilinear', align_corners=True)
            x2 = F.interpolate(x[2], size, mode='bilinear', align_corners=True)
            outputs.append(x1)
            outputs.append(x2)
        return outputs
    
class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y, z):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out

class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = PAM_Module(inter_channels)
        self.cam = CAM_Module(inter_channels)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)
    
class _MDAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_MDAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
       
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
#         self.attention = LmsPAM_CAM_Layer(inter_channels)
        self.pam = LmsPAM_Module(inter_channels)
#         self.pam = MsPAM_Module(inter_channels)
        self.cam =LmsCAM_Module(inter_channels)
        
#         self.pam = _PositionAttentionModule(inter_channels, **kwargs)
#         self.cam1 = _ChannelAttentionModule(**kwargs)
        
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x, y, z):
        feat_p = self.conv_p1(x)
        feat_p1 = self.conv_p12(y)
        feat_p2 = self.conv_p13(z)
        feat_p = self.pam(feat_p,feat_p1 ,feat_p2)
#         feat_p = self.pam1(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c1 = self.conv_c12(y)
        feat_c2 = self.conv_c13(z)
        feat_c = self.cam(feat_c,feat_c1, feat_c2)
#         feat_c = self.cam1(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)
    
class _PSMDAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_PSMDAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
       
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
     
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels*3, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels*3, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x, y, z):
        feat_p = self.conv_p1(x)
        feat_p1 = self.conv_p12(y)
        feat_p2 = self.conv_p13(z)
        
        feat_p = torch.cat((feat_p, feat_p1, feat_p2), 1)
        
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c1 = self.conv_c12(y)
        feat_c2 = self.conv_c13(z)
        feat_c = torch.cat((feat_c, feat_c1, feat_c2), 1)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)

class _MDAHead_Context(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_MDAHead_Context, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_p13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
       
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c12 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c13 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = LmsPAM_Context_Module(inter_channels)
        self.cam =LmsCAM_Context_Module(inter_channels) 
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x, y):
        feat_p = self.conv_p1(x)
        feat_p1 = self.conv_p12(y)
#         feat_p2 = self.conv_p13(z)
        feat_p = self.pam(feat_p,feat_p1)
#         feat_p = self.pam1(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c1 = self.conv_c12(y)
#         feat_c2 = self.conv_c13(z)
        feat_c = self.cam(feat_c,feat_c1)
#         feat_c = self.cam1(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)


def get_danet( backbone='resnet50',n_class = 16, pretrained_base=True, **kwargs):
    model = DANet(n_class, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model

def get_MSCSANet( backbone='resnet50',n_class = 16, pretrained_base=True, **kwargs):
    model = MSCSANet(n_class, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model

def get_psmdanet(backbone='resnet50',n_class = 16, pretrained_base=True, **kwargs):
    model = PSMDANet(n_class, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model

def get_mdanet_context( backbone='resnet50',n_class = 16, context = 'L',pretrained_base=True, **kwargs):
    model = MDANet_Context(n_class,context = context, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    return model
if __name__ == '__main__':
    img = torch.randn(1, 3, 480, 480)

    model = ResNet50()
    print(model)
    # outputs = models.base_forward(img)
    # print(outputs[0].shape)
    #print(outputs)
