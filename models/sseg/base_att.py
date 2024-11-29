from models.backbone.swin_transformer import SwinTransformer
import torch
from torch import nn
import torch.nn.functional as F

def get_backbone(backbone, pretrained):
    if backbone == 'swin_tiny':
        backbone = SwinTransformer()
        backbone.init_weights(pretrained='/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/data/pretrained_models/swin_tiny.pth')
    elif backbone == 'swin_base':
        backbone = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32], ape=False)
        backbone.init_weights(pretrained='/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/data/pretrained_models/swin_base.pth')
    elif backbone == 'swin_small':
        backbone = SwinTransformer(embed_dim=96, depths=[2, 2, 18, 2], num_heads=[3, 6, 12, 24], ape=False)
        backbone.init_weights(pretrained='/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/data/pretrained_models/swin_small.pth')

    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.PReLU(),
        nn.Dropout(p=0.1),
    )

class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone, pretrained)
        
        self.sa1 = SpatialAttention()
        self.ca1 = ChannelAttention(in_channels=600)
        self.o1_conv1 = conv2d_bn(600, 384)
        self.bn_sa1 = nn.BatchNorm2d(48)

        self.sa2 = SpatialAttention()
        self.bn_sa2 = nn.BatchNorm2d(600)

        ''' 二值变化检测输出的最后一层 '''
        self.out_conv = nn.Sequential(
            nn.Conv2d(384 // 8, 384 // 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(384 // 8),
            nn.ReLU(True),
            nn.Conv2d(384 // 8, 1, kernel_size=1)
        )
        ''' 语义分割输出的最后一层 '''
        self.seg_conv = nn.Sequential(
            nn.Conv2d(150, 150 // 2, 3, padding=1),
            nn.BatchNorm2d(150 // 2),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(150 // 2, 6, 1, bias=True)
        )

        self.conv_fus = nn.Sequential(
            nn.Conv2d(150, 600, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(600),
            nn.ReLU(True)
        )

    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape
        
        x1 = self.backbone.base_forward(x1)[-1]
        x2 = self.backbone.base_forward(x2)[-1]
       
        out1 = self.head(x1)
        out2 = self.head(x2)

        ''' change information extraction module '''
        out_bin1 = torch.abs(x1 - x2)
        out_bin2 = torch.abs(out1 - out2)
        sam_1 = self.sa2(out_bin2)

        out_bin = self.bn_sa2(out_bin1 * sam_1)
        ''' 引入残差结构 '''
        out_bin = self.ca1(out_bin) * out_bin
        out_bin = self.conv_fus(out_bin2) + out_bin
        
        ''' spatial feature enhancement module '''
        out_bin = self.o1_conv1(out_bin)

        out_mid = self.head_bin(out_bin)
        sam = self.sa1(out_mid)

        out1 = self.seg_conv(out1 * sam)
        out2 = self.seg_conv(out2 * sam)

        out_bin = self.bn_sa1(out_mid * sam)
        out_bin = F.interpolate(self.out_conv(out_bin), size=(h, w), mode='bilinear', align_corners=False)
        out_bin = torch.sigmoid(out_bin)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)
        return out1, out2, out_bin.squeeze(1)

    def forward(self, x1, x2, tta=False):
        if self.backbone_name == 'hrnet_w40' or  'swin' in self.backbone_name:
            if not tta:
                return self.base_forward(x1, x2)
            else:
                out1, out2, out_bin = self.base_forward(x1, x2)
                out1 = F.softmax(out1, dim=1)
                out2 = F.softmax(out2, dim=1)
                out_bin = out_bin.unsqueeze(1)
                origin_x1 = x1.clone()
                origin_x2 = x2.clone()

                x1 = origin_x1.flip(2)
                x2 = origin_x2.flip(2)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(2)
                out2 += F.softmax(cur_out2, dim=1).flip(2)
                out_bin += cur_out_bin.unsqueeze(1).flip(2)

                x1 = origin_x1.flip(3)
                x2 = origin_x2.flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3)
                out2 += F.softmax(cur_out2, dim=1).flip(3)
                out_bin += cur_out_bin.unsqueeze(1).flip(3)

                x1 = origin_x1.transpose(2, 3).flip(3)
                x2 = origin_x2.transpose(2, 3).flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)
                out2 += F.softmax(cur_out2, dim=1).flip(3).transpose(2, 3)
                out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

                x1 = origin_x1.flip(3).transpose(2, 3)
                x2 = origin_x2.flip(3).transpose(2, 3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)
                out2 += F.softmax(cur_out2, dim=1).transpose(2, 3).flip(3)
                out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

                x1 = origin_x1.flip(2).flip(3)
                x2 = origin_x2.flip(2).flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)
                out2 += F.softmax(cur_out2, dim=1).flip(3).flip(2)
                out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

                out1 /= 6.0
                out2 /= 6.0
                out_bin /= 6.0

                return out1, out2, out_bin.squeeze(1)