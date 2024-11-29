import torch
import torch.nn as nn
from torchvision import models

from models.sseg.uperhead import UperNetHead


class ConvNeXtTiny(nn.Module):
    def __init__(self, pretrained=True):
        super(ConvNeXtTiny, self).__init__()

        # 加载预训练的 ConvNeXt Tiny 模型
        self.model = models.convnext_tiny(pretrained=pretrained)

        # 重命名每个stage的层
        # Stage 1: Patch Embedding
        self.stage1 = self.model.features[0]  # Conv2d (3 -> 96, kernel=(4, 4), stride=(4, 4))

        # Stage 2: Layer Normalization
        self.stage2 = self.model.features[1]  # LayerNorm (96, 64, 64)

        # Stage 3: Downsampling with Conv2d
        self.stage3 = self.model.features[2]  # Conv2d (96 -> 192, kernel=(2, 2), stride=(2, 2))

        # Stage 4: Layer Normalization
        self.stage4 = self.model.features[3]  # LayerNorm (192, 32, 32)

        # Stage 5: Downsampling with Conv2d
        self.stage5 = self.model.features[4]  # Conv2d (192 -> 384, kernel=(2, 2), stride=(2, 2))

        # Stage 6: Layer Normalization
        self.stage6 = self.model.features[5]  # LayerNorm (384, 16, 16)

        # Stage 7: Downsampling with Conv2d
        self.stage7 = self.model.features[6]  # Conv2d (384 -> 768, kernel=(2, 2), stride=(2, 2))

        # Stage 8: Layer Normalization
        self.stage8 = self.model.features[7]  # LayerNorm (768, 8, 8)

    def forward(self, x):
        features = []
        x = self.stage1(x)  # 输入 stage1
        x = self.stage2(x)  # 经过 stage2  P1
        features.append(x)
        x = self.stage3(x)  # 经过 stage3
        x = self.stage4(x)  # 经过 stage4  P2
        features.append(x)
        x = self.stage5(x)  # 经过 stage5
        x = self.stage6(x)  # 经过 stage6  P3
        features.append(x)
        x = self.stage7(x)  # 经过 stage7
        x = self.stage8(x)  # 经过 stage8  P4
        features.append(x)
        return x, features


class ConvNeXt(nn.Module):
    def __init__(self, num_classes=6):
        super(ConvNeXt, self).__init__()
        self.backbone = ConvNeXtTiny(pretrained=True)

        self.classifier = UperNetHead(
            in_channels=[96, 192, 384, 768],
            num_classes=num_classes,
            channels=384
        )

        self.change_detection = nn.Sequential(
            nn.Conv2d(12, 6, kernel_size=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 2, kernel_size=1)
        )

    def forward(self, x, y):
        x, features_x = self.backbone(x)
        y, features_y = self.backbone(y)
        Seg1 = self.classifier(features_x)
        Seg2 = self.classifier(features_y)
        Change = self.change_detection(torch.cat([Seg1, Seg2], dim=1))
        return nn.functional.interpolate(Seg1, scale_factor=4, mode='bilinear', align_corners=True), \
            nn.functional.interpolate(Seg2, scale_factor=4, mode='bilinear', align_corners=True), \
            nn.functional.interpolate(Change, scale_factor=4, mode='bilinear', align_corners=True)


if __name__ == '__main__':
    model = ConvNeXt(6)

    # 输入数据
    input_tensor = torch.randn(8, 3, 256, 256)
    output = model(input_tensor, input_tensor)

    # 输出结果
    print(output[0].shape, output[1].shape, output[2].shape)  # 输出为 [1, 10]
