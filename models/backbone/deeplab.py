import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.modules.aspp import build_aspp
from models.modules.aspp_cbam import build_aspp_cbam
from models.modules.decoder import build_decoder, build_decoder1
from models.backbone import build_backbone, build_backbone1
# from modeling.canny import deeplab_canny
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet50', output_stride=16, num_classes=16,
                 sync_bn=True, freeze_bn=False,cbam=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        if cbam == True:
            self.aspp = build_aspp_cbam(backbone, output_stride, BatchNorm)
        else:
            self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

                                
class DeepLab1(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False, cbam=False):
        super(DeepLab1, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone1(backbone, output_stride, BatchNorm)
        
        
        if cbam == True:
            self.aspp = build_aspp_cbam(backbone, output_stride, BatchNorm)
        else:
            self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder1(num_classes, backbone, BatchNorm)
        
        self.freeze_bn = freeze_bn

    def forward(self, input):
        #输入的input为batch_size* 6 * 340 * 360
        x, low_level_feat,x_edge = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, x_edge)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x
        
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p                                
class DeepLab2(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=16,
                 sync_bn=True, freeze_bn=False, cbam=False):
        super(DeepLab1, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone1(backbone, output_stride, BatchNorm)
        if cbam == True:
            self.aspp = build_aspp_cbam(backbone, output_stride, BatchNorm)
        else:
            self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder1(num_classes, backbone, BatchNorm)
        
        self.freeze_bn = freeze_bn

    def forward(self, input):
        #输入的input为batch_size* 6 * 340 * 360
        x, low_level_feat,x_edge = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat, x_edge)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x
        
    
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p  
from thop import profile
import time
if __name__ == "__main__":
    device = torch.device("cuda:0")
    # model = DeepLab(backbone='mobilenet', output_stride=16)
    model = DeepLab(backbone='resnet50', output_stride=16)
    model.to(device)
#     model.eval()
    input = torch.randn(4, 3, 256, 256).to(device)
    # start = time.time()
    output = model(input)
    # torch.cuda.synchronize()
    # end = time.time()
    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #
    # iterations = 10
    # # GPU预热
    # for _ in range(50):
    #     _ = model(input)
    #
    # # 测速
    # times = torch.zeros(iterations)  # 存储每轮iteration的时间
    # with torch.no_grad():
    #     for iter in range(iterations):
    #         starter.record()
    #         _ = model(input)
    #         ender.record()
    #         # 同步GPU时间
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)  # 计算时间
    #         times[iter] = curr_time
    #         # print(curr_time)
    #
    # mean_time = times.mean().item()
    # print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))
    #
    # # print('infer_time:', end-start)
    # # print(output.size())
    # flops, params = profile(model, (input,))
    # print('flops: %.2f M, params: %.2f M' % (flops / 1e6, params / 1e6))

