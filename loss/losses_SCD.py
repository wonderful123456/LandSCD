import torch
# from segmentation_models_pytorch.losses import DiceLoss
# from models.segmentation_models_pytorch_myself.losses import DiceLoss
from torch import nn
# #ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
# This behaviour is the source of the following dependency conflicts.
# segmentation-models-pytorch 0.3.3 requires timm==0.9.2, but you have timm 0.6.13 which is incompatible.

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class BCEWithIgnoreLoss(nn.Module):
    def __init__(self, ignore_index=255, OHEM=False):
        super().__init__()
        self.ignore_index = ignore_index
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.OHEM = OHEM

    def forward(self, logits, target):
        if len(logits.shape) != len(target.shape) and logits.shape[1] == 1:
            logits = logits.squeeze(1)
            
        target = target.float()
        valid_mask = (target != self.ignore_index)
        loss = self.bce(logits, target)
        
        # OHEM
        if self.OHEM:
            loss_, _ = loss.contiguous().view(-1).sort()
            min_value = loss_[int(0.5 * loss.numel())]
            
            loss = loss[valid_mask]
            loss = loss[loss >= min_value]
        else:
            loss = loss[valid_mask]
        
        return loss.mean()
 

class BSCCLoss(nn.Module):
    '''model consistency between CVAPS module and semantic features'''
    def __init__(self, margin=2.0):
        super().__init__()
        self.m = margin
        self.eps = 1e-4

    def forward(self, pred, tar, tar_true):
        utar = 1 - tar # unchanged probability                 
        utar_true = 1 - tar_true # unchanged area truth
        n_u = ((utar_true == 1) * utar).sum() + self.eps
        n_c = ((tar_true == 1) * tar).sum() + self.eps
              
        loss = 0.5 * torch.sum(utar * (utar_true == 1) * torch.pow(pred, 2)) / n_u + \
            0.5 * torch.sum(tar * (tar_true == 1)* torch.pow(torch.clamp(self.m - pred, min=0.0), 2)) / n_c
 
        return loss / 128  # channel mean
    

class BCDLoss(nn.Module):
    def __init__(self,
                 losses=[BCEWithIgnoreLoss(), DiceLoss()],
                 loss_weight=[1, 1]):
        super(BCDLoss, self).__init__()
        self.loss_weights = loss_weight
        self.losses = losses

    def forward(self, logits, target):
        # 根据传入的logits（模型输出）和target（目标标签）计算每个损失函数的损失值，
        # 并根据损失权重对每个损失进行加权求和，最终返回总损失。
        losses = {}
        for i in range(len(self.losses)):
            loss = self.losses[i](logits, target)
            losses[i] = loss * self.loss_weights[i]
        losses["loss"] = sum(losses.values())
        return losses["loss"]
    

class AdditionalBackgroundSupervision(nn.Module):
    '''Background equal zero'''
    def __init__(self, ignore_index=7, scales=[0.5, 0.2]):
        super().__init__()
        self.ignore_index = ignore_index
        self.scales = scales
        self.bce_loss = BCEWithIgnoreLoss(ignore_index=ignore_index)
        self.dice_loss = DiceLoss()
        
    def forward(self, logits, target):
        logits_background = logits[:, 0, :, :]
        target_background = torch.where(target > 0, torch.ones_like(logits_background), torch.zeros_like(logits_background))
        
        loss_bce = self.bce_loss(logits_background, target_background) * self.scales[0]
        loss_dice = self.dice_loss(logits_background, target_background) * self.scales[1]
        
        return loss_bce + loss_dice