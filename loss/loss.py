import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
 
eps = 1e-7
import os

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1  # pt有大于1的值
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow((1 - pt), one_sided_gamma)  # 有nan
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=True):
        self.ignore_index = ignore_index
        self.weight_old = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.weight = []
        root_path = r'E:/Sentinel-2-data/毕节市/赫章县/Datasets_HZ'

        file = open(os.path.join(root_path, 'class_count.txt'))
        all_lines = file.readlines()
        for line in all_lines:
            self.weight.append(line.strip())
        print('self.weight is ', self.weight)
        sorted_list = sorted(self.weight)
        from statistics import median
        median_value = median(sorted_list)
        print('median value is ', median_value)
        new_weight_list = []
        for num in self.weight:
            new_weight_list.append(int(median_value) / int(num))
            print('median_value / num is ', int(median_value) / int(num))
        self.weight = new_weight_list
        print('self.weight after is ', self.weight)

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # nn.MultiLabelSoftMarginLoss
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, t,  c, h, w = logit.size()
        # print('FocalLoss weight is', weight)
        device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weight = torch.FloatTensor(self.weight).to(device)
        criterion = nn.CrossEntropyLoss(weight=None, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # print('logit shape is ', logit.shape)  # [2, 6, 3, 7, 7]
        # print('target shape is ', target.long().shape)  # [2, 6, 7, 7]
        # print('done')
        logit = logit.reshape(logit.size(0)*logit.size(1), logit.size(2), logit.size(3), logit.size(4))
        target = target.reshape(target.size(0)*target.size(1), target.size(2), target.size(3))
        # print('logit device is' ,logit.device)
        # print('target device is' ,target.device)

        # logit = logit.permute(0,2,3,4,1)
        # target = target.permute(0,2,3,1)

        # logit = logit.permute(0,2,3,1)
        # target = target.permute(0,2,1)

        # print('logit shape is ', logit.shape)
        # print('target shape is ', target.long().shape)


        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

def weighted_bce(bd_pre, target):
    n, l, c, h, w = bd_pre.size()
    # log_p = bd_pre.permute(0,2,3,1).contiguous().view(1, -1)  # [1, 131072]
    log_p = bd_pre.permute(0, 1, 3, 4, 2).contiguous().view(1, -1)  # [1, 131072]
    # target_t = target.view(1, -1)
    target_t = target.reshape(1,
                              target.shape[0] * target.shape[1] * target.shape[2] * target.shape[3])  # [1, 131072

    pos_index = (target_t == 1)  # [1, 131072]
    neg_index = (target_t == 0)  # [1, 131072]

    weight = torch.zeros_like(log_p)  # [1, 131072]
    pos_num = pos_index.sum()  # tensor(131072)
    neg_num = neg_index.sum()  # tensor(0)
    sum_num = pos_num + neg_num  # tensor(131072)
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    loss = F.binary_cross_entropy_with_logits(log_p, target_t, weight, reduction='mean')  # 空的，由于输入的是全1 tensor

    return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5

        # 将预测结果和标签转换为二值图像
        B = pred.shape[0]
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        target = (target > 0.5).float()

        # 计算Dice系数
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target)
        dice = (2 * intersection + smooth) / (union + smooth)

        # 计算Dice损失
        loss = (1 - dice) / B

        return loss

class BondaryLoss(nn.Module):
    def __init__(self, coeff_bce=20.0):
        super(BondaryLoss, self).__init__()
        self.coeff_bce = coeff_bce

    def forward(self, bd_pre, bd_gt):
        bce_loss = self.coeff_bce * weighted_bce(bd_pre, bd_gt)
        loss = bce_loss

        return loss

# def focal_loss(input_values, gamma):
#     """Computes the focal loss"""
#     p = torch.exp(-input_values)
#     loss = (1 - p) ** gamma * input_values
#     return loss.mean()

# def focal_loss(logits, labels,  alpha=0.05, gamma=2, weight=None, reduction='mean',ignore_index=255):
#     """Compute the focal loss between `logits` and the ground truth `labels`.
#
#     Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
#     where pt is the probability of being classified to the true class.
#     pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).
#
#     Args:
#       labels: A float tensor of size [batch, num_classes].
#       logits: A float tensor of size [batch, num_classes].
#       alpha: A float tensor of size [batch_size]
#         specifying per-example weight for balanced cross entropy.
#       gamma: A float scalar modulating loss from hard and easy examples.
#
#     Returns:
#       focal_loss: A float32 scalar representing normalized total loss.
#     """
#     # BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
#     CLoss =F.cross_entropy(input=logits, target=labels, weight=weight,
#                     ignore_index=ignore_index, reduction=reduction)
#
#     if gamma == 0.0:
#         modulator = 1.0
#     else:
#         modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
#             torch.exp(-1.0 * logits)))
#
#     loss = modulator * CLoss  # BCLoss->CLoss
#
#     weighted_loss = alpha * loss
#     focal_loss = torch.sum(weighted_loss)
#
#     focal_loss /= torch.sum(labels)
#     return focal_loss

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算 Softmax 概率
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # 计算预测概率
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss  # Focal Loss 计算

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# class FocalLoss(nn.Module):
#     def __init__(self, cls_num_list=None, weight=None, gamma=0.):
#         super(FocalLoss, self).__init__()
#         assert gamma >= 0
#         self.gamma = gamma
#         self.weight = weight
#
#     def _hook_before_epoch(self, epoch):
#         pass
#
#     def forward(self, output_logits, target):
#         return focal_loss(F.cross_entropy(output_logits, target, reduction='none', weight=self.weight), self.gamma)

class CrossEntropyLoss(nn.Module):
    def __init__(self, cls_num_list=None, reweight_CE=False):
        super().__init__()
        if reweight_CE:
            idx = 1 # condition could be put in order to set idx
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            self.per_cls_weights = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
        else:
            self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.per_cls_weights is not None:
            self.per_cls_weights = self.per_cls_weights.to(device)
        
        return self

    def forward(self, output_logits, target): # output is logits
        return F.cross_entropy(output_logits, target, weight=self.per_cls_weights)

class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list=None, max_m=0.5, s=30, reweight_epoch=-1):
        super().__init__()
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.
            self.m_list = None
        else:
            self.reweight_epoch = reweight_epoch
            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            assert s > 0
            self.s = s
            if reweight_epoch != -1:
                # CB loss
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)  # * class number
                # the effect of per_cls_weights / np.sum(per_cls_weights) can be described in the learning rate so the math formulation keeps the same.
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)
            else:
                self.per_cls_weights_enabled = None
                self.per_cls_weights = None

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)

        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights = self.per_cls_weights_enabled
            else:
                self.per_cls_weights = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)  # one-hot index
         
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1)) 
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s 

        final_output = torch.where(index, x_m, x) 
        return final_output

    def forward(self, output_logits, target):
        if self.m_list is None:
            return F.cross_entropy(output_logits, target)
        
        final_output = self.get_final_output(output_logits, target)
        return F.cross_entropy(final_output, target, weight=self.per_cls_weights)

class RIDELoss(nn.Module):
    def __init__(self, cls_num_list=None, base_diversity_temperature=1.0, max_m=0.5, s=30, reweight=True, reweight_epoch=-1, 
        base_loss_factor=1.0, additional_diversity_factor=-0.2, reweight_factor=0.05):
        super().__init__()
        self.base_loss = F.cross_entropy
        self.base_loss_factor = base_loss_factor
        if not reweight:
            self.reweight_epoch = -1
        else:
            self.reweight_epoch = reweight_epoch

        # LDAM is a variant of cross entropy and we handle it with self.m_list.
        if cls_num_list is None:
            # No cls_num_list is provided, then we cannot adjust cross entropy with LDAM.

            self.m_list = None
            self.per_cls_weights_enabled = None
            self.per_cls_weights_enabled_diversity = None
        else:
            # We will use LDAM loss if we provide cls_num_list.

            m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
            m_list = m_list * (max_m / np.max(m_list))
            m_list = torch.tensor(m_list, dtype=torch.float, requires_grad=False)
            self.m_list = m_list
            self.s = s
            assert s > 0
            
            if reweight_epoch != -1:
                idx = 1 # condition could be put in order to set idx
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                self.per_cls_weights_enabled = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False)   # 这个是logits时算CE loss的weight
            else:
                self.per_cls_weights_enabled = None

            cls_num_list = np.array(cls_num_list) / np.sum(cls_num_list)
            C = len(cls_num_list)  # class number
            per_cls_weights = C * cls_num_list * reweight_factor + 1 - reweight_factor   #Eq.3

            # Experimental normalization: This is for easier hyperparam tuning, the effect can be described in the learning rate so the math formulation keeps the same.
            # At the same time, the 1 - max trick that was previously used is not required since weights are already adjusted.
            per_cls_weights = per_cls_weights / np.max(per_cls_weights)    # the effect can be described in the learning rate so the math formulation keeps the same.

            assert np.all(per_cls_weights > 0), "reweight factor is too large: out of bounds"
            # save diversity per_cls_weights
            self.per_cls_weights_enabled_diversity = torch.tensor(per_cls_weights, dtype=torch.float, requires_grad=False).cuda()  # 这个是logits时算diversity loss的weight

        self.base_diversity_temperature = base_diversity_temperature
        self.additional_diversity_factor = additional_diversity_factor

    def to(self, device):
        super().to(device)
        if self.m_list is not None:
            self.m_list = self.m_list.to(device)
        
        if self.per_cls_weights_enabled is not None:
            self.per_cls_weights_enabled = self.per_cls_weights_enabled.to(device)

        if self.per_cls_weights_enabled_diversity is not None:
            self.per_cls_weights_enabled_diversity = self.per_cls_weights_enabled_diversity.to(device)

        return self

    def _hook_before_epoch(self, epoch):
        if self.reweight_epoch != -1:
            self.epoch = epoch

            if epoch > self.reweight_epoch:
                self.per_cls_weights_base = self.per_cls_weights_enabled
                self.per_cls_weights_diversity = self.per_cls_weights_enabled_diversity
            else:
                self.per_cls_weights_base = None
                self.per_cls_weights_diversity = None

    def get_final_output(self, output_logits, target):
        x = output_logits

        index = torch.zeros_like(x, dtype=torch.uint8, device=x.device)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.float()
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m * self.s

        final_output = torch.where(index, x_m, x)
        return final_output

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)

        loss = 0

        # Adding RIDE Individual Loss for each expert
        for logits_item in extra_info['logits']:  
            ride_loss_logits = output_logits if self.additional_diversity_factor == 0 else logits_item
            if self.m_list is None:
                loss += self.base_loss_factor * self.base_loss(ride_loss_logits, target)
            else:
                final_output = self.get_final_output(ride_loss_logits, target)
                loss += self.base_loss_factor * self.base_loss(final_output, target, weight=self.per_cls_weights_base)
            
            base_diversity_temperature = self.base_diversity_temperature

            if self.per_cls_weights_diversity is not None:
                diversity_temperature = base_diversity_temperature * self.per_cls_weights_diversity.view((1, -1))
                temperature_mean = diversity_temperature.mean().item()
            else:
                diversity_temperature = base_diversity_temperature
                temperature_mean = base_diversity_temperature
            
            output_dist = F.log_softmax(logits_item / diversity_temperature, dim=1)
            with torch.no_grad():
                # Using the mean takes only linear instead of quadratic time in computing and has only a slight difference so using the mean is preferred here
                mean_output_dist = F.softmax(output_logits / diversity_temperature, dim=1)
            
            loss += self.additional_diversity_factor * temperature_mean * temperature_mean * F.kl_div(output_dist, mean_output_dist, reduction='batchmean')
        
        return loss


class ModifiedAsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super(ModifiedAsymmetricFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        # pt = torch.exp(-BCE_loss)
        criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255,
                                        size_average=True)
        CE_loss = criterion(inputs, targets.long())
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * CE_loss  # BCE_loss

        # Ignore background class
        mask = (targets != self.ignore_index).float()
        F_loss = F_loss * mask

        return F_loss.mean()


class ModifiedAsymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, delta=0.1, ignore_index=-100):
        super(ModifiedAsymmetricFocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = (inputs.argmax(dim=1) * targets).sum(dim=1).sum(dim=1)
        FN = ((1 - inputs.argmax(dim=1)) * targets).sum(dim=1).sum(dim=1)
        FP = (inputs.argmax(dim=1) * (1 - targets)).sum(dim=1).sum(dim=1)

        mTI = (TP + self.delta * FN) / (TP + self.delta * FN + (1 - self.delta) * FP + 1e-6)

        # BCE_loss = -torch.log(mTI)
        FTL_loss = self.alpha * (1 - mTI) + self.beta * (1 - mTI) ** self.delta

        # Ignore background class
        mask = (targets != self.ignore_index).float().sum(dim=1).sum(dim=1)
        FTL_loss = FTL_loss * mask

        return FTL_loss.mean()


class UnifiedFocalLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.7, delta=0.1, gamma=2, lammbda=0.7, ignore_index=-100):
        super(UnifiedFocalLoss, self).__init__()
        self.mFL = ModifiedAsymmetricFocalLoss(alpha, gamma, ignore_index)
        self.mFTL = ModifiedAsymmetricFocalTverskyLoss(beta, delta, ignore_index)
        self.lammbda = lammbda

    def forward(self, inputs, targets):
        mFL_loss = self.mFL(inputs, targets)
        mFTL_loss = self.mFTL(inputs, targets)
        uFL_loss = self.lammbda * mFL_loss + (1 - self.lammbda) * mFTL_loss

        return uFL_loss


class DiverseExpertLoss(nn.Module):
    def __init__(self, cls_num_list=[13.812, 71.9594, 7.5256, 4.4919, 0.8043, 1.4077], max_m=0.5, s=30, tau=2):
        super().__init__()

        # self.base_loss = ModifiedAsymmetricFocalLoss(1, 2, 20)#F.cross_entropy #F.cross_entropy
        # self.base_loss = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8)
        self.base_loss = FocalLoss(alpha=0.05)
        # self.base_loss = UnifiedFocalLoss(alpha=0.7, beta=0.3, delta=0.3, gamma=4 // 3)

        prior = np.array(cls_num_list) / np.sum(cls_num_list)
        self.prior = torch.tensor(prior).float().cuda()
        self.C_number = len(cls_num_list)  # class number
        self.s = s
        self.tau = tau

    def inverse_prior(self, prior):
        value, idx0 = torch.sort(prior)
        _, idx1 = torch.sort(idx0)
        idx2 = prior.shape[0] - 1 - idx1  # reverse the order
        inverse_prior = value.index_select(0, idx2)

        return inverse_prior

    def forward(self, output_logits, target, extra_info=None):
        if extra_info is None:
            return self.base_loss(output_logits, target)  # output_logits indicates the final prediction

        loss = 0

        # Obtain logits from each expert
        expert1_logits = extra_info[:,0,:,:,:]
        expert2_logits = extra_info[:,1,:,:,:]
        expert3_logits = extra_info[:,2,:,:,:]

        # Softmax loss for expert 1
        loss += self.base_loss(expert1_logits, target)
        # print('xy shape is', torch.log(self.prior + 1e-9).shape)
        # Balanced Softmax loss for expert 2
        prior = torch.log(self.prior + 1e-9).view(1, 6, 1, 1).expand(8, -1, 256, 256)
        expert2_logits = expert2_logits + prior#.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).repeat(8, 1, 256, 256)
        loss += self.base_loss(expert2_logits, target)

        # Inverse Softmax loss for expert 3
        inverse_prior = self.inverse_prior(self.prior)

        expert3_logits = expert3_logits + torch.log(self.prior + 1e-9).view(1, 6, 1, 1).expand(8, -1, 256, 256) - self.tau * torch.log(inverse_prior + 1e-9).view(1, 6, 1, 1).expand(8, -1, 256, 256)
        loss += self.base_loss(expert3_logits, target)

        return loss


 
# class DiverseExpertLoss(nn.Module):
#     def __init__(self, cls_num_list=None,  max_m=0.5, s=30, tau=2):
#         super().__init__()
#         self.base_loss = F.cross_entropy
#
#         prior = np.array(cls_num_list) / np.sum(cls_num_list)
#         self.prior = torch.tensor(prior).float().cuda()
#         self.C_number = len(cls_num_list)  # class number
#         self.s = s
#         self.tau = tau
#
#     def inverse_prior(self, prior):
#         value, idx0 = torch.sort(prior)
#         _, idx1 = torch.sort(idx0)
#         idx2 = prior.shape[0]-1-idx1 # reverse the order
#         inverse_prior = value.index_select(0,idx2)
#
#         return inverse_prior
#
#     def forward(self, output_logits, target, extra_info=None):
#         if extra_info is None:
#             return self.base_loss(output_logits, target)  # output_logits indicates the final prediction
#
#         loss = 0
#
#         # Obtain logits from each expert
#         expert1_logits = extra_info['logits'][0]
#         expert2_logits = extra_info['logits'][1]
#         expert3_logits = extra_info['logits'][2]
#
#         # Softmax loss for expert 1
#         loss += self.base_loss(expert1_logits, target)
#
#         # Balanced Softmax loss for expert 2
#         expert2_logits = expert2_logits + torch.log(self.prior + 1e-9)
#         loss += self.base_loss(expert2_logits, target)
#
#         # Inverse Softmax loss for expert 3
#         inverse_prior = self.inverse_prior(self.prior)
#         expert3_logits = expert3_logits + torch.log(self.prior + 1e-9) - self.tau * torch.log(inverse_prior+ 1e-9)
#         loss += self.base_loss(expert3_logits, target)
#
#         return loss

if __name__ == '__main__':
    B, H, W, T = 2, 256, 256, 8
    C = 5
    # pred = torch.randint(low=0, high=6, size=(B, T, C, H, W), dtype=torch.int64, layout=torch.strided,
    #                            device=None, requires_grad=False)
    # pred = pred.argmax(dim=2).detach().cpu()
    pred = torch.randn(2, 5, 256, 256)#.argmax(dim=1)
    # pred = pred.argmax(dim=2).reshape(B*T, H, W)


    # gt = torch.randint(low=0, high=6, size=(B, T, H, W), dtype=torch.int64, layout=torch.strided,
    #                            device=None, requires_grad=False).reshape(B*T, H, W)
    gt = torch.ones(2, 256, 256).reshape(B, H, W).long()
    num_class_list = [100, 1, 10, 1000, 2]
    # loss = DiverseExpertLoss(cls_num_list=num_class_list)
    # loss = FocalLoss(gamma=2)
    loss = FocalLoss(alpha=0.05)
    print(loss(pred, gt))
    # print(gt)

 
     