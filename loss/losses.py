import torch
import torch.nn.functional as F


# add by ljc
import numpy as np
def focal_loss(labels, logits, alpha, gamma, weight=None, reduction='mean',ignore_index=255):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    # BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")
    CLoss =F.cross_entropy(input=logits, target=labels, weight=weight,
                    ignore_index=ignore_index, reduction=reduction)

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
            torch.exp(-1.0 * logits)))

    loss = modulator * CLoss  # BCLoss->CLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

def CB_loss(labels, logits, samples_per_cls, no_of_classes=1, loss_type='focal', beta=0.9999, gamma=2.0, device='cuda'):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    print(effective_num)
    print('effective_num shape is ', effective_num.shape)
    weights = (1.0 - beta) / np.array(effective_num)
    print(weights)
    weights = weights / np.sum(weights) * no_of_classes
    print('weights shape origin is ', weights.shape)


    labels = labels.reshape(labels.shape[0],labels.shape[1]*labels.shape[2])
    print(labels.shape)

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    labels_one_hot = labels_one_hot.to(device)  # [64, 100]
    print('labels_one_hot shape is ', labels_one_hot.shape)

    weights = torch.tensor(weights).float()
    print('weights shape first is ', weights.shape)
    weights = weights.unsqueeze(0)
    print('weights shape second is ', weights.shape)
    weights = weights.repeat(labels.shape[0], 1).to(device)
    print('weights shape last is ', weights.shape)
    print('weights shape is', weights.shape)  # [64, 100]
    print('labels shape is', labels.shape)
    print(weights)
    weights = weights * labels
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)  # weights->alpha

    if loss_type == "focal":
        cb_loss = focal_loss(labels, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss

def get_img_num_per_cls(cls_num, total_num, imb_type, imb_factor):  # 获取每类的数量
    # This function is excerpted from a publicly available code [commit 6feb304, MIT License]:
    # https://github.com/kaidic/LDAM-DRW/blob/master/imbalance_cifar.py
    img_max = total_num / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    print(img_num_per_cls)  # 类别0->未变化类的像素数为32768，类别1->变化类的像素数为327,[32768, 327]
    # print('img_num_per_cls shape is', len(img_num_per_cls))
    return img_num_per_cls

# add by ljc
def Balace_CE_los(input, target,device='cuda'):
    # input = input.view(input.shape[0], -1)
    # target = target.view(target.shape[0], -1).long()
    # input = input.view(input.shape[0], -1)  # input -> [8,2, 256, 256]
    # target_O = target
    target = target.view(target.shape[0], -1)  # [8, 65536]
    # print('input shape is ', input.shape)
    # print('target shape is ', target[4].shape)  # target[i]->65536
    # print('target shape[1] is ', target.shape[1])  target.shape[1]->65536
    loss = 0.0
    # version2
    for i in range(input.shape[0]):
        beta = 1 - torch.sum(target[i]) / target.shape[1]
        # print('beta is ', beta)
        x = torch.max(torch.log(input[i]).to(device), torch.tensor([-100.0]).to(device))
        y = torch.max(torch.log(1 - input[i]).to(device), torch.tensor([-100.0]).to(device))
        # print('x shape is ', x.shape)   # [2, 256, 256]
        # print('y shape is ', y.shape)   # [2, 256, 256]
        l = -(beta * target[i] * x.view(x.shape[0], -1) + (1 - beta) * (1 - target[i]) * y.view(y.shape[0], -1))
        loss += torch.sum(l)
    return loss

import torch.nn as nn
def FocalLoss(logit, target, gamma=2, alpha=0.5):
    n, c, h, w = logit.size()
    criterion = nn.CrossEntropyLoss(weight=None, ignore_index=255,
                                size_average=True)
    # if cuda:
    criterion = criterion.cuda()

    # target = target.reshape(n, h, w)
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    # print('target shape is ', target.shape)

    logpt = -criterion(logit, target.long())
    pt = torch.exp(logpt)
    # print('pt is ', pt)

    effective_num = 1.0 - np.power(0.99, [32768, 327])
    print('effective_num shape is ', effective_num.shape)
    weights = (1.0 - 0.99) / np.array(effective_num)
    weights = weights / np.sum(weights) * 2

    # labels = target.reshape(target.shape[0], target.shape[1] * target.shape[2])
    # print(labels.shape)

    # labels_one_hot = F.one_hot(labels, no_of_classes).float()
    #
    # labels_one_hot = labels_one_hot.to(device)  # [64, 100]
    # print('labels_one_hot shape is ', labels_one_hot.shape)

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(target.shape[0], 1)
    weights = weights * target
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, 2)  # weights->alpha

    if alpha is not None:
        logpt *= weights
    loss = -((1 - pt) ** gamma) * logpt
    # print('loss is ', loss)

    # if self.batch_average:
    loss /= n

    return loss


# origin 原来就有
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """

    target = target.long()

    # print('target shape is', target.shape)  # train->[8,1,256,256,3]
    # print('target.shape[1:] shape is ', target.shape[1:])

    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        # print(type(target))

        # print('input shape origin is ', input.shape)
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    # print('input shape is ', input.shape)  # [8, 2, 256, 256]
    # print('target shape is ', target.shape)  # [8, 256, 256]
    # return focal_loss(target, input, weight, 2.0)
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

# return CB_loss(labelList, logits, img_num_per_cls, 2, "softmax", 0.9999, 2.0, device)

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

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
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        # print('logit shape is ', logit.shape)  # [2, 6, 3, 7, 7]
        # print('target shape is ', target.long().shape)  # [2, 6, 7, 7]
        # print('done')
        logit = logit.reshape(logit.size(0)*logit.size(1), logit.size(2), logit.size(3), logit.size(4))
        target = target.reshape(target.size(0)*target.size(1), target.size(2), target.size(3))

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

def myCrossEntropyLoss(x, label):
    loss = []

    for i, cls in enumerate(label):
        # 对应上面公式的 -x[class]
        x_class = -x[i][cls]
        # 对应上面公式的 log(sum)
        log_x_j = np.log( sum([ np.exp(j) for j in x[i] ]) )
        loss.append(x_class + log_x_j)

    return np.mean(loss)

def get_freq(label):
    class_count = np.zeros((5))  # array([0., 0., 0., 0., 0.])
    for i in range(5):
        if label == i:
            class_count[i] = 1  # 有就对应类别位置+1
    return class_count





if __name__ == '__main__':
    import numpy as np
    x = torch.randn(16, 5, 256, 256)

    # 分类标签
    label = torch.ones(16, 256, 256).long()

    weights = [1.0, 1000, 10000, 10, 2000]
    # freq = 0
    # freq += get_freq(label)
    # freq = freq / np.sum(freq)
    # weight = np.median(freq) / freq  # np.median中位数
    class_weights = torch.FloatTensor(weights)
    loss = nn.CrossEntropyLoss(weight=class_weights)
    print(loss(x, label))

    # print("my CrossEntropyLoss output: %.4f"% myCrossEntropyLoss(x, label))
    #
    # loss = torch.nn.CrossEntropyLoss()
    # x_tensor = torch.from_numpy(x)
    # label_tensor = torch.from_numpy(label)
    # output = loss(x_tensor, label_tensor)
    # print("torch CrossEntropyLoss output: ", output)


# if __name__ == '__main__':
#     label = torch.randn(2, 256, 256)
#     nClasses = 2
#     imb_type = 'exp'
#     imb_factor = 0.01
#     total_num = label.shape[1]*label.shape[2]
#     img_num_per_cls = get_img_num_per_cls(nClasses, total_num, imb_type, imb_factor)
#     print('img_num_per_cls is ', img_num_per_cls)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     pred = torch.randn(2, 1, 256, 256).to(device)
#     label = torch.randn(2, 256, 256).to(device)
#     CB_loss(label, pred, img_num_per_cls)
    # FocalLoss(pred, label)