import numpy as np


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
        self.time_val = None
        self.time_sum = None
        self.time_avg = None
        # self.

    def initialize(self, val, time_val, weight):
        self.val = val
        self.time_val = time_val
        self.avg = val
        self.time_avg = time_val
        self.sum = val * weight
        self.time_sum = time_val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, time_val, weight=1):
        if not self.initialized:
            self.initialize(val, time_val, weight)
        else:
            self.add(val, time_val, weight)

    def add(self, val, time_val, weight):
        self.val = val
        self.time_val = time_val
        self.sum += val * weight
        self.time_sum += time_val * weight
        self.count += weight
        self.avg = self.sum / self.count
        self.time_avg = self.time_sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum, self.time_sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeterTimes(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeterTimes, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        # T = 5
        # print('pr.type is ', pr.type)
        # print('gt.v is ', gt.type)
        B, T, H, W = pr.shape
        gt_ = gt.reshape(B*H*W, T)#.cpu().numpy()
        pr_ = pr.reshape(B*H*W, T)#.cpu().numpy()
        mask_label = (gt_[:, 1:] != gt_[:, 0:T - 1])
        mask_pred = (pr_[:, 1:] != gt_[:, 0:T - 1])
        # print('mask_label size ', mask_label.shape)
        # print('mask_pred size ', mask_pred.shape)
        time_val = get_confuse_matrix(num_classes=2, label_gts=mask_label, label_preds=mask_pred)
        # print('time_val is ', time_val)
        self.update(val, time_val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum, self.time_sum)
        return scores_dict






def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean


def cm2F1(confusion_matrix):
    hist = confusion_matrix


    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix, time_confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]

    time_hist = time_confusion_matrix

    B = 8
    H, W = 256, 256
    # 时序部分
    TP = time_hist[1][1]
    # TN = time_hist[0][0]+B*H*W
    FP = time_hist[1][0]
    FN = time_hist[0][1]

    TPA = TP / (TP + FP)
    TUA = TP / (TP + FN)
    TF1 = 2 * TPA * TUA / (TPA + TUA)

    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1':mean_F1
                  ,'TPA': TPA, 'TUA': TUA, 'TF1': TF1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        # print('mask shape is ', mask.shape)
        # print('label_gts shape is', label_gts.shape)
        # print('label_preds shape is ', label_preds)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        # print('hist is ', hist)
        # print('hist shape is ', hist.shape)  # (num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):  # 按batch size遍历
        # print('lt shape is ', lt.shape)
        # print('lp shape is ', lp.shape)
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    mask_label = (label_gts[:, 1:] != label_gts[:, 0:T - 1])
    mask_pred = (label_preds[:, 1:] != label_gts[:, 0:T - 1])
    time_confusion_matrix = get_confuse_matrix(2, mask_label, mask_pred)
    score_dict = cm2score(confusion_matrix, time_confusion_matrix)
    return score_dict['miou']




# class ConfusionMatrix(object):
#     """
#     注意版本问题,使用numpy来进行数值计算的
#     """
#
#     def __init__(self, num_classes: int, labels=None):# list):
#         self.matrix = np.zeros((num_classes, num_classes))
#         self.num_classes = num_classes
#         # self.labels = labels
#
#     def update(self, preds, labels):
#         for p, t in zip(preds, labels):
#             self.matrix[t, p] += 1
#
#         return self.matrix



if __name__ == '__main__':
    import torch

    from sklearn.metrics import mean_absolute_error

    B, T, H, W = 4, 5, 2, 2
    N = 5
    pred = torch.arange(0, 4*5*H*W*5, dtype=torch.int8).reshape(B, T, N, H, W).reshape(B*H*W, T, N)
    # pred = torch.range(0, 2*5*2*2*5-1, dtype=torch.int8).reshape(B*H*W, T, N)
    pred_ = pred.argmax(dim=2).detach().cpu().numpy()
    gt = torch.arange(0, 4*5*H*W, dtype=torch.int8).reshape(B*H*W, T).detach().cpu().numpy()
    # gt = torch.arange(0, 2*5*2*2, dtype=torch.int8).reshape(B, T, H, W).detach().cpu().numpy()
    # print(mean_absolute_error(pred, gt))
    # metric = ConfuseMatrixMeter(n_class=5)
    # output1 = mse(gt, pred)
    # output = metric.update_cm(pred, gt)
    # print(output)
    # print(output1)
    #
    # # gt  # B T H W  num field (0 ~ class-1)
    # # pred = pred.argmax(dim=2)
    mask_label = (gt[:, 1:] != gt[:, 0:T - 1])

    mask_pred = (pred_[:, 1:] != pred_[:, 0:T - 1])


    # matrix = np.zeros((2, 2))
    # for p, t in zip(mask_pred, mask_label):
    #     matrix[t, p] += 1

    confusion_matrix = get_confuse_matrix(num_classes=2, label_gts=mask_label, label_preds=mask_pred)
    print('confusion_matrix is', confusion_matrix)
    print('confusion_matrix 0 0 is', confusion_matrix[0][0]+B*H*W)  # TN
    print('confusion_matrix 0 1 is', confusion_matrix[0][1])  # FN
    print('confusion_matrix 1 0 is', confusion_matrix[1][0])  # FP
    print('confusion_matrix 1 1 is', confusion_matrix[1][1])  # TP   # batch_size由于头尾误差，即头尾各一个时间点无法直接判断

    # mask_tp = torch.logical_and(mask_pred, mask_label)
    # print(mask_tp.shape)
    #
    # for ls in mask_tp:
    #     print(ls.shape)
    # print(mask_tp)

    # 错误的写法
    # mask_tp = (mask_label and mask_pred)
    # mask_fp = ((not mask_label) and (mask_pred))
    # mask_fn = ((mask_label) and (not mask_pred))
    # mask_tn = ((not mask_label) and (not mask_pred))
