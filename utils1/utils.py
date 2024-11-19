import math

import torch
import numpy as np
from typing import Optional
from torch import nn, Tensor
import torch.nn.functional as F


def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1)
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum - intersection

    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1)  # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)

    smooth = .001
    dice = 2 * (intersection + smooth) / (mask_sum + smooth)
    return dice

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)      # 满足条件的掩膜
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))  #混淆矩阵初始化
    hist_1= np.zeros((num_class, num_class))
    hist_2 = np.zeros((2, 2))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))        # set是一个集合，其中不包含重复元素，并且是无序的
        assert unique_set.issubset((set([0,1,2,3,4,5])))    # 判断unique_set是否是set的一个子集
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape
        hist += get_hist(infer_array, label_array, num_class)

        pred_building_mask = np.zeros_like(pred)
        pred_building_mask[pred != 0] = 1
        pred_mask1 = np.array(pred * pred_building_mask)
        label_mask1 = np.array(label * pred_building_mask)
        hist_1 += get_hist(pred_mask1, label_mask1, num_class)          # 预测为建筑物的混淆矩阵多列别

        pred_building_mask1 = np.zeros_like(pred)
        label_building_mask1 = np.zeros_like(label)
        pred_building_mask1[pred != 0] = 1
        label_building_mask1[label != 0] = 1
        hist_2 += get_hist(pred_building_mask1, label_building_mask1, 2)    # 建筑物与分建筑物混淆矩阵2分类

    # 得到多分类的混淆矩阵
    gt_sum = np.sum(hist, axis=1)
    pd_sum = np.sum(hist, axis=0)
    TP = np.diag(hist)
    iou = TP / (gt_sum + pd_sum - TP + np.finfo(np.float32).eps)
    miou = np.nanmean(iou)
    precision = TP / (pd_sum + np.finfo(np.float32).eps)
    recall = TP / (gt_sum + np.finfo(np.float32).eps)
    F1 = 2 * precision * recall / (recall + precision + np.finfo(np.float32).eps)
    m_F1 = np.nanmean(F1)

    # 计算FBS
    hist_1[:,0]=0        # 免去其中预测为非建筑物的值
    TPm = np.diag(hist_1)
    p1 = np.sum(TPm)/(np.sum(hist_1) + np.finfo(np.float32).eps)
    p2 = hist_2[1,1]/(np.sum(hist_2[:,1] + np.finfo(np.float32).eps))           # hist_2[1,1]/(np.sum(hist_2[1,:] + np.finfo(np.float32).eps))
    FBS = (p1 * math.exp(p2)) / math.e

    return m_F1 ,miou, iou, F1, FBS


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))

def weighted_BCE_logits(logit_pixel, truth_pixel, weight_pos=0.65, weight_neg=0.35):
    logit = logit_pixel.view(-1)  # 这里输出的通道数为1，即直接表示出变与不变而不是计算概率，因为这里是二分类。
    truth = truth_pixel.view(-1)  # 这里的打开成一个向量的目的是为了添加权重，避免样本不均匀性
    assert (logit.shape == truth.shape)

    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')  # 计算二元损失交叉熵

    pos = (truth > 0.5).float()  # 取标签中变化的像素数量
    neg = (truth < 0.5).float()  # 取标签中未发生变化的像素数量
    pos_num = pos.sum().item() + 1e-12
    neg_num = neg.sum().item() + 1e-12
    loss = (weight_pos * pos * loss / pos_num + weight_neg * neg * loss / neg_num).sum()  # 计算每个pos和每个nag的平均损失，然后加权后并相加
    return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        target = target.squeeze(2)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def label_smoothed_nll_loss(
    lprobs: torch.Tensor, target: torch.Tensor, epsilon: float, ignore_index=None, reduction="mean", dim=-1
) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param lprobs: Log-probabilities of predictions (e.g after log_softmax)
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -lprobs.gather(dim=dim, index=target)
        smooth_loss = -lprobs.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = epsilon / lprobs.size(dim)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss

class SoftCrossEntropyLoss(nn.Module):
    """
    Drop-in replacement for nn.CrossEntropyLoss with few additions:
    - Support of label smoothing
    """

    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(self, reduction: str = "mean", smooth_factor: float = 0.05, ignore_index: Optional[int] = -100, dim=1):
        super().__init__()
        self.smooth_factor = smooth_factor
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        log_prob = F.log_softmax(input, dim=self.dim)
        return label_smoothed_nll_loss(
            log_prob,
            target,
            epsilon=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )