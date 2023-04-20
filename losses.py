import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class LossMulti:
    def __init__(
        self, jaccard_weight=0, class_weights=None, num_classes=1, device=None
    ):
        self.device = device
        if class_weights is not None:
            nll_weight = torch.from_numpy(class_weights.astype(np.float32)).to(
                self.device
            )
        else:
            nll_weight = None

        self.BCELoss = nn.BCELoss(weight=nll_weight)
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes

    def __call__(self, outputs, targets):

        targets = targets.float()
        # print("Min target value:", targets.min().item())
        # print("Max target value:", targets.max().item())
        # outputs = outputs.squeeze(1)
        loss = (1 - self.jaccard_weight) * self.BCELoss(outputs, targets)
        if self.jaccard_weight:
            eps = 1e-7
            for cls in range(self.num_classes):
                jaccard_target = (targets == cls).float()
                jaccard_output = outputs[:, cls].exp()
                intersection = (jaccard_output * jaccard_target).sum()

                union = jaccard_output.sum() + jaccard_target.sum()
                loss -= (
                    torch.log((intersection + eps) / (union - intersection + eps))
                    * self.jaccard_weight
                )

        return loss

class LossUNet:
    def __init__(self, jaccard_weight, num_classes=2, weights=[1, 1, 1], device='cpu'):

        self.LossMulti = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=2, device=device)

    def __call__(self, outputs, targets):

        loss = self.LossMulti(outputs, targets)

        return loss


class LossDCAN:
    def __init__(self, jaccard_weight, num_classes=2, weights=[1, 1, 1], device='cpu'):

        self.criterion1 = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=num_classes, device=device)
        self.criterion2 = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=num_classes, device=device)
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):

        criterion = self.weights[0] * self.criterion1(
            outputs1, targets1
        ) + self.weights[1] * self.criterion2(outputs2, targets2)

        return criterion


class LossDMTN:
    def __init__(self, jaccard_weight, num_classes=2, weights=[1, 1, 1], device='cpu'):

        self.criterion1 = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=num_classes, device=device)
        self.criterion2 = nn.MSELoss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, targets1, targets2):

        loss1 = self.criterion1(outputs1, targets1)
        loss2 = self.criterion2(outputs2, targets2)
        criterion = self.weights[0] * loss1 + self.weights[1] * loss2

        return criterion


class LossPsiNet:
    def __init__(self, jaccard_weight, num_classes=2, weights=[1, 1, 1], device='cpu'):

        self.criterion1 = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=num_classes, device=device)
        self.criterion2 = LossMulti(jaccard_weight=jaccard_weight, class_weights=None, num_classes=num_classes, device=device)
        self.criterion3 = nn.MSELoss()
        self.criterion4 = nn.CrossEntropyLoss()
        self.weights = weights

    def __call__(self, outputs1, outputs2, outputs3, outputs4, targets1, targets2, targets3, targets4):

        mask_loss = self.weights[0] * self.criterion1(outputs1, targets1)
        contour_loss = self.weights[1] * self.criterion2(outputs2, targets2)
        dist_loss = self.weights[2] * self.criterion3(outputs3, targets3)
        # targets4 = targets4.float()
        # targets4 和 outputs4 都要加1，为了防止出现0
        # targets4 = targets4 + 1.0
        # outputs4 = outputs4 + 1.0
        cls_loss = self.weights[3] * self.criterion4(outputs4, targets4)

        criterion = mask_loss + contour_loss + dist_loss + cls_loss

        return criterion, mask_loss, contour_loss, dist_loss, cls_loss

class LossSoftDice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LossSoftDice, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)   # batch_size
        smooth = 1

        # 针对每一个batch中的每一个样本，计算其dice loss
        loss = 0
        for i in range(num):
            m1 = probs[i]
            m2 = targets[i]
            # 先将m1, m2类型从tensor转换为numpy
            SR = m1.data.cpu().numpy()
            GT = m2.data.cpu().numpy()
            # 将m1, m2中的值转换为0或1
            SR = (SR > 0.5).astype(np.float)
            GT = (GT == np.max(GT)).astype(np.float)
            intersection = (m1 * m2)
            # 计算acc
            corr = np.sum(SR == GT)
            acc = float(corr) / float(SR.shape[0])
            if acc == 1:
                score = 1
            else:
                score = 2. * (intersection.sum() + smooth) / (m1.sum() + m2.sum() + smooth)
            loss += 1 - score
        loss = loss / num
        return loss