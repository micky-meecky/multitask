import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SegmentEvaluation():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # self.reset()

    def get_clsaccuracy(self, SR, GT):
        acc = 0
        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()
        SR = SR.astype(float)
        GT = GT.astype(float)

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])

        return acc

    def get_accuracy(self, SR, GT, threshold=0.5):
        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == (np.max(GT))).astype(float)

        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        corr = np.sum(SR == GT)

        acc = float(corr) / float(SR.shape[0])

        return acc

    def get_sensitivity(self, SR, GT, threshold=0.5):
        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == np.max(GT)).astype(float)

        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        # TP : True Positive
        # FN : False Negative
        TP = (((SR == 1.).astype(float) + (GT == 1.).astype(float)) == 2.).astype(float)
        FN = (((SR == 0.).astype(float) + (GT == 1.).astype(float)) == 2.).astype(float)

        corr = np.sum(SR == GT)

        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            SE = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而SE表示的是正样本被正确分类的概率，所以当acc=1时，SE=1
        else:
            SE = float(np.sum(TP)) / (float(np.sum(TP + FN)) + 1e-6)  # 1e-6是为了防止分母为0
        return SE

    def get_specificity(self, SR, GT, threshold=0.5):
        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == np.max(GT)).astype(float)

        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        # TN : True Negative
        # FP : False Positive
        TN = (((SR == 0.).astype(float) + (GT == 0.).astype(float)) == 2.).astype(float)
        FP = (((SR == 1.).astype(float) + (GT == 0.).astype(float)) == 2.).astype(float)

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            SP = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而SP表示的是负样本被正确分类的概率，所以当acc=1时，SP=1
        else:
            SP = float(np.sum(TN)) / (float(np.sum(TN + FP)) + 1e-6)

        return SP

    def get_precision(self, SR, GT, threshold=0.5):

        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == (np.max(GT))).astype(float)

        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        # TP : True Positive
        # FP : False Positive
        # TP = ((SR == 1) + (GT == 1)) == 2
        # FP = ((SR == 1) + (GT == 0)) == 2

        TP = (((SR == 1.).astype(float) + (GT == 1.).astype(float)) == 2.).astype(float)
        FP = (((SR == 1.).astype(float) + (GT == 0.).astype(float)) == 2.).astype(float)

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            PC = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而PC表示的是预测为正的样本中，真正为正的概率，所以当acc=1时，PC=1
        else:
            PC = float(np.sum(TP)) / (float(np.sum(TP + FP)) + 1e-6)
        return PC


    def get_F1(self, SR, GT, threshold=0.5):
        # Sensitivity == Recall F度量
        SE = self.get_sensitivity(SR, GT, threshold=threshold)
        PC = self.get_precision(SR, GT, threshold=threshold)
        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()
        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            F1 = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而F1表示的是精确度（Precision）和召回率（recall）
            # 是你高我低的关系，所以当acc=1时，F1=1
        else:
            F1 = 2 * SE * PC / (SE + PC + 1e-6)

        return F1

    # ----------------by hand-----------------

    def get_JS(self, SR, GT, threshold=0.5):
        # JS : Jaccard similarity 越大越好

        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == (np.max(GT))).astype(float)
        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        Inter = np.sum((SR + GT) == 2)
        Union = np.sum((SR + GT) >= 1)

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            JS = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而JS表示的是交集和并集的比值，所以当acc=1时，JS=1
        else:
            JS = float(Inter) / (float(Union) + 1e-6)

        return JS


    def get_DC(self, SR, GT, threshold=0.5):
        # DC : Dice Coefficient
        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == np.max(GT)).astype(float)
        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        Inter = np.sum(((SR + GT) == 2).astype(float))

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            DC = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而DC表示的是相似度，所以当acc=1时，DC=1
        else:
            DC = float(2 * Inter) / (float(np.sum(SR) + np.sum(GT)) + 1e-6)
        return DC


    def get_IOU(self, SR, GT, threshold=0.5):

        SR = SR.view(-1)
        GT = GT.view(-1)

        SR = SR.data.cpu().numpy()
        GT = GT.data.cpu().numpy()

        SR = (SR > threshold).astype(float)
        # GT = (GT == np.max(GT)).astype(float)

        if np.any(GT > 0):
            GT = (GT == np.max(GT)).astype(float)
        else:
            GT = np.zeros_like(GT, dtype=float)

        # TP : True Positive
        # FP : False Positive
        # FN : False Negative
        TP = (((SR == 1.).astype(float) + (GT == 1.).astype(float)) == 2.).astype(float)
        FP = (((SR == 1.).astype(float) + (GT == 0.).astype(float)) == 2.).astype(float)
        FN = (((SR == 0.).astype(float) + (GT == 1.).astype(float)) == 2.).astype(float)

        corr = np.sum(SR == GT)
        acc = float(corr) / float(SR.shape[0])
        if acc == 1:
            IOU = 1  # 因为当acc=1时，说明无论是正样本还是负样本，都被正确分类了，而IOU表示的是交集和并集的比值，所以当acc=1时，IOU=1
        else:
            IOU = float(np.sum(TP)) / (float(np.sum(TP + FP + FN)) + 1e-6)

        return IOU


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)  # batch_size
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
            SR = (SR > 0.5).astype(float)
            GT = (GT == np.max(GT)).astype(float)
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

