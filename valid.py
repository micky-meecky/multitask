import torch
import os
import glob

from tqdm import tqdm
from torch.utils.data import DataLoader
import xlwt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from dataset import DatasetImageMaskContourDist
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from utils import create_validation_arg_parser
from utilize.trainfunc import train_model, printProgressBar, char_color
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet, LossSoftDice
from utilize.solver import Solver
from utilize.evaluation import SegmentEvaluation

# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x

sep = os.sep
def build_model(model_type, num_classes):  # 构建模型

    # todo: add transformer into it.
    # 选择模型
    if model_type == "unet":
        # self.unet = UNet(input_channels=self.img_ch, num_classes=self.num_classes)
        unet = UNet(input_channels=1, num_classes=1, padding_mode='zeros', add_output=True, dropout=True)
    if model_type == "dcan":
        unet = UNet_DCAN(num_classes=num_classes)
    if model_type == "dmtn":
        unet = UNet_DMTN(num_classes=num_classes)
    if model_type == "psinet":
        unet = PsiNet(num_classes=num_classes)
    if model_type == "convmcd":
        unet = UNet_ConvMCD(num_classes=num_classes)

    return unet

def define_loss( model_type, loss_type, num_classes, device ,weights=[1, 1, 1]):
    if model_type == "unet":
        if loss_type == "Dice":
            criterion = LossSoftDice(weights)
        if loss_type == "BCE":
            criterion = LossUNet(0, num_classes=num_classes, weights=weights, device=device)
    if model_type == "dcan":
        criterion = LossDCAN(weights)
    if model_type == "dmtn":
        criterion = LossDMTN(weights)
    if model_type == "psinet" or model_type == "convmcd":
        # Both psinet and convmcd uses same mask,contour and distance loss function
        criterion = LossPsiNet(weights)

    return criterion


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    val_path_img = args.val_path + str(args.fold_id) + '/test/images/'
    val_path = os.path.join(val_path_img, "*.png")
    if args.use_best_model:
        model_file = args.model_file + args.project_name + '/models/best_model/' + 'best_unet_score.pt'
    else:
        model_file = args.model_file + args.project_name + '/models/' + args.pretrained_model_name

    save_path = args.save_path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = args.save_path + args.project_name + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model_type = args.model_type
    project_name = args.project_name
    num_classes = args.num_classes
    save_cls_data = args.save_cls_data
    plt_flag = args.plt_flag
    distance_type = args.distance_type
    normal_flag = args.normal_flag
    loss_type = args.loss_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, distance_type, normal_flag), num_workers=5,)

    fold_id = args.fold_id
    fold_flag = args.fold_flag
    fold_K = args.fold_K

    excel_path = args.excel_path + project_name + r'/' + project_name + r'.xls'

    model = build_model(model_type, num_classes)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    # 指标
    IOU_list = []
    DSC_list = []
    ACC_list = []
    SE_list = []
    PC_list = []
    SP_list = []
    F1_list = []
    JS_list = []
    Cls_list = []
    GT_cls_list = []
    ioumin3 = 0
    cls_1 = []
    cls_2 = []
    IOU_score = 0


    # 创建分割输出结果的excel表格
    if save_cls_data:
        wb = xlwt.Workbook(encoding='utf-8')
        ws = wb.add_sheet('test_sheet')
        ws.write(0, 0, 'id.png')
        ws.write(0, 1, 'cls_GT')
        ws.write(0, 2, 'ACC')
        ws.write(0, 3, 'IOU')
        ws.write(0, 4, 'DSC')
        ws.write(0, 5, 'SE')
        ws.write(0, 6, 'PC')
        ws.write(0, 7, 'SP')
        ws.write(0, 8, 'F1')
        ws.write(0, 9, 'JS')
        wb.save(excel_path)  # ====================================================
        # wb.save(r'./step3_save_pic/xplcs/xplcs.xls')

    ev = SegmentEvaluation(1)
    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(valLoader):
        mask_order = img_file_name[0].split('\\')[-1]
        with torch.no_grad():
            images_path = list(img_file_name)
            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)
            targets4 = targets4.to(device)
            targets = [targets1, targets2, targets3, targets4]

            SR = model(inputs)
            SR = SR[0]  # SR 是一个tensor
            SR1 = SR.data.cpu().numpy()
            targets1 = targets1.data.cpu().numpy()
            SR_tmp = SR1[0, :].reshape(-1)
            GT_tmp = targets1[0, :].reshape(-1)
            tmp_index = images_path[0].split('/')[-1].split('\\')[-1]
            tmp_index = int(tmp_index.split('.')[0][:])

            # 将其二值化
            SR_tmp = SR_tmp > 0.5

            SR_tmp = torch.from_numpy(SR_tmp).to(device)
            GT_tmp = torch.from_numpy(GT_tmp).to(device)

            if True:
                IOU_score = ev.get_IOU(SR_tmp, GT_tmp),
                IOU_score = IOU_score[0]
                if IOU_score < 0.3:
                    ioumin3 = ioumin3 + 1
                IOU_list.append(IOU_score)
                print('IOU:', IOU_score)
                IOU_final = np.mean(IOU_list)
                print('fold:', fold_id, '  IOU_final', IOU_final)
                DSC_score = ev.get_DC(SR_tmp, GT_tmp)
                # DSC_score = DSC_score[0]
                DSC_list.append(DSC_score)
                print('DSC:', DSC_score)
                DSC_final = np.mean(DSC_list)
                print('fold:', fold_id, '  DSC_final', DSC_final)

                ACC_score = ev.get_accuracy(SR_tmp, GT_tmp)
                # ACC_score = ACC_score[0]
                ACC_list.append(ACC_score)
                ACC_final = np.mean(ACC_list)

                SE_score = ev.get_sensitivity(SR_tmp, GT_tmp)
                # SE_score = SE_score[0]
                SE_list.append(SE_score)
                SE_final = np.mean(SE_list)

                PC_score = ev.get_precision(SR_tmp, GT_tmp)
                # PC_score = PC_score[0]
                PC_list.append(PC_score)
                PC_final = np.mean(PC_list)

                SP_score = ev.get_specificity(SR_tmp, GT_tmp)
                # SP_score = SP_score[0]
                SP_list.append(SP_score)
                SP_final = np.mean(SP_list)

                F1_score = ev.get_F1(SR_tmp, GT_tmp)
                # F1_score = F1_score[0]
                F1_list.append(F1_score)
                F1_final = np.mean(F1_list)

                JS_score = ev.get_JS(SR_tmp, GT_tmp)
                # JS_score = JS_score[0]
                JS_list.append(JS_score)
                JS_final = np.mean(JS_list)

            # 保存图像
            if save_path is not None:
                # 将tensor SR_tmp转化为numpy
                # SR_tmp = SR_tmp.data.cpu().numpy()
                # SR1 是一个numpy，形状是1x1x256x256的，我要转化成256x256的
                SR_tmp = SR1.reshape(256, 256)
                # 二值化
                SR_tmp[SR_tmp > 0.5] = 1
                SR_tmp[SR_tmp <= 0.5] = 0
                final_mask = SR_tmp * 255
                final_mask = final_mask.astype(np.uint8)
                print(np.unique(final_mask), '\n')  # 查看图像中的像素值,unique()函数去除重复值
                # print(np.max())
                images_path = save_path + 'images/'
                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                images_name = mask_order
                final_savepath = save_path + 'images/' + images_name
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            exl_idx = i + 1
            ws.write(exl_idx, 0, mask_order)
            ws.write(exl_idx, 1, "None")
            ws.write(exl_idx, 2, str(round(ACC_score, 4)))
            ws.write(exl_idx, 3, str(round(IOU_score, 4)))
            ws.write(exl_idx, 4, str(round(DSC_score, 4)))
            ws.write(exl_idx, 5, str(round(SE_score, 4)))
            ws.write(exl_idx, 6, str(round(PC_score, 4)))
            ws.write(exl_idx, 7, str(round(SP_score, 4)))
            ws.write(exl_idx, 8, str(round(F1_score, 4)))
            ws.write(exl_idx, 9, str(round(JS_score, 4)))
            wb.save(excel_path)  # ======================================================

        if plt_flag:
            plt.subplot(2, 2, 1)
            plt.title('img_' + mask_order, color='blue')
            plt.imshow(inputs, cmap=plt.cm.gray)


            # plt.show()
            plt.savefig(save_path + sep + mask_order, dpi=400)
        print('')

    exl_idx = exl_idx + 1
    ws.write(exl_idx, 0, "final_score")
    ws.write(exl_idx, 1, "None")
    ws.write(exl_idx, 2, str(round(ACC_final, 4)))
    ws.write(exl_idx, 3, str(round(IOU_final, 4)))
    ws.write(exl_idx, 4, str(round(DSC_final, 4)))
    ws.write(exl_idx, 5, str(round(SE_final, 4)))
    ws.write(exl_idx, 6, str(round(PC_final, 4)))
    ws.write(exl_idx, 7, str(round(SP_final, 4)))
    ws.write(exl_idx, 8, str(round(F1_final, 4)))
    ws.write(exl_idx, 9, str(round(JS_final, 4)))
    wb.save(excel_path)  # ================================================================
    print(ioumin3)

    print('Finished Testing')

