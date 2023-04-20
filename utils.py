import torch
import os
from tqdm import tqdm
from torch import nn
import numpy as np
import torchvision
from torch.nn import functional as F
import time
import argparse


def evaluate(device, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()  # start time
    with torch.no_grad():   # no need to track gradients

        for iter, data in enumerate(tqdm(data_loader)):

            _, inputs, targets, _, _, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = F.nll_loss(outputs[0], targets.squeeze(1))
            losses.append(loss.item())

        writer.add_scalar("Dev_Loss", np.mean(losses), epoch)

    return np.mean(losses), time.perf_counter() - start


def visualize(device, epoch, model, data_loader, writer, val_batch_size, train=False):
    def save_image(image, tag, val_batch_size):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(
            image, nrow=int(np.sqrt(val_batch_size)), pad_value=0, padding=25
        )
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            _, inputs, targets, _, _, _ = data

            inputs = inputs.to(device)

            targets = targets.to(device)
            outputs = model(inputs)

            output_mask = outputs[0].detach().cpu().numpy()
            output_final = np.argmax(output_mask, axis=1).astype(float)

            output_final = torch.from_numpy(output_final).unsqueeze(1)

            if train == "True":
                save_image(targets.float(), "Target_train", val_batch_size)
                save_image(output_final, "Prediction_train", val_batch_size)
            else:
                save_image(targets.float(), "Target", val_batch_size)
                save_image(output_final, "Prediction", val_batch_size)

            break


def create_train_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--normal_flag", type=bool, default=False, help="normalization flag")
    parser.add_argument("--train_path", type=str, default='./train_path/fold/fold', help="path to img png files")
    parser.add_argument("--val_path", type=str, default='./train_path/fold/fold', help="path to img png files")
    parser.add_argument("--test_path", type=str, default='./train_path/fold/fold', help="path to img png files")

    parser.add_argument('--image_size', type=int, default=256)  # 网络输入img的size, 即输入会被强制resize到这个大小
    parser.add_argument('--img_ch', type=int, default=1)    # 输入img的通道数
    parser.add_argument('--num_classes', type=int, default=1)  # 网络输出的通道数, 一般为1
    parser.add_argument('--output_ch', type=int, default=1)  # 网络输出的通道数, 一般为1

    parser.add_argument('--mode', type=str, default='train', help='train/test')  # 训练or测试
    parser.add_argument("--model_type", type=str, default="convmcd", help="model type: unet,dcan,dmtn,psinet,convmcd")
    parser.add_argument("--object_type", type=str, default='dataset', help="Dataset.")
    parser.add_argument("--distance_type", type=str, default="dist_mask", help="distance transform type - dist_mask,dist_contour,dist_signed")

    parser.add_argument("--batch_size", type=int, default=80, help="train batch size")
    parser.add_argument("--val_batch_size", type=int, default=160, help="validation batch size")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument('--DataParallel', type=bool, default=True)  # 是否使用多gpu训练

    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_low', type=float, default=1e-13)# 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
    parser.add_argument('--lr_warm_epoch', type=int, default=20)  # warmup的epoch数,一般就是5~20,为0或False则不使用
    parser.add_argument('--lr_cos_epoch', type=int, default=600)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用
    parser.add_argument("--lr_use_decay", type=bool, default=False, help="use lr decay")  # 是否使用lr衰减
    parser.add_argument('--num_epochs_decay', type=int, default=20)  # decay开始的最小epoch数
    parser.add_argument('--decay_ratio', type=float, default=0.01)  # 0~1,每次decay到1*ratio
    parser.add_argument('--decay_step', type=int, default=40)  # epoch
    parser.add_argument('--loss_type', type=str, default='BCE', help='loss type: BCE, Dice')  # loss类型

    # optimizer param
    parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

    parser.add_argument("--use_pretrained", type=bool, default=False, help="Load pretrained checkpoint.")
    parser.add_argument("--pretrained_model_name", type=str, default='485.pt', help="If use_pretrained is true, provide checkpoint.")
    parser.add_argument("--use_best_model", type=bool, default=False, help="Load best checkpoint.")

    # result&save
    parser.add_argument('--test_flag', type=bool, default=False)  # 训练过程中是否测试,不测试会节省很多时间
    parser.add_argument("--save_path", type=str, default='./savemodel', help="Model save path.")
    parser.add_argument('--log_path', type=str, default='./result/TNSCUI/mylogs')   # 日志保存路径
    parser.add_argument('--save_detail_result', type=bool, default=True)    # 是否保存详细的结果
    parser.add_argument('--save_image', type=bool, default=True)  # 训练过程中观察图像和结果
    parser.add_argument('--save_model_step', type=int, default=20)  # 保存模型时的间隔步数
    parser.add_argument('--plt_flag', type=bool, default=False)  # 是否在训练过程中的validation使用plt

    parser.add_argument('--val_step', type=int, default=1)  # 进行测试集或验证集评估的间隔步数
    parser.add_argument('--tta_mode', type=bool, default=False)  # 是否在训练过程中的validation使用tta

    parser.add_argument("--fold_all_train", type=bool, default=True, help="fold all train")
    parser.add_argument("--fold_id", type=int, default=1, help="fold id")
    parser.add_argument("--fold_num", type=int, default=5, help="fold num")
    parser.add_argument("--auto_select_fold", type=bool, default=False, help="auto select fold")

    parser.add_argument("--project_name", type=str, default='unetconvmcd_01_f1', help="project name")

    parser.add_argument('--exp_prefix', type=str, default='unetconvmcd_01_f')  # 实验名前缀，就是project_name的前面部分
    parser.add_argument('--record_n_rows', type=int, default=20)  # 取记录的倒数20行，做平均分数

    parser.add_argument('--is_use_hyper_search', type=bool, default=False)  # 是否使用超参搜索
    parser.add_argument('--num_samples', type=int, default=10)  # 超参搜索的样本数
    parser.add_argument('--cv', type=int, default=5)  # 超参搜索的交叉验证数
    parser.add_argument('--verbose', type=int, default=2)  # 超参搜索是否显示详细信息

    return parser


def create_validation_arg_parser():

    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--model_type", type=str, default="unet", help="select model type: unet,dcan,dmtn,psinet,convmcd")
    parser.add_argument("--normal_flag", type=bool, default=False, help="normalization flag")
    parser.add_argument("--project_name", type=str, default='unet_04_f2', help="project name")
    parser.add_argument("--pretrained_model_name", type=str, default='440.pt',
                        help="If use_pretrained is true, provide checkpoint.")
    parser.add_argument('--loss_type', type=str, default='BCE', help='loss type: BCE, Dice')  # loss类型

    # path
    parser.add_argument("--val_path", type=str, default='./train_path/fold/fold', help="path to img jpg files")
    parser.add_argument("--distance_type", type=str, default="dist_mask",
                        help="distance transform type - dist_mask,dist_contour,dist_signed")
    parser.add_argument("--model_file", type=str, default='./savemodel/', help="model_file")
    parser.add_argument("--save_path", type=str, default='./valid_result/', help="results save path.")
    # parser.add_argument("--excel_path", type=str, default='./savemodel/', help="excel path")
    parser.add_argument("--save_step3_path", type=str, default='./valid_result', help="results save path.")
    parser.add_argument('--plt_flag', type=bool, default=False)  # 是否在训练过程中的validation使用plt
    parser.add_argument('--save_cls_data', type=bool, default=True)  # 是否保存分类数据


    parser.add_argument("--cuda_no", type=int, default=0, help="cuda number")
    parser.add_argument('--DataParallel', type=bool, default=True)  # 是否使用多gpu训练

    # fold相关
    parser.add_argument("--fold_K", type=int, default=5, help="fold num")
    parser.add_argument("--fold_id", type=int, default=2, help="fold id")
    parser.add_argument("--fold_flag", type=bool, default=True, help="fold flag") # 预测测试集则设置为False直接读取img_path中PNG文件进行测试,True则使用分折信息
    parser.add_argument("--excel_path", type=str, default='./valid_result/', help="excel path")

    # 加载模型相关
    parser.add_argument("--use_best_model", type=bool, default=True, help="Load best checkpoint.")
    parser.add_argument('--num_classes', type=int, default=1)  # 网络输出的通道数, 一般为1






    return parser
