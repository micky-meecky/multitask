import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import glob
from torch.optim import Adam
from tqdm import tqdm
import logging
from torch import nn
import numpy as np
import h5py
import torchvision
import random
from tensorboardX import SummaryWriter

from utils import visualize, evaluate, create_train_arg_parser
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from dataset import DatasetImageMaskContourDist


def define_loss(loss_type, weights=[1, 1, 1]):

    if loss_type == "unet":
        criterion = LossUNet(weights)
    if loss_type == "dcan":
        criterion = LossDCAN(weights)
    if loss_type == "dmtn":
        criterion = LossDMTN(weights)
    if loss_type == "psinet" or loss_type == "convmcd":
        # Both psinet and convmcd uses same mask,contour and distance loss function
        criterion = LossPsiNet(weights)

    return criterion


def build_model(model_type):

    if model_type == "unet":
        model = UNet(num_classes=2)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
    if model_type == "psinet":
        model = PsiNet(num_classes=2)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=2)

    return model


def train_model(model, targets, model_type, criterion, optimizer):

    if model_type == "unet":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs[0], targets[0])
            loss.backward()
            optimizer.step()

    if model_type == "dcan":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs[0], outputs[1], targets[0], targets[1])
            loss.backward()
            optimizer.step()

    if model_type == "dmtn":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs[0], outputs[1], targets[0], targets[2])
            loss.backward()
            optimizer.step()

    if model_type == "psinet" or model_type == "convmcd":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(
                outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[2]
            )
            loss.backward()
            optimizer.step()

    return loss


if __name__ == "__main__":

    args = create_train_arg_parser().parse_args()

    CUDA_SELECT = "cuda:{}".format(args.cuda_no) # format是指定格式化字符串，{}是占位符，.format()是格式化函数
    log_path = args.save_path + "/summary"
    writer = SummaryWriter(log_dir=log_path)  # tensorboardX，用于可视化

    # logging.basicConfig()函数对日志的输出格式及方式做相关配置
    logging.basicConfig(
        filename="".format(args.object_type),   # 日志文件名
        filemode="a",                           # 写入模式，a表示追加
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",  # 日志格式
        datefmt="%Y-%m-%d %H:%M",            # 时间格式
        level=logging.INFO,                 # 日志级别
    )
    logging.info("")    # 日志记录

    # glob是python的一个模块，用于查找符合特定规则的文件路径名，glob.glob()返回所有匹配的文件路径列表
    # os.path.join(args.train_path, "*.jpg")是指在args.train_path路径下查找所有.jpg文件, 然后将其路径存入train_file_names
    train_path = args.train_path + str(args.fold_id) + '/train/images/'
    val_path = args.train_path + str(args.fold_id) + '/validation/images/'
    test_path = args.train_path + str(args.fold_id) + '/test/images/'
    train_file_names = glob.glob(train_path + "*.png")
    random.shuffle(train_file_names)    # 打乱训练集图片路径
    val_file_names = glob.glob(os.path.join(args.val_path, "*.png"))    # 获取验证集图片路径
    test_file_names = glob.glob(os.path.join(args.test_path, "*.png"))  # 获取测试集图片路径


    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")  # 要么使用GPU，要么使用CPU
    model = build_model(args.model_type)    # 构建模型

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model = model.to(device)

    # To handle epoch start number and pretrained weight
    epoch_start = "0"   # epoch_start是指从第几个epoch开始训练
    if args.use_pretrained:
        print("Loading Model {}".format(os.path.basename(args.pretrained_model_path))) # os.path.basename()返回文件名，不包含路径，包含文件后缀
        model.load_state_dict(torch.load(args.pretrained_model_path))
        epoch_start = os.path.basename(args.pretrained_model_path).split(".")[0]
        print(epoch_start)

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_file_names, args.distance_type),
        batch_size=args.batch_size,
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type)
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type),
        batch_size=args.val_batch_size,
    )

    optimizer = Adam(model.parameters(), lr=1e-4)
    criterion = define_loss(args.model_type)

    for epoch in tqdm(
        range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs)
    ):

        global_step = epoch * len(trainLoader)
        running_loss = 0.0

        for i, (img_file_name, inputs, targets1, targets2, targets3) in enumerate(
            tqdm(trainLoader)
        ):

            model.train()

            inputs = inputs.to(device)
            targets1 = targets1.to(device)
            targets2 = targets2.to(device)
            targets3 = targets3.to(device)

            targets = [targets1, targets2, targets3]

            loss = train_model(model, targets, args.model_type, criterion, optimizer)

            writer.add_scalar("loss", loss.item(), epoch)

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_file_names)

        if epoch % 1 == 0:

            dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer)
            writer.add_scalar("loss_valid", dev_loss, epoch)
            visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
            print("Global Loss:{} Val Loss:{}".format(epoch_loss, dev_loss))
        else:
            print("Global Loss:{} ".format(epoch_loss))

        logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), os.path.join(args.save_path, str(epoch) + ".pt")
            )
