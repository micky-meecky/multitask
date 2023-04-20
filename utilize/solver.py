import shutil

import torch
import os
import time
import datetime
import warnings


import numpy as np
import pandas as pd
import openpyxl
import matplotlib
import torchvision
from torch import optim
from torch import nn
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from skorch import NeuralNetRegressor
from skorch.helper import SliceDataset
from sklearn.model_selection import RandomizedSearchCV
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from utilize.lrwarmup import GradualWarmupScheduler
from utilize.trainfunc import train_model, printProgressBar, char_color
from losses import LossUNet, LossDCAN, LossDMTN, LossPsiNet, LossSoftDice
from utilize.evaluation import SegmentEvaluation


os.environ['CUDA_LAUNCH_BLOCKING'] = '0'    # 用于调试cuda程序，当程序出现问题时，可以打印出出错的行号, 实现CPU与GPU同步或者异步

matplotlib.use('Agg')  # 控制绘图不显示,以免在linux下程序崩溃
class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):
        # prohject name
        self.project_name = config.project_name

        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.num_classes = config.num_classes
        # self.augmentation_prob = config.augmentation_prob
        self.sizeFlag = True
        self.pretrained_model_name = config.pretrained_model_name
        self.model_type = config.model_type

        # Hyper-parameters
        self.lr = config.lr
        self.lr_low = config.lr_low
        if self.lr_low is None:
            self.lr_low = self.lr / 1e+6
            print("auto set minimun lr :", self.lr_low)

        # optimizer param
        self.beta1 = config.beta1  # for adam
        self.beta2 = config.beta2  # for adam
        self.criterion = None

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size
        self.use_pretrained_model = config.use_pretrained
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.DataParallel = config.DataParallel
        self.loss_type = config.loss_type
        self.use_best_model = config.use_best_model # 是否使用最好的模型

        # Step size
        self.save_model_step = config.save_model_step
        self.val_step = config.val_step
        self.decay_step = config.decay_step

        # Path
        self.train_path = config.train_path + str(config.fold_id) + '/train/images/'  # train_path是指训练集图片路径
        self.val_path = config.train_path + str(config.fold_id) + '/validation/images/'  # val_path是指验证集图片路径
        self.test_path = config.train_path + str(config.fold_id) + '/test/images/'  # test_path是指测试集图片路径
        self.result_path = os.path.join(config.save_path + '/' + config.project_name)  # result_path是指保存模型的路径
        self.mode = config.mode
        self.save_image = config.save_image
        self.save_detail_result = config.save_detail_result
        self.save_step_model = config.save_path + '/' + config.project_name
        self.log_dir = self.result_path + '/logs'
        self.DataParallel = config.DataParallel
        self.save_path = config.save_path

        self.test_flag = config.test_flag
        self.fold_id = config.fold_id

        # 设置学习率策略相关超参数
        self.decay_ratio = config.decay_ratio
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr

        # 设置是否使用超参数搜索
        self.is_use_hyper_search = config.is_use_hyper_search
        self.param_distributions = {
            'lr': [0.001, 0.01],
            'max_epochs': [10, 5],
            'batch_size': [8, 16, 32],
            # Add more hyperparameters here
        }
        self.num_samples = config.num_samples
        self.cv = config.cv
        self.verbose = config.verbose

        self.tta_mode = config.tta_mode

        # 执行个初始化函数
        self.my_init()

        # Make record file
        # self.record_file = self.result_path + '/record.txt'
        # f = open(self.record_file, 'w')
        # f.close()

        # 模型参数总数
        self.sizetotal = 0

        self.device_count = 1
        self.device_count = torch.cuda.device_count()

        if self.device_count > 1:
            print("Let's use", self.device_count, "GPUs!")
        else:
            print("Let's use", self.device_count, "GPU!")

        f = open(os.path.join(self.result_path, 'config.txt'), 'w')
        for key in config.__dict__:
            print('%s: %s' % (key, config.__getattribute__(key)), file=f)
        f.close()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        f = open(self.record_file, 'a')
        print(*args, file=f)
        f.close()

    def my_init(self):  # 初始化函数
        self.result_path = os.path.join(self.save_path + '/' + self.project_name)
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        # Make record file
        self.record_file = self.result_path + '/record.txt'
        f = open(self.record_file, 'w')
        f.close()
        self.myprint("这是第%d个fold的训练" % (self.fold_id))
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        # self.print_date_msg()
        self.build_model()


    def getModelSize(self, model):
        param_size = 0
        param_sum = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            param_sum += param.nelement()
        buffer_size = 0
        buffer_sum = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            buffer_sum += buffer.nelement()
        all_size = (param_size + buffer_size) / 1024 / 1024
        self.myprint('模型总大小为：{:.3f}MB'.format(all_size))
        return param_size, param_sum, buffer_size, buffer_sum, all_size
        # param_size:参数大小，param_sum:参数个数，buffer_size:缓冲区大小，buffer_sum:缓冲区个数，all_size:总大小
        # 什么叫缓冲区？   缓冲区是用来存储一些中间变量的，比如batchnorm的running_mean和running_var，
        # 这些变量在训练过程中是会变化的，所以需要存储起来，所以就有了缓冲区。
        # 什么叫缓冲区个数？ 缓冲区个数就是缓冲区的个数，比如batchnorm的running_mean和running_var，就是两个缓冲区。
        # 那么这些缓冲区的数据可以放到GPU上吗？ 可以的！

    def build_model(self):  # 构建模型

        # todo: add transformer into it.
        # 选择模型
        if self.model_type == "unet":
            # self.unet = UNet(input_channels=self.img_ch, num_classes=self.num_classes)
            self.unet = UNet(input_channels=self.img_ch, num_classes=self.num_classes, padding_mode='reflect', add_output=True, dropout=True)
            # padding_mode有四种模式，分别是zeros,reflect,replicate,circular
        if self.model_type == "dcan":
            self.unet = UNet_DCAN(input_channels=self.img_ch, num_classes=self.num_classes, padding_mode='reflect', add_output=True, dropout=True)
        if self.model_type == "dmtn":
            self.unet = UNet_DMTN(input_channels=self.img_ch, num_classes=self.num_classes, padding_mode='reflect', add_output=True, dropout=True)
        if self.model_type == "psinet":
            self.unet = PsiNet(num_classes=self.num_classes)
        if self.model_type == "convmcd":
            self.unet = UNet_ConvMCD(input_channels=self.img_ch, num_classes=self.num_classes, padding_mode='reflect', add_output=True, dropout=True)

        # 模型并行化
        if self.DataParallel:
            self.unet = nn.DataParallel(self.unet)

        # 打印getModelSize函数返回的网络
        self.myprint(self.getModelSize(self.unet))

        if self.sizeFlag:   # 这里是打印网络的参数量
            self.sizeFlag = False
            self.sizetotal = sum([param.nelement() for param in self.unet.parameters()])
            self.myprint("Number of parameter: %.2fM" % (self.sizetotal / 1e6))
        ######

        # 优化器修改
        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2])  # 优化器

        # lr schachle策略(要传入optimizer才可以)
        # 暂时有三种情况,(1)只用cosine decay,(2)只用warmup,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:  # zhishiyong
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            print('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            print('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                           self.lr_cos_epoch,
                                                           eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            print('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                print('use linear decay')

        self.unet.to(self.device)  # 将模型放到GPU上

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        self.myprint(model)
        self.myprint(name)
        self.myprint("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu() # .cpu()是将tensor转换为cpu的tensor, 这两个区别在于, .cuda()是将tensor转换为gpu的tensor, .cpu()是将tensor转换为cpu的tensor
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def tensor2img(self, x):
        """Convert tensor to img (numpy)."""
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def define_loss(self, model_type, loss_type, weights=[0.5, 0.1, 0.1, 0.3]):
        if model_type == "unet":
            if loss_type == "Dice":
                criterion = LossSoftDice(weights)
            if loss_type == "BCE":
                criterion = LossUNet(0, num_classes=self.num_classes, weights=weights, device=self.device)
        if model_type == "dcan":
            criterion = LossDCAN(0, num_classes=self.num_classes, weights=weights, device=self.device)
        if model_type == "dmtn":
            criterion = LossDMTN(0, num_classes=self.num_classes, weights=weights, device=self.device)
        if model_type == "psinet" or model_type == "convmcd":
            # Both psinet and convmcd uses same mask,contour and distance loss function
            criterion = LossPsiNet(0, num_classes=self.num_classes, weights=weights, device=self.device)

        return criterion

    def train(self):
        # 局部变量
        models_path = self.result_path + '/models/'  # models_path是指模型的路径
        images_path = self.result_path + '/images/'  # images_path是指图片的路径
        best_model_path = models_path + 'best_model/'  # best_model_path是指最好的模型的路径
        # 如果best_model_path路径不存在,则创建
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path)
        # 如果模型路径不存在,则创建
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        # 如果图片路径不存在,则创建
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        # 如果self.log_dir路径不存在,则创建
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        pretrained_model_path = models_path + self.pretrained_model_name  # pretrained_model_path是指预训练模型的路径
        best_unet_score = 0.
        Iter = 0
        train_len = len(self.train_loader)
        total_time = 0


        writer = SummaryWriter(log_dir=self.log_dir)  # tensorboard
        valid_record = np.zeros((1, 9))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU, cls_acc]
        # test_record = np.zeros((1, 8))  # [epoch, Iter, acc, SE, SP, PC, Dice, IOU]
        valid_record = np.array(['epoch', 'Iter', 'acc', 'SE', 'SP', 'PC', 'Dice', 'IOU', 'cls_acc'])
        # valid_record = np.array(['epoch', 'Iter', 'acc', 'SE', 'SP', 'PC', 'Dice', 'IOU'])
        """Train encoder, generator and discriminator."""
        self.myprint('-----------------------%s-----------------------------' % self.project_name)

        # 断点继续训练,看看是否有上一训练时候保存的最优模型
        epoch_start = "0"  # epoch_start是指从第几个epoch开始训练
        if self.use_best_model:
            best_model_path = models_path + 'best_model/'
            # 如果best_model_path路径存在，则获取该目录下的文件夹名
            best_unet_score_path = None
            best_model_list = None
            if os.path.exists(best_model_path):
                best_model_list = os.listdir(best_model_path)
                # 获取最新的文件夹名， best_model_list[-1]是指最新的文件夹名
                best_model_list.sort(key=lambda fn: os.path.getmtime(best_model_path + fn))
                # 然后获取该文件夹下best_unet_score.pt
                best_unet_score_path = best_model_path + '/best_unet_score.pt'
                # best_model_list[-1]是指最新的文件夹名
            if os.path.exists(best_unet_score_path):
                self.unet.load_state_dict(torch.load(best_unet_score_path))  # 加载预训练模型
                self.myprint(epoch_start)  # 打印epoch_start
                self.myprint("Loading Model {}".format(best_model_list[-1]))  # 打印最新的文件夹名


        if self.use_pretrained_model:
            print("Loading Model {}".format(os.path.basename(pretrained_model_path)))  # basename返回文件名，不包含路径
            self.unet.load_state_dict(torch.load(pretrained_model_path))  # 加载预训练模型
            # 返回预训练模型文件的文件名（不包含路径和文件后缀），然后使用 split(".")[0] 来去掉文件后缀部分，只留下文件名。
            epoch_start = os.path.basename(pretrained_model_path).split(".")[0]
            print(epoch_start)  # 打印epoch_start

        self.criterion = self.define_loss(self.model_type, self.loss_type)

        self.myprint('Training...')
        for epoch in range(self.num_epochs):
            tic = datetime.datetime.now()
            epochtic = datetime.datetime.now()

            self.unet.train(True)   # 设置为训练模式

            epoch_loss = 0  # 指一个epoch里的loss
            length = 0  # 指一个epoch里包含的样本数量

            self.myprint('enumerating')
            with warnings.catch_warnings():
                for i, sample in enumerate(self.train_loader):

                    # print('enumerate finished')
                    current_lr = self.optimizer.param_groups[0]['lr']  # 获取当前lr
                    print(current_lr)
                    (img_file_name, inputs, targets1, targets2, targets3, targets4) = sample
                    # 将targets2转成numpyarray
                    targets2_tmp = targets2.cpu().detach().numpy()
                    inputs = inputs.to(self.device)
                    targets1 = targets1.to(self.device)
                    targets2 = targets2.to(self.device)
                    targets3 = targets3.to(self.device)
                    targets4 = targets4.to(self.device)
                    # 查看inputs, targets1, targets2, targets3, targets4的shape
                    # print('inputs shape: ', inputs.shape)
                    # print('targets1 shape: ', targets1.shape)


                    targets = [targets1, targets2, targets3, targets4]

                    loss, output, mask_loss, contour_loss, dist_loss, cls_loss = train_model(self.unet, inputs, targets, self.model_type, self.criterion, self.optimizer)

                    length += 1
                    Iter += 1
                    writer.add_scalars('Loss', {'loss': loss}, Iter)  # 往文件里写进度

                    # Out = output[:, 0:1, :, :]
                    # if (self.save_image) and (i % 20 == 0):
                    #     images_all = torch.cat((inputs, Out, targets1), 0)
                    #     torchvision.utils.save_image(images_all.data.cpu(),
                    #                                  os.path.join(self.result_path, 'images', 'Train_%d_image.png' % i),
                    #                                  nrow=self.batch_size)  # 生成雪碧图

                    print_content = 'batch_total_loss:' + str(loss.data.cpu().numpy())
                    print_content += ' mask_loss:' + str(mask_loss.data.cpu().numpy())
                    print_content += ' contour_loss:' + str(contour_loss.data.cpu().numpy())
                    print_content += ' dist_loss:' + str(dist_loss.data.cpu().numpy())
                    print_content += ' cls_loss:' + str(cls_loss.data.cpu().numpy())
                    printProgressBar(i + 1, train_len, content=print_content)

                    epoch_loss += loss.item()
                    print('wait for enumerate')

            # 计时结束
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
            print(char_color(time_str))

            tic = datetime.datetime.now()

            epoch_loss = epoch_loss / length
            self.myprint('Epoch [%d/%d], Loss: %.10f' % (epoch + 1 + int(epoch_start), self.num_epochs + int(epoch_start), epoch_loss))

            # 记录下lr到log里(并且记录到图片里)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(current_lr)
            self.lr_list.append(current_lr)
            writer.add_scalars('Learning_rate', {'lr': current_lr}, epoch + int(epoch_start))
            # 保存lr为png
            figg = plt.figure()
            plt.plot(self.lr_list)
            figg.savefig(os.path.join(self.result_path, 'lr.png'))
            plt.close()

            figg, axis = plt.subplots()
            plt.plot(self.lr_list)
            axis.set_yscale("log")
            figg.savefig(os.path.join(self.result_path, 'lr_log.png'))
            plt.close()

            # 学习率策略部分 =========================
            # lr scha way 1:
            if self.lr_sch is not None:
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()

            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):  # 根据设置衰减速率来更新lr
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        # self.lr /= 100.0
                        self.update_lr(self.lr)
                        self.myprint('Decay Learning_rate to lr: {}.'.format(self.lr))

            if (epoch + 1) % self.val_step == 0:
                # Validation #
                if self.tta_mode:  # (默认为False)
                    # acc, SE, SP, PC, DC, IOU, cls_acc = self.test_tta(mode='valid')
                    acc, SE, SP, PC, DC, IOU, cls_acc = self.test_tta(mode='valid')
                else:
                    acc, SE, SP, PC, DC, IOU, cls_acc = self.test_tta(mode='valid')
                # valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, PC, DC, IOU, cls_acc])))
                valid_record = np.vstack((valid_record, np.array([epoch + 1 + int(epoch_start), Iter, acc, SE, SP, PC, DC, IOU, cls_acc])))
                # unet_score = 1.0 * cls_acc  # TODO
                unet_score = 1.0 * IOU  # TODO
                # writer.add_scalars('Valid', {'Dice': DC, 'IOU': IOU, 'acc': acc, 'SE': SE, 'SP': SP, 'PC': PC, 'Cls_acc': cls_acc}, epoch)
                writer.add_scalars('Valid', {'Dice': DC, 'IOU': IOU, 'acc': acc, 'SE': SE, 'SP': SP, 'PC': PC, 'Cls_acc': cls_acc},
                                   epoch + int(epoch_start))
                # self.myprint(
                #     '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f, cls_acc: %.4f' % (
                #         acc, SE, SP, PC, DC, IOU, cls_acc))
                self.myprint(
                    '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, Dice: %.4f, IOU: %.4f, cls_acc: %.4f' % (
                        acc, SE, SP, PC, DC, IOU, cls_acc))

                # 保存最好的模型
                if unet_score > best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()

                    self.myprint('Best model in epoch %d, score : %.4f' % (best_epoch + 1 + int(epoch_start), best_unet_score))

                    # 删除现有子文件夹
                    for folder in os.listdir(best_model_path):
                        folder_path = os.path.join(best_model_path, folder)
                        if os.path.isdir(folder_path):
                            shutil.rmtree(folder_path)

                    # 创建一个名为当前 epoch 数的新子文件夹
                    new_folder_path = os.path.join(best_model_path, str(epoch))
                    os.makedirs(new_folder_path)

                    torch.save(best_unet, os.path.join(best_model_path, "best_unet_score.pt"))

                    #  Test (默认Fals), 因为不用训练中test, 所以这里注释掉

                # save_record_in_xlsx
                if (True):
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    record = pd.ExcelWriter(excel_save_path)
                    detail_result1 = pd.DataFrame(valid_record)
                    detail_result1.to_excel(record, 'valid', float_format='%.5f')  # 三个参数分别代表：文件名，表名，保留小数位数
                    if self.test_flag:
                        detail_result2 = pd.DataFrame(test_record)
                        detail_result2.to_excel(record, 'test', float_format='%.5f')
                    # record.save()
                    record.close()
            # Save model
            if epoch % 5 == 0:
                torch.save(self.unet.state_dict(), os.path.join(models_path, str(epoch + int(epoch_start)) + ".pt"))

            # 计算每个epoch的时间
            toc = datetime.datetime.now()
            h, remainder = divmod((toc - tic).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
            self.myprint(char_color(time_str))

            # 计算剩余时间
            epochtoc = datetime.datetime.now()
            time_seconds = (epochtoc - epochtic).seconds
            total_time = total_time + time_seconds
            per_epoch_time = total_time / (epoch + 1)
            h, remainder = divmod(time_seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "per whole epoch cost Time %02d h:%02d m:%02d s" % (h, m, s)
            self.myprint(char_color(time_str))
            remain_time_sec = (self.num_epochs - epoch - 1) * per_epoch_time
            h, remainder = divmod(remain_time_sec, 3600)
            m, s = divmod(remainder, 60)
            time_str = "perhaps need Time %02d h:%02d m:%02d s" % (h, m, s)
            self.myprint(char_color(time_str))

        # 计算训练总时间
        h, remainder = divmod(total_time, 3600)
        m, s = divmod(remainder, 60)
        time_str = "total use time for %02d h:%02d m:%02d s" % (h, m, s)
        self.myprint(char_color(time_str))

        # self.myprint('Finished!')
        self.myprint("第%d个fold的训练结束了！！！！！！！", self.fold_id)
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

    def test_tta(self, mode='train', unet_path=None):
        """Test model & Calculate performances."""
        ev = SegmentEvaluation(1)
        print(char_color('@,,@   testing with TTA'))
        if not unet_path is None:
            if os.path.isfile(unet_path):
                self.unet.load_state_dict(torch.load(unet_path))
                self.myprint('Successfully Loaded from %s' % (unet_path))

        self.unet.train(False)
        self.unet.eval()

        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'test':
            data_loader = self.test_loader
        elif mode == 'valid':
            data_loader = self.valid_loader
        else:
            data_loader = self.valid_loader

        acc = 0.  # Accuracy
        SE = 0.  # Sensitivity (Recall)
        SP = 0.  # Specificity
        PC = 0.  # Precision
        DC = 0.  # Dice Coefficient
        IOU = 0.  # IOU
        Cls_acc = 0.  # Class_acc
        length = 0

        conacc = 0.
        conSE = 0.
        conSP = 0.
        conPC = 0.
        conDC = 0.
        conIOU = 0.
        conlength = 0
        condetail_result = []

        distacc = 0.
        distSE = 0.
        distSP = 0.
        distPC = 0.
        distDC = 0.
        distIOU = 0.
        distlength = 0
        distdetail_result = []

        valid_record = np.array(['id', 'acc', 'SE', 'SP', 'PC', 'Dice', 'IOU', 'Cls_acc'])
        con_valid_record = np.array(['id', 'acc', 'SE', 'SP', 'PC', 'Dice', 'IOU'])
        condetail_result.append(con_valid_record)
        distdetail_result.append(con_valid_record)
        detail_result = []
        detail_result.append(valid_record)


        with torch.no_grad():
            for i, sample in enumerate(data_loader):
                (img_file_name, inputs, targets1, targets2, targets3, targets4) = sample
                images_path = list(img_file_name)
                inputs = inputs.to(self.device)
                targets1 = targets1.to(self.device)
                targets2 = targets2.to(self.device)
                targets3 = targets3.to(self.device)
                targets4 = targets4.to(self.device)

                # targets = [targets1, targets2, targets3, targets4]

                SR = self.unet(inputs)
                SR_contour = SR[1]              # SR_contour 是一个tensor
                SR_contour1 = SR_contour.data.cpu().numpy()

                SR_dist = SR[2]                 # SR_dist 是一个tensor
                SR_dist1 = SR_dist.data.cpu().numpy()

                SR_cls = torch.argmax(SR[3], dim=1)                  # SR_cls 是一个tensor
                SR_cls1 = SR_cls.data.cpu().numpy()

                SR = SR[0]                   # SR 是一个tensor
                SR1 = SR.data.cpu().numpy()  # SR1 是一个numpy数组

                if self.save_image:
                    # 判断SR_cls类别的ACC
                    SR_cls = SR_cls.data.cpu().numpy()
                    targets4 = targets4.data.cpu().numpy()

                    corr = np.sum(SR_cls == targets4)
                    cls_acc = float(corr) / float(SR_cls.shape[0])


                    if SR.shape[1] == 1:
                        # 把SR做二值化，SR是一个tensor
                        threshold = 0.5  # 二值化阈值
                        device = SR.device  # 获取 SR 张量所在的设备
                        SR_binary = torch.where(SR > threshold, torch.tensor(1.0, device=device),
                                                torch.tensor(0.0, device=device))
                        images_all = torch.cat((inputs, SR_binary, targets1), 0)
                    else:
                        SRc = SR[:, 0:1, :, :]
                        images_all = torch.cat((inputs, SRc, targets1), 0)
                    torchvision.utils.save_image(images_all.data.cpu(), os.path.join(self.result_path, 'images',
                                                                                     '%s_%d_image.png' % (mode, i)),
                                                 nrow=self.batch_size)
                    if SR_contour.shape[1] == 1:
                        threshold = 0.5
                        device = SR_contour.device
                        SR_contour_binary = torch.where(SR_contour > threshold, torch.tensor(1.0, device=device),
                                                        torch.tensor(0.0, device=device))
                        images_all = torch.cat((inputs, SR_contour_binary, targets2), 0)
                    else:
                        SRc = SR_contour[:, 0:1, :, :]
                        images_all = torch.cat((inputs, SRc, targets2), 0)
                    torchvision.utils.save_image(images_all.data.cpu(), os.path.join(self.result_path, 'images',
                                                                                        '%s_%d_image_contour.png' % (mode, i)),
                                                    nrow=self.batch_size)

                    if SR_dist.shape[1] == 1:
                        threshold = 0.5
                        device = SR_dist.device
                        SR_dist_binary = torch.where(SR_dist > threshold, torch.tensor(1.0, device=device),
                                                        torch.tensor(0.0, device=device))
                        images_all = torch.cat((inputs, SR_dist_binary, targets3), 0)
                    else:
                        SRc = SR_dist[:, 0:1, :, :]
                        images_all = torch.cat((inputs, SRc, targets3), 0)
                    torchvision.utils.save_image(images_all.data.cpu(), os.path.join(self.result_path, 'images',
                                                                                        '%s_%d_image_contour.png' % (mode, i)),
                                                    nrow=self.batch_size)


                # SR1 = SR1.data.cpu().numpy()
                targets1 = targets1.data.cpu().numpy()
                # SR_contour1 = SR_contour1.data.cpu().numpy()
                targets2 = targets2.data.cpu().numpy()
                # SR_dist1 = SR_dist1.data.cpu().numpy()
                targets3 = targets3.data.cpu().numpy()


                for i in range(SR.shape[0]):
                    SR_tmp = SR1[i, :].reshape(-1)
                    GT_tmp = targets1[i, :].reshape(-1)
                    tmp_index = images_path[i].split('/')[-1].split('\\')[-1]
                    tmp_index = int(tmp_index.split('.')[0][:])

                    SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                    GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                    result_tmp = np.array([tmp_index,
                                           ev.get_accuracy(SR_tmp, GT_tmp),
                                           ev.get_sensitivity(SR_tmp, GT_tmp),
                                           ev.get_specificity(SR_tmp, GT_tmp),
                                           ev.get_precision(SR_tmp, GT_tmp),
                                           ev.get_DC(SR_tmp, GT_tmp),
                                           ev.get_IOU(SR_tmp, GT_tmp),
                                           float(cls_acc)])
                                           # ev.get_clsaccuracy(lable_idx, class_GT)])

                    acc += result_tmp[1]
                    SE += result_tmp[2]
                    SP += result_tmp[3]
                    PC += result_tmp[4]
                    DC += result_tmp[5]
                    IOU += result_tmp[6]
                    Cls_acc += result_tmp[7]
                    detail_result.append(result_tmp)

                    length += 1

                # for contour
                for i in range(SR_contour.shape[0]):
                    SR_tmp = SR_contour1[i, :].reshape(-1)
                    GT_tmp = targets2[i, :].reshape(-1)
                    tmp_index = images_path[i].split('/')[-1].split('\\')[-1]
                    tmp_index = int(tmp_index.split('.')[0][:])

                    SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                    GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                    result_tmp = np.array([tmp_index,
                                           ev.get_accuracy(SR_tmp, GT_tmp),
                                           ev.get_sensitivity(SR_tmp, GT_tmp),
                                           ev.get_specificity(SR_tmp, GT_tmp),
                                           ev.get_precision(SR_tmp, GT_tmp),
                                           ev.get_DC(SR_tmp, GT_tmp),
                                           ev.get_IOU(SR_tmp, GT_tmp)])
                                           # ev.get_clsaccuracy(lable_idx, class_GT)])

                    conacc += result_tmp[1]
                    conSE += result_tmp[2]
                    conSP += result_tmp[3]
                    conPC += result_tmp[4]
                    conDC += result_tmp[5]
                    conIOU += result_tmp[6]
                    # Cls_acc += result_tmp[7]
                    condetail_result.append(result_tmp)

                    length += 1

                # for dist
                for i in range(SR_dist.shape[0]):
                    SR_tmp = SR_dist1[i, :].reshape(-1)
                    GT_tmp = targets3[i, :].reshape(-1)
                    tmp_index = images_path[i].split('/')[-1].split('\\')[-1]
                    tmp_index = int(tmp_index.split('.')[0][:])

                    SR_tmp = torch.from_numpy(SR_tmp).to(self.device)
                    GT_tmp = torch.from_numpy(GT_tmp).to(self.device)

                    result_tmp = np.array([tmp_index,
                                           ev.get_accuracy(SR_tmp, GT_tmp),
                                           ev.get_sensitivity(SR_tmp, GT_tmp),
                                           ev.get_specificity(SR_tmp, GT_tmp),
                                           ev.get_precision(SR_tmp, GT_tmp),
                                           ev.get_DC(SR_tmp, GT_tmp),
                                           ev.get_IOU(SR_tmp, GT_tmp)])
                                           # ev.get_clsaccuracy(lable_idx, class_GT)])

                    distacc += result_tmp[1]
                    distSE += result_tmp[2]
                    distSP += result_tmp[3]
                    distPC += result_tmp[4]
                    distDC += result_tmp[5]
                    distIOU += result_tmp[6]
                    # Cls_acc += result_tmp[7]
                    condetail_result.append(result_tmp)

                    length += 1

        accuracy = acc / length
        sensitivity = SE / length
        specificity = SP / length
        precision = PC / length
        disc = DC / length
        iou = IOU / length
        # cls_acc = Cls_acc / length
        detail_result = np.array(detail_result)

        conaccuracy = conacc / length
        consensitivity = conSE / length
        conspecificity = conSP / length
        conprecision = conPC / length
        condisc = conDC / length
        coniou = conIOU / length
        # cls_acc = Cls_acc / length
        condetail_result = np.array(condetail_result)

        distaccuracy = distacc / length
        distsensitivity = distSE / length
        distspecificity = distSP / length
        distprecision = distPC / length
        distdisc = distDC / length
        distiou = distIOU / length
        # cls_acc = Cls_acc / length
        distdetail_result = np.array(distdetail_result)

        # print('conaccuracy = ', conaccuracy)
        # print('consensitivity = ', consensitivity)
        # print('conspecificity = ', conspecificity)
        # print('conprecision = ', conprecision)
        # print('condisc = ', condisc)
        # print('coniou = ', coniou)

        # 将condetail_result的结果保存到condetail_result.xlsx中
        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            txt_save_path = os.path.join(self.result_path, mode + '_pre_detail_contour_result.xlsx')
            writer = pd.ExcelWriter(txt_save_path)
            condetail_result = pd.DataFrame(condetail_result)
            condetail_result.to_excel(writer, mode, float_format='%.5f', index=False, header=False)
            # writer.save()
            writer.close()


        # 定义一个变量，用于设置求平均的个数
        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            excel_save_path = os.path.join(self.result_path, mode + '_pre_detail_result.xlsx')
            writer = pd.ExcelWriter(excel_save_path)
            detail_result = pd.DataFrame(detail_result)
            detail_result.to_excel(writer, mode, float_format='%.5f', index=False, header=False)
            # writer.save()
            writer.close()

        # 将distdetail_result的结果保存到distdetail_result.xlsx中
        if (self.save_detail_result):  # detail_result = [id, acc, SE, SP, PC, dsc, IOU]
            txt_save_path = os.path.join(self.result_path, mode + '_pre_detail_dist_result.xlsx')
            writer = pd.ExcelWriter(txt_save_path)
            distdetail_result = pd.DataFrame(distdetail_result)
            distdetail_result.to_excel(writer, mode, float_format='%.5f', index=False, header=False)
            # writer.save()
            writer.close()


        return accuracy, sensitivity, specificity, precision, disc, iou, cls_acc

        # return accuracy, sensitivity, specificity, precision, disc, iou

        # return accuracy, sensitivity, specificity, precision, disc, iou, cls_acc

    def hyper_search(self, devLoader, testLoader):
        print('hyper_search')
        train_data = next(iter(self.train_loader))
        X_train, y_train = train_data[1], train_data[2]
        X_dev_data = next(iter(devLoader))
        X_dev, y_dev = X_dev_data[1], X_dev_data[2]

        net = UNet(input_channels=self.img_ch, num_classes=self.num_classes, padding_mode='reflect', add_output=True,
             dropout=True)
        criterion = LossUNet(0, num_classes=self.num_classes)
        net = NeuralNetRegressor(
            module=net,
            max_epochs=10,
            batch_size=10,
            optimizer=torch.optim.Adam,
            criterion=criterion,
            lr=0.001,
            device=self.device,
            train_split=SliceDataset,
            iterator_train__shuffle=True,
            iterator_train__num_workers=0,
            iterator_valid__shuffle=False,
            iterator_valid__num_workers=0,
            verbose=2,
        )
        # 设置joblib启动方式为forkserver
        # net.initialize()
        # net.initialize_optimizer()
        # net.initialize_criterion()
        # net.initialize_callbacks()
        # net._initialize_module()

        search = RandomizedSearchCV(
            net,
            self.param_distributions,
            n_iter=self.num_samples,
            scoring='r2',
            cv=self.cv,
            refit=True,
            verbose=self.verbose,
            n_jobs=4,

        )

        search.fit(X_train, y_train)

        best_params = search.best_params_
        print(f"Best hyperparameters: {best_params}")









