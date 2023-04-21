import pathlib
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import torch
import torch.nn.functional as F



def train_model(model, inputs, targets, model_type, is_use_dist, criterion, optimizer):

    if model_type == "unet":

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            # loss = criterion(outputs[:, 0:1, :, :], targets[0])
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
            if not is_use_dist:
                # outputs[3]是一个10x3的tensor，代表的是10个样本，每个样本对应三个类别的概率，现在需要对每一个样本得到概率最大的那个类别
                outputs[2] = torch.argmax(outputs[2], dim=1)
                    # .float()
                loss, mask_loss, contour_loss, cls_loss = criterion(
                    outputs[0], outputs[1], outputs[2], targets[0], targets[1], targets[3]
                )
            else:
                # outputs[3]是一个10x3的tensor，代表的是10个样本，每个样本对应三个类别的概率，现在需要对每一个样本得到概率最大的那个类别
                outputs[3] = torch.argmax(outputs[3], dim=1)
                # .float()
                loss, mask_loss, contour_loss, dist_loss, cls_loss = criterion(
                    outputs[0], outputs[1], outputs[2], outputs[3], targets[0], targets[1], targets[2], targets[3]
                )
            loss.backward()
            optimizer.step()

    if is_use_dist:
        return loss, outputs, mask_loss, contour_loss, dist_loss, cls_loss
    else:
        return loss, outputs, mask_loss, contour_loss, cls_loss

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█',content =None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    if content:
        print('\r%s |%s| %s%% %s %s' % (prefix, bar, percent, suffix, content), end = ' ')
    else:
        print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = ' ')

    # Print New Line on Complete
    if iteration == total:
        print()

def char_color(s,front=50,word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;"+str(int(word))+";"+str(int(front))+"m"+s+"\033[0m"
    return new_char