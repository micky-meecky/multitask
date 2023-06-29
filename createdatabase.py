import os
import shutil
from time import sleep

import cv2
import numpy as np
import argparse
import datetime
import csv
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from collections import Counter

sep = os.sep # os.sep根据你所处的平台，自动采用相应的分隔符号

class ContourDistance():
    def __init__(self, config):
        self.logname = config.project_name + '.log'
        self.train_path = config.train_path
        self.labels = {}  # lables是一个字典，key是图片的序号，value是图片的类别, 序号用[]表示，类别用()表示
        self.csv_file = config.csv_file
        self.image_path = config.image_path
        self.mask_path = config.mask_path
        self.save_contour_path = config.save_contour_path
        self.logpath = config.log_path
        self.save_distance_path = config.save_distance_path
        self.contour_flag = config.contour_flag
        self.distance_flag = config.distance_flag
        self.save_distance_path_D1 = config.save_distance_path_D1
        self.save_distance_path_D2 = config.save_distance_path_D2
        self.save_distance_path_D3 = config.save_distance_path_D3

        self.shownum = 10

        self.benign = 0
        self.malignant = 0
        self.normal = 0

    def check_save_contour_path(self):
        if not os.path.exists(self.save_contour_path):
            os.makedirs(self.save_contour_path)

    def check_save_distance_paths(self):
        if not os.path.exists(self.save_distance_path_D1):
            os.makedirs(self.save_distance_path_D1)

        if not os.path.exists(self.save_distance_path_D2):
            os.makedirs(self.save_distance_path_D2)

        if not os.path.exists(self.save_distance_path_D3):
            os.makedirs(self.save_distance_path_D3)

    # 写入log文件
    def write_log(self, log):
        if not os.path.exists(self.logname):
            with open(self.logname, 'w') as f:
                f.write(log + '\n')
                f.write(str(config) + '\n\n')
                f.write(str(datetime.datetime.now()) + '\n\n')
                f.write('*********************************\n\n')
                f.close()
        else:
            with open(self.logname, 'a') as f:
                f.write(log + '\n')
                f.write(str(config) + '\n\n')
                f.write(str(datetime.datetime.now()) + '\n\n')
                f.write('*********************************\n\n')
                f.close()

    def GenerateDistanceMap(self, img_file, img_id, img, mask):
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        if np.sum(mask) == 0 or self.labels[img_id] == 2:
            print("为正常图像，没有结节")
            distance_map_D1 = np.zeros_like(img)
            distance_map = distance_map_D1
        else:
            # 计算距离图 D1
            distance_map_D1 = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            distance_map = cv2.normalize(distance_map_D1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 保存距离图 D1
        distance_map_D1_path = os.path.join(self.save_distance_path_D1, f"{img_id}.png")
        cv2.imwrite(distance_map_D1_path, distance_map)

        # 生成轮廓图像
        contour_mask = mask.copy()
        contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            distance_map_D2 = np.zeros_like(img)
            distance_map = distance_map_D2
        else:
            # 创建一个新图像
            contour_img = np.zeros_like(img)

            # 绘制轮廓
            cv2.drawContours(contour_img, contours, -1, 255, 1)

            # 计算距离图 D2
            # q: 最后一个参数是什么意思？
            # a: 3表示3*3的邻域，5表示5*5的邻域，以此类推
            # DIST_L2: 欧式距离
            distance_map_D2 = cv2.distanceTransform(contour_img, cv2.DIST_L2, 3)
            # 归一化处理
            distance_map = cv2.normalize(distance_map_D2, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # 保存距离图 D2
        distance_map_D2_path = os.path.join(self.save_distance_path_D2, f"{img_id}.png")
        cv2.imwrite(distance_map_D2_path, distance_map)

        # 生成距离图 D3
        if len(contours) == 0:
            distance_map_D3 = np.zeros_like(img)
            distance_map = distance_map_D3
        else:
            # 创建一个新图像
            contour_img = np.zeros_like(img)

            # 绘制轮廓
            cv2.drawContours(contour_img, contours, -1, 255, 1)

            # 计算正距离（轮廓内的距离）
            # pos_distance_map = cv2.distanceTransform(contour_img, cv2.DIST_L2, 3)
            raw_signed_distance = cv2.distanceTransform(contour_img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            raw_signed_distance = raw_signed_distance - cv2.distanceTransform(255 - contour_img, cv2.DIST_L2,
                                                                              cv2.DIST_MASK_PRECISE)

            min_val, max_val, _, _ = cv2.minMaxLoc(raw_signed_distance)
            distance_map = cv2.normalize(raw_signed_distance, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                                         dtype=cv2.CV_8U)

        # 保存距离图 D3
        distance_map_D3_path = os.path.join(self.save_distance_path_D3, f"{img_id}.png")
        cv2.imwrite(distance_map_D3_path, distance_map)

        print(f"Processed image {img_file} successfully.")
        if self.labels[img_id] == 0:
            return 1, 0, 0
        elif self.labels[img_id] == 1:
            return 0, 1, 0
        else:
            return 0, 0, 1

    def SaveContour(self, img_file, img_id, img, mask):
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # 这里是二值化，阈值为127
        # 检测读取进来的mask是否全为零以及对应序号的csv文件中那个label是否为零
        if np.sum(mask) == 0 or self.labels[img_id] == 2:
            print("为正常图像，没有结节")
            # 则contour为mask掩膜图像即可
            contour_img = mask
            cv2.imwrite(self.save_contour_path + img_file, contour_img)
        else:
            # 生成轮廓图像
            contour_mask = mask.copy()
            contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 创建一个新图像
            contour_img = np.zeros_like(img)

            # 绘制所有轮廓
            # 2 表示轮廓的粗细
            cv2.drawContours(contour_img, contours, -1, 255, 1)
            # cv2.drawContours(contour_img, contours, -1, 255, 1)

            # 保存轮廓图像
            # contour_path = os.path.join(self.save_contour_path, img_file)
            cv2.imwrite(self.save_contour_path + img_file, contour_img)

        # 输出提示信息
        print(f"Processed image {img_file} successfully.")
        # 统计处理的图片中三个类别的数量
        if self.labels[img_id] == 0:
            return 1, 0, 0
        elif self.labels[img_id] == 1:
            return 0, 1, 0
        else:
            return 0, 0, 1

def CreateContourDistance(config):
    contour = ContourDistance(config)

    # 读取CSV文件，获取每张图片的序号和类别
    with open(contour.csv_file, "r") as f:
        for line in f:
            # 去掉第一行，因为第一行是表头
            if line.startswith("ID"):
                continue
            parts = line.strip().split(",")
            # 图片ID为001.jpg格式，所以去掉后面的.jpg
            parts_id = parts[0].split(".")[0]  # 001
            parts_jpg = parts[0].split(".")[1]  # jpg

            img_id = int(parts_id)
            label = int(parts[1])
            contour.labels[img_id] = label

    # 初始化三个类别的数量
    b = 0
    m = 0
    n = 0

    # 检测目标文件夹是否存在
    contour.check_save_contour_path()
    contour.check_save_distance_paths()

    # 循环处理每张图像
    for img_file in os.listdir(contour.image_path):
        # 如果是图像文件
        if img_file.endswith(".png"):
            # 读取原图和掩码
            img_id = int(img_file.split(".")[0])  # 获取图片序号
            img_path = os.path.join(contour.image_path, img_file)  # 图片路径
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            mask_file = os.path.join(contour.mask_path, img_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

            if contour.contour_flag:
                # 保存轮廓图像
                new_b, new_m, new_n = contour.SaveContour(img_file, img_id, img, mask)
                b += new_b
                m += new_m
                n += new_n
            if contour.distance_flag:
                # 保存距离图
                new_b, new_m, new_n = contour.GenerateDistanceMap(img_file, img_id, img, mask)
                b += new_b
                m += new_m
                n += new_n

    # 将b,m,n写入log文件
    log = f"Processed {len(os.listdir(contour.image_path))} images " + config.project_name + f"successfully.\n"
    log += f"Benign: {b}, Malignant: {m}, Normal: {n}\n"
    # 再加上总共处理的图片数量
    log += f"Total: {b + m + n}\n"
    contour.write_log(log)
    print("All images processed successfully.")
    # 保存log文件
    log = f"Processed {len(os.listdir(contour.image_path))} images successfully."

def DatasetDivided(config):
    # Todo: 读取csv文件，将图片分为训练集、验证集和测试集

    pass

def readCsv(csvfname):
    # read csv to list of lists
    print(csvfname)
    with open(csvfname, 'r',) as csvf:
        reader = csv.reader(csvf)
        csvlines = list(reader)
    return csvlines

def get_fold_filelist(csv_file, K=3, fold=1, random_state=2020, validation=False, validation_r = 0.2):
    """
    获取分折结果的API（基于size分3层的类别平衡分折）
    :param csv_file: 带有ID、CATE、size的文件
    :param K: 分折折数
    :param fold: 返回第几折,从1开始
    :param random_state: 随机数种子, 固定后每次实验分折相同(注意,sklearn切换版本可能会导致相同随机数种子产生不同分折结果)
    :param validation: 是否需要验证集（从训练集随机抽取部分数据当作验证集）
    :param validation_r: 抽取出验证集占训练集的比例
    :return: train和test的list，带有label和size
    """
    CTcsvlines = readCsv(csv_file)
    header = CTcsvlines[0]
    print('header', header)
    nodules = CTcsvlines[1:]

    # 提取size的三分点
    sizeall = [int(i[2]) for i in nodules]
    sizeall.sort()  # 按升序排列
    low_mid_thre = sizeall[int(len(sizeall)*1/3)]
    mid_high_thre = sizeall[int(len(sizeall)*2/3)]

    # 根据size三分位数分为low，mid，high三组

    low_size_list = [i for i in nodules if int(i[2]) < low_mid_thre]
    mid_size_list = [i for i in nodules if int(i[2]) < mid_high_thre and int(i[2]) >= low_mid_thre]
    high_size_list = [i for i in nodules if int(i[2]) >= mid_high_thre]

    # 将lable划分为三组
    low_label = [int(i[1]) for i in low_size_list]
    mid_label = [int(i[1]) for i in mid_size_list]
    high_label = [int(i[1]) for i in high_size_list]


    low_fold_train = []
    low_fold_test = []

    mid_fold_train = []
    mid_fold_test = []

    high_fold_train = []
    high_fold_test = []

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(low_label, low_label):
        low_fold_train.append([low_size_list[i] for i in train])
        low_fold_test.append([low_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(mid_label, mid_label):
        mid_fold_train.append([mid_size_list[i] for i in train])
        mid_fold_test.append([mid_size_list[i] for i in test])

    sfolder = StratifiedKFold(n_splits=K, random_state=random_state, shuffle=True)
    for train, test in sfolder.split(high_label, high_label):
        high_fold_train.append([high_size_list[i] for i in train])
        high_fold_test.append([high_size_list[i] for i in test])

    if validation is False: # 不设置验证集，则直接返回
        train_set = low_fold_train[fold-1]+mid_fold_train[fold-1]+high_fold_train[fold-1]
        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]
        return [train_set, test_set]
    else:  # 设置验证集合，则从训练集“类别 且 size平衡地”抽取一定数量样本做验证集
        # 分离第fold折各size分层的正负以及正常组织样本，其label分别为1，0，2
        low_fold_train_p = [i for i in low_fold_train[fold-1] if int(i[1]) == 1]
        low_fold_train_n = [i for i in low_fold_train[fold-1] if int(i[1]) == 0]
        low_fold_train_o = [i for i in low_fold_train[fold-1] if int(i[1]) == 2]    # o for other


        mid_fold_train_p = [i for i in mid_fold_train[fold-1] if int(i[1]) == 1]
        mid_fold_train_n = [i for i in mid_fold_train[fold-1] if int(i[1]) == 0]
        mid_fold_train_o = [i for i in mid_fold_train[fold-1] if int(i[1]) == 2]

        high_fold_train_p = [i for i in high_fold_train[fold-1] if int(i[1]) == 1]
        high_fold_train_n = [i for i in high_fold_train[fold-1] if int(i[1]) == 0]
        high_fold_train_o = [i for i in high_fold_train[fold-1] if int(i[1]) == 2]

        # 抽取出各size层验证集并组合
        validation_set = low_fold_train_p[0:int(len(low_fold_train_p) * validation_r)] + \
                         low_fold_train_n[0:int(len(low_fold_train_n) * validation_r)] + \
                         low_fold_train_o[0:int(len(low_fold_train_o) * validation_r)] + \
                            mid_fold_train_p[0:int(len(mid_fold_train_p) * validation_r)] + \
                            mid_fold_train_n[0:int(len(mid_fold_train_n) * validation_r)] + \
                            mid_fold_train_o[0:int(len(mid_fold_train_o) * validation_r)] + \
                            high_fold_train_p[0:int(len(high_fold_train_p) * validation_r)] + \
                            high_fold_train_n[0:int(len(high_fold_train_n) * validation_r)] + \
                            high_fold_train_o[0:int(len(high_fold_train_o) * validation_r)]

        # 抽取出各size层训练集并组合
        train_set = low_fold_train_p[int(len(low_fold_train_p) * validation_r):] + \
                         low_fold_train_n[int(len(low_fold_train_n) * validation_r):] + \
                            low_fold_train_o[int(len(low_fold_train_o) * validation_r):] + \
                         mid_fold_train_p[int(len(mid_fold_train_p) * validation_r):] + \
                         mid_fold_train_n[int(len(mid_fold_train_n) * validation_r):] + \
                            mid_fold_train_o[int(len(mid_fold_train_o) * validation_r):] + \
                         high_fold_train_p[int(len(high_fold_train_p) * validation_r):] + \
                         high_fold_train_n[int(len(high_fold_train_n) * validation_r):] + \
                            high_fold_train_o[int(len(high_fold_train_o) * validation_r):]

        test_set = low_fold_test[fold-1]+mid_fold_test[fold-1]+high_fold_test[fold-1]

        # 那我如果还要返回哪些id的图片作为了训练集哪些作为了验证集和测试集呢，该怎么添加代码？
        # 你可以在这里添加代码，将训练集、验证集和测试集的id返回，以便在之后的代码中使用
        # 例如：return [train_set, validation_set, test_set, train_id, validation_id, test_id]


        return [train_set, validation_set, test_set]

def DivideData(config):

    csv_path = config.csv_file
    K = config.K
    fold = config.fold
    validation = config.validation
    validation_r = config.validation_r

    # 划分数据集
    train_set, validation_set, test_set = get_fold_filelist(csv_path, K, fold, validation, validation_r)

    # 将train_set按照id号排序
    train_set = sorted(train_set, key=lambda x: int(x[0][:-4]))
    validation_set = sorted(validation_set, key=lambda x: int(x[0][:-4]))
    test_set = sorted(test_set, key=lambda x: int(x[0][:-4]))


    # 读取训练集、验证集和测试集的id
    train_id = [i[0] for i in train_set]
    validation_id = [i[0] for i in validation_set]
    test_id = [i[0] for i in test_set]

    # 读取训练集、验证集和测试集的label
    train_label = [i[1] for i in train_set]
    validation_label = [i[1] for i in validation_set]
    test_label = [i[1] for i in test_set]

    # 读取训练集、验证集和测试集的size
    train_size = [i[2] for i in train_set]
    validation_size = [i[2] for i in validation_set]
    test_size = [i[2] for i in test_set]

    # 将训练集、验证集和测试集的id、label和size汇总起来保存到一个log文件中
    # 检测log文件夹是否存在，存在则删除，不存在则创建
    log_name = config.divide_log_name.split('/')[-1]
    # if os.path.exists(config.log_path + log_name + '_' + str(K) + '.txt'):
    #     shutil.rmtree(config.log_path + log_name + '_' + str(K) + '.txt')

    divided_log_path = os.path.join(config.log_path + log_name + '_' + str(fold) + '.txt')
    with open(divided_log_path, 'w') as f:
        '''
        格式如下：
        train set：
        id: 1, label: 0, size: 0
        id: 2, label: 1, size: 1
        ...

        validation set：
        id: 1, label: 0, size: 0
        id: 2, label: 1, size: 1
        ...

        test set：
        id: 1, label: 0, size: 0
        id: 2, label: 1, size: 1
        ...

        total:
        train set: label 0: 100, label 1: 100, label 2: 100
        validation set: label 0: 100, label 1: 100, label 2: 100
        test set: label 0: 100, label 1: 100, label 2: 100
        '''
        f.write("train set：\n")
        for i, (id, label, size) in enumerate(zip(train_id, train_label, train_size)):
            f.write(f"id: {id}, label: {label}, size: {size}\n")

        f.write("\nvalidation set：\n")
        for i, (id, label, size) in enumerate(zip(validation_id, validation_label, validation_size)):
            f.write(f"id: {id}, label: {label}, size: {size}\n")

        f.write("\ntest set：\n")
        for i, (id, label, size) in enumerate(zip(test_id, test_label, test_size)):
            f.write(f"id: {id}, label: {label}, size: {size}\n")

        # 计算每个集合中每个标签的数量
        train_label_counts = Counter(train_label)
        validation_label_counts = Counter(validation_label)
        test_label_counts = Counter(test_label)

        sum = 0
        f.write("\ntotal:\n")
        f.write("train set: \n")
        for label, count in sorted(train_label_counts.items()):
            f.write(f"        label {label}: {count}")
            sum += count
        f.write(f"       train set total: {sum}")
        sum = 0

        f.write("\nvalidation set: \n")
        for label, count in sorted(validation_label_counts.items()):
            f.write(f"        label {label}: {count}")
            sum += count
        f.write(f"       validation set total: {sum}")
        sum = 0

        f.write("\ntest set: \n")
        for label, count in sorted(test_label_counts.items()):
            f.write(f"        label {label}: {count}")
            sum += count
        f.write(f"       test set total: {sum}")


    # save_fold_path
    save_fold_path = config.save_fold_path
    if not os.path.exists(save_fold_path):
        os.makedirs(save_fold_path)

    # 在这个文件夹下新建五个文件夹，分别存放fold1、fold2、fold3、fold4、fold5
    for i in range(K):
        fold_path = os.path.join(save_fold_path, 'fold' + str(i + 1))
        if not os.path.exists(fold_path):
            os.makedirs(fold_path)

    # 现在是在fold1中新建train、validation和test文件夹
    for i in range(K):
        fold_path = os.path.join(save_fold_path, 'fold' + str(i + 1))
        train_path = os.path.join(fold_path, 'train')
        validation_path = os.path.join(fold_path, 'validation')
        test_path = os.path.join(fold_path, 'test')
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

    # 再在每个train、validation和test文件夹下面新建images,mask,contour,dist_contour,contour_mask,dist_signed文件夹
    for i in range(K):
        fold_path = os.path.join(save_fold_path, 'fold' + str(i + 1))
        train_path = os.path.join(fold_path, 'train')
        validation_path = os.path.join(fold_path, 'validation')
        test_path = os.path.join(fold_path, 'test')

        train_images_path = os.path.join(train_path, 'images')
        train_mask_path = os.path.join(train_path, 'mask')
        train_contour_path = os.path.join(train_path, 'contour')
        train_dist_contour_path = os.path.join(train_path, 'dist_contour')
        train_dist_mask_path = os.path.join(train_path, 'dist_mask')
        train_dist_signed_path = os.path.join(train_path, 'dist_signed')

        validation_images_path = os.path.join(validation_path, 'images')
        validation_mask_path = os.path.join(validation_path, 'mask')
        validation_contour_path = os.path.join(validation_path, 'contour')
        validation_dist_contour_path = os.path.join(validation_path, 'dist_contour')
        validation_dist_mask_path = os.path.join(validation_path, 'dist_mask')
        validation_dist_signed_path = os.path.join(validation_path, 'dist_signed')

        test_images_path = os.path.join(test_path, 'images')
        test_mask_path = os.path.join(test_path, 'mask')
        test_contour_path = os.path.join(test_path, 'contour')
        test_dist_contour_path = os.path.join(test_path, 'dist_contour')
        test_dist_mask_path = os.path.join(test_path, 'dist_mask')
        test_dist_signed_path = os.path.join(test_path, 'dist_signed')

        if not os.path.exists(train_images_path):
            os.makedirs(train_images_path)
        if not os.path.exists(train_mask_path):
            os.makedirs(train_mask_path)
        if not os.path.exists(train_contour_path):
            os.makedirs(train_contour_path)
        if not os.path.exists(train_dist_contour_path):
            os.makedirs(train_dist_contour_path)
        if not os.path.exists(train_dist_mask_path):
            os.makedirs(train_dist_mask_path)
        if not os.path.exists(train_dist_signed_path):
            os.makedirs(train_dist_signed_path)

        if not os.path.exists(validation_images_path):
            os.makedirs(validation_images_path)
        if not os.path.exists(validation_mask_path):
            os.makedirs(validation_mask_path)
        if not os.path.exists(validation_contour_path):
            os.makedirs(validation_contour_path)
        if not os.path.exists(validation_dist_contour_path):
            os.makedirs(validation_dist_contour_path)
        if not os.path.exists(validation_dist_mask_path):
            os.makedirs(validation_dist_mask_path)
        if not os.path.exists(validation_dist_signed_path):
            os.makedirs(validation_dist_signed_path)

        if not os.path.exists(test_images_path):
            os.makedirs(test_images_path)
        if not os.path.exists(test_mask_path):
            os.makedirs(test_mask_path)
        if not os.path.exists(test_contour_path):
            os.makedirs(test_contour_path)
        if not os.path.exists(test_dist_contour_path):
            os.makedirs(test_dist_contour_path)
        if not os.path.exists(test_dist_mask_path):
            os.makedirs(test_dist_mask_path)
        if not os.path.exists(test_dist_signed_path):
            os.makedirs(test_dist_signed_path)

    # 现在是Fold1，将p_mask中的mask文件按照train_set, validation_set, test_set复制到对应的./train_path/fold1/...的mask文件夹下，其他的如images,contour,dist_contour,dist_mask,dist_signed也是如此
    fold_idx = fold - 1
    i = fold_idx
    fold_path = os.path.join(save_fold_path, 'fold' + str(i + 1))
    train_path = os.path.join(fold_path, 'train')
    validation_path = os.path.join(fold_path, 'validation')
    test_path = os.path.join(fold_path, 'test')

    train_images_path = os.path.join(train_path, 'images')
    train_mask_path = os.path.join(train_path, 'mask')
    train_contour_path = os.path.join(train_path, 'contour')
    train_dist_contour_path = os.path.join(train_path, 'dist_contour')
    train_dist_mask_path = os.path.join(train_path, 'dist_mask')
    train_dist_signed_path = os.path.join(train_path, 'dist_signed')

    validation_images_path = os.path.join(validation_path, 'images')
    validation_mask_path = os.path.join(validation_path, 'mask')
    validation_contour_path = os.path.join(validation_path, 'contour')
    validation_dist_contour_path = os.path.join(validation_path, 'dist_contour')
    validation_dist_mask_path = os.path.join(validation_path, 'dist_mask')
    validation_dist_signed_path = os.path.join(validation_path, 'dist_signed')

    test_images_path = os.path.join(test_path, 'images')
    test_mask_path = os.path.join(test_path, 'mask')
    test_contour_path = os.path.join(test_path, 'contour')
    test_dist_contour_path = os.path.join(test_path, 'dist_contour')
    test_dist_mask_path = os.path.join(test_path, 'dist_mask')
    test_dist_signed_path = os.path.join(test_path, 'dist_signed')

    image_path = config.image_path
    mask_path = config.mask_path
    contour_path = config.save_contour_path
    dist_contour_path = config.save_distance_path_D2
    dist_mask_path = config.save_distance_path_D1
    dist_signed_path = config.save_distance_path_D3


    train_list = [image_path + i[0] for i in train_set]
    validation_list = [image_path + i[0] for i in validation_set]
    test_list = [image_path + i[0] for i in test_set]

    train_list_mask = [mask_path + i[0] for i in train_set]
    validation_list_mask = [mask_path + i[0] for i in validation_set]
    test_list_mask = [mask_path + i[0] for i in test_set]

    train_list_contour = [contour_path + i[0] for i in train_set]
    validation_list_contour = [contour_path + i[0] for i in validation_set]
    test_list_contour = [contour_path + i[0] for i in test_set]

    train_list_dist_contour = [dist_contour_path + i[0] for i in train_set]
    validation_list_dist_contour = [dist_contour_path + i[0] for i in validation_set]
    test_list_dist_contour = [dist_contour_path + i[0] for i in test_set]

    train_list_dist_mask = [dist_mask_path + i[0] for i in train_set]
    validation_list_dist_mask = [dist_mask_path + i[0] for i in validation_set]
    test_list_dist_mask = [dist_mask_path + i[0] for i in test_set]

    train_list_dist_signed = [dist_signed_path + i[0] for i in train_set]
    validation_list_dist_signed = [dist_signed_path + i[0] for i in validation_set]
    test_list_dist_signed = [dist_signed_path + i[0] for i in test_set]



    # 将train_set中的图片复制到train_images_path中
    for i in train_list:
        shutil.copy(i, train_images_path)
    # 将validation_set中的图片复制到validation_images_path中
    for i in validation_list:
        shutil.copy(i, validation_images_path)
    # 将test_set中的图片复制到test_images_path中
    for i in test_list:
        shutil.copy(i, test_images_path)
    print('copy images done')

    # 将train_set中的mask复制到train_mask_path中
    for i in train_list_mask:
        shutil.copy(i, train_mask_path)
    # 将validation_set中的mask复制到validation_mask_path中
    for i in validation_list_mask:
        shutil.copy(i, validation_mask_path)
    # 将test_set中的mask复制到test_mask_path中
    for i in test_list_mask:
        shutil.copy(i, test_mask_path)
    print('copy mask done')

    # 将train_set中的contour复制到train_contour_path中
    for i in train_list_contour:
        shutil.copy(i, train_contour_path)
    # 将validation_set中的contour复制到validation_contour_path中
    for i in validation_list_contour:
        shutil.copy(i, validation_contour_path)
    # 将test_set中的contour复制到test_contour_path中
    for i in test_list_contour:
        shutil.copy(i, test_contour_path)
    print('copy contour done')


    # 将train_set中的dist_contour复制到train_dist_contour_path中
    for i in train_list_dist_contour:
        shutil.copy(i, train_dist_contour_path)
    # 将validation_set中的dist_contour复制到validation_dist_contour_path中
    for i in validation_list_dist_contour:
        shutil.copy(i, validation_dist_contour_path)
    # 将test_set中的dist_contour复制到test_dist_contour_path中
    for i in test_list_dist_contour:
        shutil.copy(i, test_dist_contour_path)
    print('copy dist_contour done')


    # 将train_set中的dist_mask复制到train_dist_mask_path中
    for i in train_list_dist_mask:
        shutil.copy(i, train_dist_mask_path)
    # 将validation_set中的dist_mask复制到validation_dist_mask_path中
    for i in validation_list_dist_mask:
        shutil.copy(i, validation_dist_mask_path)
    # 将test_set中的dist_mask复制到test_dist_mask_path中
    for i in test_list_dist_mask:
        shutil.copy(i, test_dist_mask_path)
    print('copy dist_mask done')


    # 将train_set中的dist_signed复制到train_dist_signed_path中
    for i in train_list_dist_signed:
        shutil.copy(i, train_dist_signed_path)
    # 将validation_set中的dist_signed复制到validation_dist_signed_path中
    for i in validation_list_dist_signed:
        shutil.copy(i, validation_dist_signed_path)
    # 将test_set中的dist_signed复制到test_dist_signed_path中
    for i in test_list_dist_signed:
        shutil.copy(i, test_dist_signed_path)
    print('copy dist_signed done')
    sleep(1)

    print("train_set: ", len(train_set))
    print("validation_set: ", len(validation_set))
    print("test_set: ", len(test_set))


if __name__ == "__main__":
    # 读取配置文件，以及创建各个轮廓图距离图并保存
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, default='./train_path/p_images/', help="path to img png files")
    parser.add_argument("--mask_path", type=str, default='./train_path/p_mask/', help="path to mask png files")
    parser.add_argument("--save_contour_path", type=str, default='./train_path/p_contour/', help="contour png")
    parser.add_argument("--save_distance_path", type=str, default='./train_path/p_distance/', help="distance png")
    parser.add_argument("--project_name", type=str, default='contoursaving ', help="project name")
    parser.add_argument("--log_path", type=str, default='./train_path/', help="project name")
    parser.add_argument("--csv_file", type=str, default='./train_path/train.csv', help="csv_file")
    parser.add_argument("--image_path", type=str, default='./train_path/p_image/', help="image_path")
    parser.add_argument("--contour_flag", type=bool, default=True, help="if True, generate contour")
    parser.add_argument("--distance_flag", type=bool, default=True, help="if True, generate distance")
    parser.add_argument("--save_distance_path_D1", type=str, default='./train_path/p_distance_D1/',
                        help="distance D1 png")
    parser.add_argument("--save_distance_path_D2", type=str, default='./train_path/p_distance_D2/',
                        help="distance D2 png")
    parser.add_argument("--save_distance_path_D3", type=str, default='./train_path/p_distance_D3/',
                        help="distance D3 png")

    # 以下是划分数据集的参数
    parser.add_argument("--K", type=int, default=5, help="K fold")
    parser.add_argument("--fold", type=int, default=5, help="fold")
    parser.add_argument("--validation", type=bool, default=False, help="if True, generate validation")
    parser.add_argument("--validation_r", type=float, default=0.2, help="validation ratio")
    parser.add_argument("--divide_log_name", type=str, default='./train_path/divide_log', help="divide log path")
    parser.add_argument("--save_fold_path", type=str, default='./train_path/fold/', help="fold path")
    parser.add_argument("--save_fold_path_f1", type=str, default='./train_path/fold_D1/', help="fold path")

    # parser.add_argument("--random_state", type=int, default=1, help="random state")
    # parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    # parser.add_argument("--num_workers", type=int, default=0, help="num workers")
    # parser.add_argument("--epochs", type=int, default=100, help="epochs")


    args = parser.parse_args()
    config = args
    CreateContourDistance(config)
    # DivideData(config)
