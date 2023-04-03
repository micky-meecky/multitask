import os
import cv2
import numpy as np
import argparse
import datetime
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
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
    parser.add_argument("--distance_flag", type=bool, default=False, help="if True, generate distance")
    parser.add_argument("--save_distance_path_D1", type=str, default='./train_path/p_distance_D1/',
                        help="distance D1 png")
    parser.add_argument("--save_distance_path_D2", type=str, default='./train_path/p_distance_D2/',
                        help="distance D2 png")
    parser.add_argument("--save_distance_path_D3", type=str, default='./train_path/p_distance_D3/',
                        help="distance D3 png")

    config = parser.parse_args()
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
