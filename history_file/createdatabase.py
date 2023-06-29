import os
import cv2
import numpy as np
import argparse


def SaveContour(path, contour):
    # 读取CSV文件，获取每张图片的序号和类别
    csv_file = "./train_path/train.csv"
    image_path = "./train_path/p_image/"
    mask_path = "./train_path/p_mask/"
    save_contour_path = "./train_path/contour/"
    labels = {}  # lables是一个字典，key是图片的序号，value是图片的类别, 序号用[]表示，类别用()表示

    # 检测文件夹是否存在，不存在则创建
    if not os.path.exists(save_contour_path):
        os.makedirs(save_contour_path)

    with open(csv_file, "r") as f:
        for line in f:
            # 去掉第一行，因为第一行是表头
            if line.startswith("ID"):
                continue
            parts = line.strip().split(",")
            # 图片ID为001.jpg格式，所以去掉后面的.jpg
            parts_id = parts[0].split(".")[0]   # 001
            parts_jpg = parts[0].split(".")[1]  # jpg

            img_id = int(parts_id)
            label = int(parts[1])
            labels[img_id] = label


    # 循环处理每张图像
    for img_file in os.listdir(image_path):
        # 如果是图像文件
        if img_file.endswith(".png"):
            # 读取原图和掩码
            img_id = int(img_file.split(".")[0])  # 获取图片序号
            img_path = os.path.join(image_path, img_file)   # 图片路径
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            mask_file = os.path.join(mask_path, img_file)
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)  # 这里是二值化，阈值为127

            # 检测读取进来的mask是否全为零以及对应序号的csv文件中那个label是否为零
            if np.sum(mask) == 0 or labels[img_id] == 2:
                print("为正常图像，没有结节")
                # 则contour为mask掩膜图像即可
                contour_img = mask
                cv2.imwrite(save_contour_path + img_file, contour_img)
            else:
                # 复制掩码并查找轮廓
                contour_mask = mask.copy()
                contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 找到最大的轮廓
                max_contour = max(contours, key=cv2.contourArea)

                # 创建一个新图像
                contour_img = np.zeros_like(img)    # zero_like()函数创建一个和img同样大小的全0图像

                # 绘制最大的轮廓
                cv2.drawContours(contour_img, [max_contour], -1, 255, 2)    # contour_img是绘制轮廓的图像，
                # [max_contour]是轮廓，-1是轮廓的索引,代表绘制所有轮廓，255是轮廓的颜色，2是轮廓的粗细

                # 保存轮廓图像
                contour_path = os.path.join(save_contour_path, img_file)
                cv2.imwrite(save_contour_path + img_file, contour_img)

            # 输出提示信息
            print(f"Processed image {img_file} successfully.")


class ContourDistance():
    def __init__(self, config):
        self.savepath = config.save_contour_path
        self.logname = config.project_name
        self.train_path = config.train_path
        self.mask_path = config.mask_path


    # 写入log文件
    def write_log(self, log):
        with open(self.logname, 'a') as f:
            f.write(log + '\n')
            f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train setup for segmentation")
    parser.add_argument("--train_path", type=str, default='./train_path/p_images/', help="path to img png files")
    parser.add_argument("--mask_path", type=str, default='./train_path/p_mask/', help="path to mask png files")
    parser.add_argument("--save_contour_path", type=str, default='./train_path/p_contour/', help="contour png")
    parser.add_argument("--project_name", type=str, default='contoursaving ', help="project name")
    parser.add_argument("--log_path", type=str, default='./train_path/', help="project name")
    config = parser.parse_args()
    contour = ContourDistance(config)


