import os
from torch.utils.data import DataLoader
import glob
import logging
import random

from utilize.solver import Solver
from utils import visualize, evaluate, create_train_arg_parser
from dataset import DatasetImageMaskContourDist
from utilize.ExpAnalize import ResultSaver


def char_color(s, front=50, word=32):
    """
    # 改变字符串颜色的函数
    :param s:
    :param front:
    :param word:
    :return:
    """
    new_char = "\033[0;" + str(int(word)) + ";" + str(int(front)) + "m" + s + "\033[0m"
    return new_char

def main(args):
    # logging.basicConfig()函数对日志的输出格式及方式做相关配置
    logging.basicConfig(
        filename="".format(args.object_type),  # 日志文件名
        filemode="a",  # 写入模式，a表示追加
        format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",  # 日志格式
        datefmt="%Y-%m-%d %H:%M",  # 时间格式
        level=logging.INFO,  # 日志级别
    )
    logging.info("")  # 日志记录

    # glob是python的一个模块，用于查找符合特定规则的文件路径名，glob.glob()返回所有匹配的文件路径列表
    # os.path.join(args.train_path, "*.jpg")是指在args.train_path路径下查找所有.jpg文件, 然后将其路径存入train_file_names
    train_path = args.train_path + str(args.fold_id) + '/train/images/'  # train_path是指训练集图片路径
    val_path = args.train_path + str(args.fold_id) + '/validation/images/'  # val_path是指验证集图片路径
    test_path = args.train_path + str(args.fold_id) + '/test/images/'  # test_path是指测试集图片路径
    train_file_names = glob.glob(train_path + "*.png")  # 获取训练集图片路径

    # 为了避免模型只记住了数据的顺序，而非真正的特征，代码使用了 random.shuffle() 函数对 train_file_names
    # 变量中存储的图片路径进行了随机打乱操作，从而增加了数据的随机性，更有助于训练出鲁棒性更强的模型。
    random.shuffle(train_file_names)  # 打乱训练集图片路径
    val_file_names = glob.glob(val_path + "*.png")  # 获取验证集图片路径
    test_file_names = glob.glob(test_path + "*.png")  # 获取测试集图片路径

    # todo: add TTA here

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_file_names, args.distance_type, args.normal_flag),
        batch_size=args.batch_size,
        # batch_size=50,
        num_workers=5,
    )
    devLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type, args.normal_flag),
        batch_size=args.val_batch_size,
        num_workers=5,
    )
    displayLoader = DataLoader(
        DatasetImageMaskContourDist(val_file_names, args.distance_type, args.normal_flag),
        # batch_size=args.val_batch_size,
        num_workers=5,
    )
    testLoader = DataLoader(
        DatasetImageMaskContourDist(test_file_names, args.distance_type, args.normal_flag),
        num_workers=5,
        batch_size=10,
    )
    # solver = Solver(args, testLoader, devLoader, testLoader)

    # if args.is_use_hyper_search:
    #     solver.hyper_search(devLoader, testLoader)
    #     print("Finished Hyper Search")
    #     return

    # Train and sample the images
    if args.mode == 'train':
        if args.fold_all_train:
            for args.fold_id in range(1, 6):
                args.project_name = 'unetDCAN_01_f' + str(args.fold_id)
                solver = Solver(args, trainLoader, devLoader, testLoader)
                solver.train()
        else:
            solver = Solver(args, trainLoader, devLoader, testLoader)
            solver.train()

    print("Finished Training")

    # 整理结果
    organize_result = ResultSaver(args.save_path, args.exp_prefix, args.record_n_rows)

if __name__ == "__main__":
    args = create_train_arg_parser().parse_args()  # 获取参数
    main(args)

