import os
from torch.utils.data import DataLoader
import glob
import logging
import random
from utilize.solver import Solver
from utils import visualize, evaluate, create_train_arg_parser
from dataset import DatasetImageMaskContourDist


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


if __name__ == "__main__":

    # 创建用于matplotlib可视化的列表
    train_loss = []
    valid_loss = []
    # sys.stdout = open('output.txt', 'w')  # 将输出重定向到output.txt文件中

    args = create_train_arg_parser().parse_args()   # 获取参数
    total_time = 0

    # optimizer param
    beta1 = args.beta1  # for adam
    beta2 = args.beta2  # for adam

    result_path = os.path.join(args.save_path + '/' + args.project_name)  # result_path是指保存模型的路径
    if not os.path.exists(result_path):  # 如果result_path路径不存在，则创建该路径
        os.makedirs(result_path)

    pretrained_model_path = result_path + '/' + args.pretrained_model_name  # pretrained_model_path是指预训练模型的路径

    # CUDA_SELECT = "cuda:{}".format(args.cuda_no)  # format功能是将字符串中的大括号{}替换为format()中的参数
    # CUDA_SELECT = "cuda"
    # log_path = result_path + "/summary/"
    # writer = SummaryWriter(log_dir=log_path)  # tensorboardX，用于可视化

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
    train_path = args.train_path + str(args.fold_id) + '/train/images/'  # train_path是指训练集图片路径
    val_path = args.train_path + str(args.fold_id) + '/validation/images/'  # val_path是指验证集图片路径
    test_path = args.train_path + str(args.fold_id) + '/test/images/'   # test_path是指测试集图片路径
    train_file_names = glob.glob(train_path + "*.png")  # 获取训练集图片路径

    # 为了避免模型只记住了数据的顺序，而非真正的特征，代码使用了 random.shuffle() 函数对 train_file_names
    # 变量中存储的图片路径进行了随机打乱操作，从而增加了数据的随机性，更有助于训练出鲁棒性更强的模型。
    random.shuffle(train_file_names)    # 打乱训练集图片路径
    val_file_names = glob.glob(val_path + "*.png")   # 获取验证集图片路径
    test_file_names = glob.glob(test_path + "*.png")  # 获取测试集图片路径


    # device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")  # 要么使用GPU，要么使用CPU


    # model = build_model(args.model_type, args.num_classes)    # 构建模型
    #
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model)
    #
    # model = model.to(device)

    # To handle epoch start number and pretrained weight
    # epoch_start = "0"   # epoch_start是指从第几个epoch开始训练
    # if args.use_pretrained:
    #     print("Loading Model {}".format(os.path.basename(pretrained_model_path)))  # basename返回文件名，不包含路径
    #     model.load_state_dict(torch.load(pretrained_model_path))   # 加载预训练模型
    #     # 返回预训练模型文件的文件名（不包含路径和文件后缀），然后使用 split(".")[0] 来去掉文件后缀部分，只留下文件名。
    #     epoch_start = os.path.basename(pretrained_model_path).split(".")[0]
    #     print(epoch_start)  # 打印epoch_start

    # todo: add TTA here

    trainLoader = DataLoader(
        DatasetImageMaskContourDist(train_file_names, args.distance_type, args.normal_flag),
        batch_size=args.batch_size,
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
        batch_size=1,
    )
    solver = Solver(args, trainLoader, devLoader, testLoader)

    # Train and sample the images
    if args.mode == 'train':
        solver.train()
    # optimizer = Adam(model.parameters(), lr=args.lr)    # 优化器
    # optimizer = Adam(model.parameters(), 1e-6, [beta1, beta2])  # 优化器
    # criterion = define_loss(args.model_type)    # 损失函数 *



    # train_len = len(trainLoader)
    # epoch_len = args.num_epochs
    # epoch_idx = 0
    # for epoch in range(int(epoch_start) + 1, int(epoch_start) + 1 + args.num_epochs):  # loop over 150 epochs
    #
    #     global_step = epoch * len(trainLoader)
    #     running_loss = 0.0
    #     tic = datetime.datetime.now()
    #     epochtic = datetime.datetime.now()
    #     print("Epoch:{}\n".format(epoch))
    #
    #     for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(trainLoader):
    #         model.train()
    #
    #         inputs = inputs.to(device)
    #         targets1 = targets1.to(device)
    #         targets2 = targets2.to(device)
    #         targets3 = targets3.to(device)
    #         targets4 = targets4.to(device)
    #
    #         targets = [targets1, targets2, targets3, targets4]
    #
    #         loss = train_model(model, inputs, targets, args.model_type, criterion, optimizer)
    #
    #         train_loss.append(loss.item())
    #         # if i % 5 == 0:
    #         #     print("Epoch:{} Step:{} Loss:{}".format(epoch, i, loss.item()))
    #
    #         writer.add_scalar("loss", loss.item(), epoch)
    #
    #         running_loss += loss.item() * inputs.size(0)
    #
    #         print_content = 'batch_total_loss:' + str(loss.data.cpu().numpy()) + ' batch_size:' + str(
    #             args.batch_size) + ' lr:' + str(args.lr)
    #
    #         if i % 2 == 0:
    #             printProgressBar(i, train_len, content=print_content)
    #         # print("sssssss")
    #
    #     epoch_loss = running_loss / len(train_file_names)
    #
    #     # 计时结束
    #     toc = datetime.datetime.now()
    #     h, remainder = divmod((toc - tic).seconds, 3600)
    #     m, s = divmod(remainder, 60)
    #     time_str = "per epoch training cost Time %02d h:%02d m:%02d s" % (h, m, s)
    #     print(char_color(time_str))
    #
    #     tic = datetime.datetime.now()
    #
    #     # validation part
    #     if epoch % 1 == 0:
    #         # todo: create metrics such as dice, jaccard, precision, recall, f1, etc.
    #
    #         dev_loss, dev_time = evaluate(device, epoch, model, devLoader, writer)
    #         writer.add_scalar("loss_valid", dev_loss, epoch)
    #         valid_loss.append(dev_loss.item())
    #         print("\nEpoch Loss:{} Val Loss:{}".format(epoch_loss, dev_loss))
    #         visualize(device, epoch, model, displayLoader, writer, args.val_batch_size)
    #     else:
    #         print("\nEpoch Loss:{} ".format(epoch_loss))

        # todo: lr decay(done)

        # todo: 针对输出内容，做一些名字规范化，还有归档处理，方便管理。

        # todo: 添加一个excel，记录每次训练的最终结果，包括loss，dice，jaccard，precision，recall，f1等。这些指标有均有两种计算方式(done)
        # todo: 一种是对这次的训练的最后10个epoch的平均值，另一种是取其中的最大值。

        # todo: 记录训练数据到txt文件中，方便后续分析。(done)

        # todo: 添加模型大小的记录，方便后续分析。(done)

        # logging.info("epoch:{} train_loss:{} ".format(epoch, epoch_loss))
        # if epoch % 5 == 0:
        #     torch.save(
        #         model.state_dict(), os.path.join(args.save_path + '/' + args.project_name, str(epoch) + ".pt")
        #     )

        # 计时结束
        # toc = datetime.datetime.now()
        # h, remainder = divmod((toc - tic).seconds, 3600)
        # m, s = divmod(remainder, 60)
        # time_str = "per epoch testing&vlidation cost Time %02d h:%02d m:%02d s" % (h, m, s)
        # print(char_color(time_str))
        #
        # epochtoc = datetime.datetime.now()
        # time_seconds = (epochtoc - epochtic).seconds
        # total_time = total_time + time_seconds
        # per_epoch_time = total_time / (epoch_idx + 1)
        # h, remainder = divmod(time_seconds, 3600)
        # m, s = divmod(remainder, 60)
        # time_str = "per whole epoch cost Time %02d h:%02d m:%02d s" % (h, m, s)
        # print(char_color(time_str))
        # remain_time_sec = (args.num_epochs - epoch_idx - 1) * per_epoch_time
        # h, remainder = divmod(remain_time_sec, 3600)
        # m, s = divmod(remainder, 60)
        # time_str = "perhaps need Time %02d h:%02d m:%02d s" % (h, m, s)
        # print(char_color(time_str))
        #
        # epoch_idx += 1

    # 绘制训练损失的折线图


    print("Finished Training")