import torch
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from dataset import DatasetImageMaskContourDist
import glob
from models import UNet, UNet_DCAN, UNet_DMTN, PsiNet, UNet_ConvMCD
from tqdm import tqdm
import numpy as np
import cv2
from utils import create_validation_arg_parser

# def cuda(x):
# return x.cuda(async=True) if torch.cuda.is_available() else x


def build_model(model_type):

    if model_type == "unet":
        model = UNet(input_channels=1, num_classes=2)
    if model_type == "dcan":
        model = UNet_DCAN(num_classes=2)
    if model_type == "dmtn":
        model = UNet_DMTN(num_classes=2)
    if model_type == "psinet":
        model = PsiNet(num_classes=2)
    if model_type == "convmcd":
        model = UNet_ConvMCD(num_classes=2)

    return model


if __name__ == "__main__":

    args = create_validation_arg_parser().parse_args()
    val_path_img = args.val_path + str(args.fold_id) + '/validation/images/'
    val_path = os.path.join(val_path_img, "*.png")
    model_file = args.model_file + args.project_name + '/' + args.pretrained_model_name
    # model_file = args.model_file
    save_path = args.save_path
    model_type = args.model_type

    cuda_no = args.cuda_no
    CUDA_SELECT = "cuda:{}".format(cuda_no)
    device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")

    val_file_names = glob.glob(val_path)
    valLoader = DataLoader(DatasetImageMaskContourDist(val_file_names, args.distance_type), num_workers=5)

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = build_model(model_type)
    model = model.to(device)
    model.load_state_dict(torch.load(model_file))
    model.eval()

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(
        tqdm(valLoader)
    ):

        inputs = inputs.to(device)
        # outputs1, outputs2, outputs3 = model(inputs)
        outputs1 = model(inputs)

        # 对outputs1中的每个Tensor对象进行detach操作
        for i in range(len(outputs1)):
            outputs1[i] = outputs1[i].detach().cpu().numpy().squeeze()
        res = np.zeros((256, 256))
        indices = np.argmax(outputs1[0], axis=0)
        output = outputs1[0][1]
        output = 1 / (1 + np.exp(-output))
        res = np.where(output > 0.5, 1, 0)
        # res[indices >= 0.5] = 255
        # res[indices < 0] = 0

        output_path = os.path.join(
            save_path, "mask_" + os.path.basename(img_file_name[0])
        )
        cv2.imwrite(output_path, res)
