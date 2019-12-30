import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import os

from dataset import Brain_Segmentation_Dataset
from Data_loader import data_loader
from model import UNet
import  argparse
from model_utils import postprocess_per_volume


parser = argparse.ArgumentParser(description='Brain Cancer Segmentation Hyper Parameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 64)')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
parser.add_argument('--image_size', type=int, default=256, help='image size (default: 256)')
parser.add_argument('--aug_scale', type=float, default=0.05, help='Augmentation Scale in Data Augmentation')
parser.add_argument('--aug_angle', type=int, default=15, help='Augmentation Angle rotation in Data Augmentation')
args = parser.parse_args()

state_dict = torch.load(os.path.join('./', 'unet.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device : ", device)

unet = UNet(in_channels=Brain_Segmentation_Dataset.in_channels, out_channels=Brain_Segmentation_Dataset.out_channels)
print(unet)
unet.to(device)

unet.load_state_dict(state_dict)
unet.eval()

train_loader, valid_loader = data_loader(args.batch_size, args.workers, args.image_size, args.aug_scale, args.aug_angle)

input_list = []
pred_list = []
true_list = []

for i, data in enumerate(valid_loader):
    x, y_true = data
    x, y_true = x.to(device), y_true.to(device)
    with torch.set_grad_enabled(False):
        y_pred = unet(x)
        y_pred_np = y_pred.detach().cpu().numpy()
        pred_list.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
        y_true_np = y_true.detach().cpu().numpy()
        true_list.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
        x_np = x.detach().cpu().numpy()
        input_list.extend([x_np[s] for s in range(x_np.shape[0])])

volumes = postprocess_per_volume(
    input_list,
    pred_list,
    true_list,
    valid_loader.dataset.patient_slice_index,
    valid_loader.dataset.patients,
)

def gray2rgb(image):
    w, h = image.shape
    image += np.abs(np.min(image)) #np.abs : 절대값을 구하는 함수.
    image_max = np.abs(np.max(image))
    if image_max > 0:
        image /= image_max
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = image * 255
    return ret


def outline(image, mask, color):
    mask = np.round(mask)
    yy, xx = np.nonzero(mask)
    for y, x in zip(yy, xx):
        if 0.0 < np.mean(mask[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]) < 1.0:
            image[max(0, y) : y + 1, max(0, x) : x + 1] = color
    return image


for p in volumes:  # 이렇게 하면 key가 출력됨.
    x = volumes[p][0]
    y_pred = volumes[p][1]
    y_true = volumes[p][2]

    for s in range(x.shape[0]):
        image = gray2rgb(x[s, 1])
        image = outline(image, y_pred[s, 0], color=[255, 0, 0])
        image = outline(image, y_true[s, 0], color=[0, 255, 0])
        print(image.shape)
        plt.imshow(image)
        plt.show()