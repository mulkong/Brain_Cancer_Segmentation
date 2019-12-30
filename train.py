import torch
import torch.optim as optim

import numpy as np
import os
import argparse

from Data_loader import data_loader
from model import UNet
from model_utils import DiceLoss, log_loss_summary, log_scalar_summary, dsc_per_volume, postprocess_per_volume
from dataset import Brain_Segmentation_Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device : ", device)

parser = argparse.ArgumentParser(description='Brain Cancer Segmentation Hyper Parameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size in training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, help='Epochs in training')
parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
parser.add_argument('--image_size', type=int, default=256, help='image size (default: 256)')
parser.add_argument('--aug_scale', type=float, default=0.05, help='Augmentation Scale in Data Augmentation')
parser.add_argument('--aug_angle', type=int, default=15, help='Augmentation Angle rotation in Data Augmentation')
args = parser.parse_args()

train_loader, valid_loader = data_loader(args.batch_size, args.workers, args.image_size, args.aug_scale, args.aug_angle)
loader = {'train': train_loader, 'valid':valid_loader}
# a, b = next(iter(train_loader))
# image = a[0]
# mask = b[0]
# print(image.shape)
# print(mask.shape)
# image_numpy = image.numpy()
# mask_numpy = mask.numpy()
# print(image_numpy.shape)
# print(mask_numpy.shape)
# image_numpy_transpose = np.transpose(image_numpy, (1, 2, 0))
# mask_numpy_transpose = np.transpose(mask_numpy, (1, 2, 0))
# print(image_numpy_transpose.shape)
# print(mask_numpy_transpose.shape)
# plt.imshow(image_numpy_transpose)
# plt.show()
# plt.imshow(mask_numpy_transpose.squeeze(), cmap='gray')
# plt.show()

unet = UNet(in_channels=Brain_Segmentation_Dataset.in_channels, out_channels=Brain_Segmentation_Dataset.out_channels)
print(unet)
unet.to(device)

dsc_loss = DiceLoss()
best_validation_dsc = 0.0
optimizer = optim.Adam(unet.parameters(), lr=args.lr)

loss_train = []
loss_valid = []

step = 0

for epoch in range(args.epochs):
    for phase in ['train', 'valid']:
        if phase == 'train':
            unet.train()
        else:
            unet.eval()

        validation_pred = []
        validation_true = []
        for i, data in enumerate(loader[phase]):
            if phase == 'train':
                step += 1
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            optimizer.zero_grad()
            # 학습 시에만 연산 기록을 추적
            with torch.set_grad_enabled(phase == "train"):
                y_pred = unet(x)

                loss = dsc_loss(y_pred, y_true)

                if phase == 'valid':
                    loss_valid.append(loss.item())  # validation loss값 리스트에 저장.
                    y_pred_np = y_pred.detach().cpu().numpy()  # y prediction numpy
                    # extend하는게 어떤 의미인지는 잘 모루게따..
                    validation_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])
                    y_true_np = y_true.detach().cpu().numpy()
                    validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])
                if phase == 'train':
                    loss_train.append(loss.item())
                    loss.backward()
                    optimizer.step()
        if phase == 'train':
            log_loss_summary(loss_train, epoch)
            loss_train = []

        if phase == 'valid':
            log_loss_summary(loss_valid, epoch, prefix='val_')
            mean_dsc = np.mean(
                dsc_per_volume(
                    validation_pred,
                    validation_true,
                    valid_loader.dataset.patient_slice_index,
                )
            )
            log_scalar_summary("val_dsc", mean_dsc, epoch)
            if mean_dsc > best_validation_dsc:
                best_validation_dsc = mean_dsc
                torch.save(unet.state_dict(), os.path.join('./', "unet.pt"))
            loss_valid = []
print("\nBest validation mean DSC: {:4f}\n".format(best_validation_dsc))
