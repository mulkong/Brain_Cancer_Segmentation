import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

import matplotlib.pyplot as plt
#from matplotlib.backends.backend_agg import FigureCanvasAgg
import tqdm as tqdm
import numpy as np
import os
import random
import argparse

from Data_loader import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Brain Cancer Segmentation Hyper Parameter')
parser.add_argument('--batch_size', type=int, default=64, help='batch size in training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, help='Epochs in training')
parser.add_argument('--lr', type=float, default=1e-04, help='learning rate (default: 0.0001)')
parser.add_argument('--workers', type=int, default=4, help='number of workers in dataset loader (default: 4)')
parser.add_argument('--image_size', type=int, default=256, help='image size (default: 256)')
parser.add_argument('--aug_scale', type=float, default=0.05, help='Augmentation Scale in Data Augmentation')
parser.add_argument('--aug_angle', type=int, default=15, help='Augmentation Angle rotation in Data Augmentation')
args = parser.parse_args()

train_loader, valid_loader = data_loader(args.batch_size, args.workers, args.image_size, args.aug_scale, args.aug_angle)
