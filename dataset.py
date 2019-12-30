import os
import torch
from torch.utils.data import Dataset
from skimage.io import imread, imsave
import numpy as np
import random

class Brain_Segmantation_Dataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(self, image_dir, transform=None, image_size = 256, subset='train', random_sampling=True, seed=2019):

        # 1. Read Images
        images_dict = {}
        masks_dict = {}

        print('reading {} images........'.format(subset))
        # 1-1. image와 mask를 구분해서 각각 slice list에 담아두기. --> 데이터 폴더 안에는 mask와 image가 다 같이 들어있으니 이것을 분류해서 리스트로 갖기 위함.
        for (dir_path, dir_names, file_names) in os.walk(image_dir):
            image_slices = [] # images에 대한 파일들을 읽어서 numpy 형태로 저장된 값이 들어갈 예정
            mask_slices = [] # mask에 대한 파일들을 읽어서 numpy 형태로 저장된 값이 들어갈 예
            for file_name in sorted(filter(lambda f: '.tif' in f, file_names), key = lambda x: int(x.split('.')[-2].split('_')[4])):
                file_path = os.path.join(dir_path, file_name)
                if 'mask' in file_name:
                    mask_slices.append(imread(file_path, as_gray=True))
                else: #image
                    image_slices.append(imread(file_path))



            # 1-2. Dictionary에 환자 id별로 이미지에 대한 numpy array 값을 저장.
            if len(image_slices) > 0: # 오류가 뜨는 것은, 그건 이제 파일을 잘못 reading 했다는 의미이다.
                patient_id = dir_path.split('/')[-1] # 환자 ID (case에 해당)
                images_dict[patient_id] = np.array(image_slices[1: -1])
                masks_dict[patient_id] = np.array(mask_slices[1: -1])

        self.patients = sorted(images_dict) #각 Dictionary 별로 sorted 해서 patients에 저장함 (아직 data split은 안한 상황)

        # 2. subset(train or test) 별로 환자 case를 선택하는 구문.
        if not subset == 'all':
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=10) # validation dataset으로 10개의 case 만 random하게 뽑아서 저장.

            if subset == 'validation':
                self.patients = validation_patients
            else:
                self.patients = sorted(list(set(self.patients).difference(validation_patients)))

        # ========================================================================================= #
        # =======여기까지가 기본적인 validation dataset과 train dataset으로 split하고 전처리해주는 구문===========#
        # ========================================================================================= #

