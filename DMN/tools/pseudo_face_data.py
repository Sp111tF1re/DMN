import os, sys
import numpy as np
import cv2
from numpy.lib.type_check import imag

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import torch

import glob

class faces_data(Dataset):
    def __init__(self, data_lr, data_hr=None, b_train=True, rgb=False, img_range=1, shuffle=True, z_size=(12, 12)):
        self.ready_hr = data_hr is not None
        if data_hr is not None:
            self.hr_files = glob.glob(os.path.join(data_hr, "**/*.jpg"), recursive=True)
            self.hr_files.sort()
        self.lr_files = glob.glob(os.path.join(data_lr, "*.jpg"), recursive=True)
        self.lr_files.sort()
        if shuffle:
            if data_hr is not None:
                np.random.shuffle(self.hr_files)
            np.random.shuffle(self.lr_files)
        ##
        self.training = b_train
        self.rgb = rgb
        self.z_size = z_size
        self.img_min_max = (0, img_range)
        if self.training:
            self.preproc = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.5),
                transforms.ToTensor()])
        ##
        else:
            self.preproc = transforms.Compose([
                transforms.ToTensor()])
    def __len__(self):
        if self.ready_hr:
            return min([len(self.lr_files), len(self.hr_files)])
        else:
            return len(self.lr_files)
    ##
    def __getitem__(self, index):
        data = dict()

        one_matrix = np.ones(self.z_size)
        if np.prod(self.z_size) > 0:
            # data["z"] = torch.unsqueeze(torch.as_tensor(one_matrix, dtype=torch.float32), dim=0)
            data["z"] = torch.randn(1, *self.z_size, dtype=torch.float32)
        lr_idx = index % len(self.lr_files)
        lr = cv2.imread(self.lr_files[lr_idx], 0)
        data["lr_original"] = self.preproc(lr) * self.img_min_max[1]
        data["lr_path"] = self.lr_files[lr_idx]
        data["lr"] = nnF.interpolate(data["lr_original"].unsqueeze(0), scale_factor=1, mode="bicubic", align_corners=False).clamp(min=self.img_min_max[0], max=self.img_min_max[1]).squeeze(0)
        if self.ready_hr:
            hr_idx = index % len(self.hr_files)
            hr = cv2.imread(self.hr_files[hr_idx], 0)
            # if self.rgb:
            #     hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
            data["hr"] = self.preproc(hr) * self.img_min_max[1]
            data["hr_path"] = self.hr_files[hr_idx]
            data["hr_down"] = nnF.interpolate(data["hr"].unsqueeze(0), scale_factor=0.5, mode="bicubic", align_corners=False, recompute_scale_factor=False).clamp(min=self.img_min_max[0], max=self.img_min_max[1]).squeeze(0)
        return data

    def get_noises(self, n):
        return torch.randn(n, 1, *self.z_size, dtype=torch.float32)

    def permute_data(self):
        if self.ready_hr:
            np.random.shuffle(self.hr_files)
        np.random.shuffle(self.lr_files)


if __name__ == "__main__":
    DATA_TRAIN = "D:/python_code/MRI_SR_ver1_1/Dataset/TRAIN"
    DATA_TEST = "D:/python_code/MRI_SR_ver1_1/Dataset/TEST/LR"
    high_folder = os.path.join(DATA_TRAIN, "HR")
    low_folder = os.path.join(DATA_TRAIN, "LR")
    test_folder = os.path.join(DATA_TEST)
    img_range = 1
    data = faces_data(low_folder, high_folder, img_range=img_range)
    flag = 0
    for i in range(len(data)):
        d = data[i]
        for elem in d:
            if elem in ['z', 'lr_path', 'hr_path']: continue
            img = np.around((d[elem].numpy().transpose(1, 2, 0) / img_range) * 255.0).astype(np.uint8)
            # flag += 1
            # cv2.imshow(elem, img[:, :, ::-1])
        # cv2.waitKey()
        # flag += 1
        # print('flag', flag)
    print(d['lr'].size())

    print("fin.")

