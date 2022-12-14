# -*- coding: UTF-8 -*-
# @Time    : 04/02/2020 10:58
# @Author  : BubblyYi
# @FileName: seeds_net_data_provider_aug.py
# @Software: PyCharm

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import os
import numpy as np
import SimpleITK as sitk
import random
import torchio as tio


class DataGenerater(Dataset):

    def __init__(self,data_path, pre_fix_path, transform = None, flag = '', target_transform = None):

        self.flag = flag
        data = []
        print("csv path:",data_path)
        csv_data = pd.read_csv(data_path)
        x_data = csv_data['patch_name']
        if self.flag == 'train' or self.flag == 'val':
            proximity = csv_data["proximity"]

            for i in range(len(x_data)):
                print(f'seeds_net_data_provider_aug DataGenerater __init__ if: {i}')
                if pre_fix_path is None:
                    data.append((temp, proximity[i]))
                else:
                    temp = os.path.join(pre_fix_path,x_data[i])
                    data.append((temp, proximity[i]))
        else:
            for i in range(len(x_data)):
                print(f'seeds_net_data_provider_aug DataGenerater __init__ else: {i}')
                if pre_fix_path is None:
                    data.append(x_data[i])
                else:
                    temp = os.path.join(pre_fix_path, x_data[i])
                    data.append(temp)

        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.p_gaussian_noise = 0.2

    def __getitem__(self, index):
        if self.flag == 'train':
            data_path, p = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            proximity = p
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            img = (img - mean_intensity) / (std_intensity+1e-9)
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
            img = img.unsqueeze(0)

            transforms = tio.Compose([
                tio.OneOf({
                    tio.RandomFlip(): 0.3,
                    tio.RandomAffine(scales=(0.9, 1.2), degrees=15): 0.3,
                    tio.RandomElasticDeformation(num_control_points=(6, 6, 6)): 0.4
                }, p=0.4),
                tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=2, p=0.2),
                tio.RandomBlur(p=0.25),
                tio.RandomNoise(p=0.2),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.15),
            ])
            img_transformed = transforms(img)

            return img_transformed, proximity

        elif self.flag == 'val':
            data_path, p = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            proximity = p
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            img = (img - mean_intensity) / (std_intensity + 1e-9)
            img = img.astype(np.float32)
            img = torch.from_numpy(img)

            return img.unsqueeze(0), proximity

        elif self.flag == 'test':
            data_path = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            # ?????????0
            img = (img - mean_intensity) / (std_intensity+1e-9)
            img = torch.from_numpy(img)
            return img.unsqueeze(0)

    def augment_gaussian_noise(self,data_sample, noise_variance=(0, 0.1)):
        if noise_variance[0] == noise_variance[1]:
            variance = noise_variance[0]
        else:

            variance = random.uniform(noise_variance[0], noise_variance[1])
        data_sample = data_sample + np.random.normal(0.0, variance, size=data_sample.shape)

        return data_sample

    def __len__(self):
        return len(self.data)