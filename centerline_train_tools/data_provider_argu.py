# -*- coding: UTF-8 -*-
# @Time    : 04/02/2020 10:58
# @Author  : Lijiannan
# @Update  : BubblyYi
# @FileName: data_provider.py
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

    def __init__(self,data_path, pre_fix_path, number_points, transform = None, flag = '', target_transform = None):
        self.flag = flag
        data = []
        print("csv path:",data_path)
        csv_data = pd.read_csv(data_path)
        x_data = csv_data['patch_name']
        if self.flag == 'train' or self.flag == 'val':
            pre_ind = csv_data["pre_ind"]
            next_ind = csv_data["next_ind"]
            radials = csv_data["radials"]

            for i in range(len(x_data)):
                print(f'data_provider_argu DataGenerater __init__ if i: {i}')
                if pre_fix_path is None:
                    data.append((temp, pre_ind[i], next_ind[i], radials[i]))
                else:
                    temp = os.path.join(pre_fix_path,x_data[i])
                    data.append((temp, pre_ind[i], next_ind[i], radials[i]))
        else:
            for i in range(len(x_data)):
                print(f'data_provider_argu DataGenerater __init__ else i: {i}')
                if pre_fix_path is None:
                    data.append(x_data[i])
                else:
                    temp = os.path.join(pre_fix_path, x_data[i])
                    data.append(temp)

        self.data = data
        self.transform = transform
        self.target_transform = target_transform
        self.number_points = number_points
        self.p_gaussian_noise = 0.2

    def __getitem__(self, index):
        if self.flag == 'train':
            data_path, p1, p2, r = self.data[index]  # 通过index索引返回一个图像路径fn 与 标签label
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            shell_data = np.zeros(self.number_points)

            shell_data[p1] = 0.5
            shell_data[p2] = 0.5
            radials = r
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
                tio.RandomAnisotropy(axes=(0, 1, 2), downsampling=2, p=0.3),
                tio.RandomBlur(p=0.4),
                tio.RandomNoise(p=0.3),
                tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.15),
            ])
            img_transformed = transforms(img)

            return img_transformed, shell_data, radials

        elif self.flag == 'val':
            data_path, p1, p2, r = self.data[index]  # 通过index索引返回一个图像路径fn 与 标签label
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            shell_data = np.zeros(self.number_points)

            shell_data[p1] = 0.5
            shell_data[p2] = 0.5
            radials = r
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            img = (img - mean_intensity) / (std_intensity+1e-9)
            img = img.astype(np.float32)
            img = torch.from_numpy(img)

            return img.unsqueeze(0), shell_data, radials

        elif self.flag == 'test':
            data_path = self.data[index]
            img = sitk.GetArrayFromImage(sitk.ReadImage(data_path, sitk.sitkFloat32))
            upper_bound = np.percentile(img, 99.5)
            lower_bound = np.percentile(img, 00.5)
            img = np.clip(img, lower_bound, upper_bound)
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
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