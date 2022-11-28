# -*- coding: UTF-8 -*-
# @Time    : 25/05/2020 16:36
# @Author  : BubblyYi
# @FileName: creat_spacinginfo_data_tool.py
# @Software: PyCharm

import configparser
import os
from pathlib import Path
from glob import glob
import SimpleITK as sitk
import pandas as pd



# Through this script, the mhd data in the CAT08 data set can be automatically read and converted into nii.gz format data


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.ini'))
path_name = config['PATH']['input_folder']

datasets = sorted(glob(path_name + '*'))

spacing_x = []
spacing_y = []
spacing_z = []

for dataset in datasets:
    dataset_index: str = Path(dataset).name[7:]
    pre_fix = path_name + dataset_index + '/' + 'image' + dataset_index
    file_name = pre_fix + '.nii.gz'
    print(file_name)
    save_name = pre_fix + '.nii.gz'
    print(save_name)
    itkimage = sitk.ReadImage(file_name)
    spacing = itkimage.GetSpacing()
    spacing_x.append(spacing[0])
    spacing_y.append(spacing[1])
    spacing_z.append(spacing[2])
    print("spacing", spacing)
    out_arr = sitk.GetArrayFromImage(itkimage)
    out = sitk.GetImageFromArray(out_arr)
    sitk.WriteImage(out, save_name)
    print(out_arr.shape)

dataframe = pd.DataFrame({'spacing_x': spacing_x, 'spacing_y': spacing_y, 'spacing_z': spacing_z})
print(dataframe.head())
csv_name = "spacing_info.csv"
dataframe.to_csv(csv_name, index=False,sep=',', header=None)
print("create spacing_info csv")
print("down")