# -*- coding: UTF-8 -*-
# @Time    : 12/05/2020 20:06
# @Author  : BubblyYi
# @FileName: patch_generater.py
# @Software: PyCharm

import configparser
import SimpleITK as sitk
import matplotlib
matplotlib.use('AGG')
import numpy as np
import pandas as pd
import os
from glob import glob
np.random.seed(4)
from utils import resample, get_proximity, get_closer_distence

def creat_data(path_name,spacing_path,save_num,cut_size = 19,move_step = 3):
    spacing_info = np.loadtxt(spacing_path, delimiter=",", dtype=np.float32)
    proximity_list = []
    patch_name = []
    i = save_num
    print("processing dataset %d" % i)

    if i < 10:
        image_pre_fix = path_name + '0' + str(i) + '/' + 'image' + '0' + str(i)
    else:
        image_pre_fix = path_name + str(i) + '/' + 'image' + str(i)

    file_name = image_pre_fix + '.nii.gz'
    src_array = sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32))

    spacing_x = spacing_info[i][0]
    spacing_y = spacing_info[i][1]
    spacing_z = spacing_info[i][2]
    re_spacing_img, curr_spacing, resize_factor = resample(src_array,
                                                           np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))
    vessels = []

    ds_dir = os.path.dirname(image_pre_fix)
    vessel_dirs = sorted(glob(os.path.join(ds_dir, 'vessel*')))

    for vessel_dir in vessel_dirs:
        reference_path = os.path.join(vessel_dir, 'reference.txt')
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        center = txt_data[..., 0:3]
        vessels.append(center)

    z, h, w = re_spacing_img.shape

    error_count: int = 0
    for iz in range(int((z - cut_size) / move_step + 1)):
        for ih in range(int((h - cut_size) / move_step + 1)):
            for iw in range(int((w - cut_size) / move_step + 1)):
                sz = iz * move_step
                ez = iz * move_step+cut_size

                sh = ih * move_step
                eh = ih * move_step+cut_size

                sw = iw * move_step
                ew = iw * move_step+cut_size
                center_z = (ez - sz) // 2 + sz
                center_y = (eh - sh) // 2 + sh
                center_x = (ew - sw) // 2 + sw
                target_point = np.array([center_x,center_y,center_z])
                print("new center:",target_point)
                min_dis = get_closer_distence(vessels, target_point)
                print('min dis:',min_dis)
                curr_proximity = get_proximity(min_dis)
                print('proximity:',curr_proximity)
                if curr_proximity<=0.0:
                    proximity_list.append(curr_proximity)
                    new_src_arr = np.zeros((cut_size, cut_size, cut_size))
                    for ind in range(sz, ez):
                        src_temp = re_spacing_img[ind].copy()
                        new_src_arr[ind - sz] = src_temp[sh:eh, sw:ew]

                    folder_path = './patch_data/seeds_patch/negative/'+'gp_' + str(move_step)+'/d'+str(i)
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                    record_name = 'seeds_patch/negative/' + 'gp_' + str(move_step)+'/d'+str(i)+'/' + 'd_' + str(i) + '_' + 'x_' + str(center_x) + '_y_'+str(center_y)+'_z_'+str(center_z)+'.nii.gz'
                    # print(record_name)
                    org_name = './patch_data/' + record_name
                    out = sitk.GetImageFromArray(new_src_arr)
                    sitk.WriteImage(out, org_name)
                    patch_name.append(record_name)

    return patch_name, proximity_list, error_count

def create_patch_images(path_name,spacing_path,cut_size = 19,move_step = 19):
    datasets = sorted(glob(path_name + '*'))

    for dataset in datasets:
        ds_idx = int(dataset[-2:])
        patch_name, proximity_list, error_count = creat_data(path_name, spacing_path, ds_idx, cut_size, move_step)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'proximity': proximity_list})
        print(dataframe.head())
        csv_name = "./patch_data/seeds_patch/negative/" + 'gp_' + str(
            move_step) + '/'+'d' + str(ds_idx) + "_patch_info.csv"
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'proximity'], sep=',')
        print("create patch info csv")
        print("down")

        if error_count > 0:
            # Сохранение информационного файла
            error_count_message: str = "Error point count that out of image borders: " + str(error_count)
            print(error_count_message)
            info_file_path: str = "./patch_data/seeds_patch/negative/" + 'gp_' + str(move_step) + '/'+'d' + '.txt'
            f = open(info_file_path, 'w')
            f.write(error_count_message)
            f.close()


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.ini'))
path_name = config['PATH']['input_folder']

spacing_path = 'spacing_info.csv'

create_patch_images(path_name,spacing_path,cut_size = 19,move_step = 19)