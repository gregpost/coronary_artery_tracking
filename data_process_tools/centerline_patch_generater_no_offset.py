# -*- coding: UTF-8 -*-
# @Time    : 12/05/2020 20:06
# @Author  : BubblyYi
# @FileName: patch_generater.py
# @Software: PyCharm

import configparser
import SimpleITK as sitk
import numpy as np
import pandas as pd
import os
from glob import glob
np.random.seed(4)
from utils import resample, get_spacing_res2, get_start_ind, get_end_ind, get_new_radial_ind, get_shell, get_pre_next_point_ind, find_closer_point_angle


def creat_data(max_points, path_name, spacing_path, gap_size, save_num):
    spacing_info = np.loadtxt(spacing_path,
                              delimiter=",", dtype=np.float32)
    pre_ind_list = []
    next_ind_list = []
    radials_list = []
    patch_name = []
    i = save_num
    print("processing dataset %d" % i)

    if i < 10:
        image_pre_fix = path_name + '0' + str(i) + '/' + 'image' + '0' + str(i)
    else:
        image_pre_fix = path_name + str(i) + '/' + 'image' + str(i)

    file_name = image_pre_fix + '.nii.gz'
    src_array = sitk.GetArrayFromImage(sitk.ReadImage(file_name, sitk.sitkFloat32))
    image = sitk.ReadImage(file_name)
    spacing_x = spacing_info[i][0]
    spacing_y = spacing_info[i][1]
    spacing_z = spacing_info[i][2]
    re_spacing_img, curr_spacing, resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([0.5, 0.5, 0.5]))

    image.SetSpacing([float(spacing_x), float(spacing_y), float(spacing_z)])

    ds_dir = os.path.dirname(image_pre_fix)
    vessels = sorted(glob(os.path.join(ds_dir, 'vessel*')))

    error_count: int = 0
    for v, vessel in enumerate(vessels):
        print("processing vessel %d" % v)
        reference_path = os.path.join(vessel, 'reference.txt')
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        center = txt_data[..., 0:3]

        radials_data = txt_data[..., 3]
        start_ind = get_start_ind(center, radials_data)

        end_ind = get_end_ind(center, radials_data)

        print("start ind:", start_ind)
        print("end ind:", end_ind)
        counter = 0

        last_center_x_pixel = -1
        last_center_y_pixel = -1
        last_center_z_pixel = -1

        for j in range(start_ind, end_ind + 1):
            if j % gap_size == 0:
                print('j:', j)
                center_x = center[j][0]
                center_y = center[j][1]
                center_z = center[j][2]

                org_x_pixel = get_spacing_res2(center_x, spacing_x, resize_factor[1])
                org_y_pixel = get_spacing_res2(center_y, spacing_y, resize_factor[2])
                org_z_pixel = get_spacing_res2(center_z, spacing_z, resize_factor[0])

                if org_x_pixel != last_center_x_pixel or org_y_pixel != last_center_y_pixel or org_z_pixel != last_center_z_pixel:
                    print("last:", [last_center_x_pixel, last_center_y_pixel, last_center_z_pixel])
                    print("curr:", [org_x_pixel, org_y_pixel, org_z_pixel])

                    last_center_x_pixel = org_x_pixel
                    last_center_y_pixel = org_y_pixel
                    last_center_z_pixel = org_z_pixel

                    radial = radials_data[j]

                    pre_ind, next_ind = get_pre_next_point_ind(center, radials_data, j)
                    if pre_ind != -1 and next_ind != -1:
                        pre_x = center[pre_ind][0]
                        pre_y = center[pre_ind][1]
                        pre_z = center[pre_ind][2]

                        next_x = center[next_ind][0]
                        next_y = center[next_ind][1]
                        next_z = center[next_ind][2]

                        sx, sy, sz = get_shell(max_points, radial)
                        shell_arr = np.zeros((len(sx), 3))
                        for s_ind in range(len(sx)):
                            shell_arr[s_ind][0] = sx[s_ind]
                            shell_arr[s_ind][1] = sy[s_ind]
                            shell_arr[s_ind][2] = sz[s_ind]

                        center_x_pixel = get_spacing_res2(center_x, spacing_x, resize_factor[1])
                        center_y_pixel = get_spacing_res2(center_y, spacing_y, resize_factor[2])
                        center_z_pixel = get_spacing_res2(center_z, spacing_z, resize_factor[0])

                        curr_c = [center_x, center_y, center_z]
                        p = [pre_x, pre_y, pre_z]
                        pre_sim = find_closer_point_angle(shell_arr, p, curr_c)
                        p = [next_x, next_y, next_z]
                        next_sim = find_closer_point_angle(shell_arr, p, curr_c)

                        pre_ind_list.append(pre_sim)
                        next_ind_list.append(next_sim)
                        radials_list.append(radial)

                        cut_size = 9

                        left_x = center_x_pixel - cut_size
                        right_x = center_x_pixel + cut_size
                        left_y = center_y_pixel - cut_size
                        right_y = center_y_pixel + cut_size
                        left_z = center_z_pixel - cut_size
                        right_z = center_z_pixel + cut_size

                        new_src_arr = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                        for ind in range(left_z, right_z + 1):
                            try:
                                src_temp = re_spacing_img[ind].copy()
                                new_src_arr[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
                            except Exception:
                                # Создадим переменную со средним значением яркости за пределами сосуда.
                                # Это нужно для того, чтобы заполнить те воксели кубика, которые оказались
                                # за пределами объёма по причине близости кубика с артерией по краям сцены.
                                # Сама артерия не выходит за пределы сцены, но кубик может выйти так как
                                # артерия находится в центре кубика, а отступ от центра равен нескольким
                                # пикселям
                                fill_value: int = -50

                                new_src_arr = np.full((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1), fill_value)
                                for ind_z in range(left_z, right_z + 1):
                                    if 0 <= ind_z < re_spacing_img.shape[0]:
                                        for ind_x in range(left_x, right_x + 1):
                                            if 0 <= ind_x < re_spacing_img.shape[1]:
                                                for ind_y in range(left_y, right_y + 1):
                                                    if 0 <= ind_y < re_spacing_img.shape[2]:
                                                        src_temp = re_spacing_img[ind_z].copy()
                                                        new_src_arr[ind_z - left_z, ind_x - left_x, ind_y - left_y] = src_temp[ind_x, ind_y]
                                break

                        folder_path = './patch_data/centerline_patch/no_offset/point_' + str(max_points) + '_gp_' + str(
                            gap_size) + '/' + 'd' + str(i)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        record_name = 'centerline_patch/no_offset/point_' + str(max_points) + '_gp_' + str(
                            gap_size) + '/' + 'd' + str(i) + '/' + 'd_' + str(i) + '_' + 'v_' + str(
                            v) + '_' + 'patch_%d' % counter + '.nii.gz'

                        org_name = './patch_data/' + record_name
                        out = sitk.GetImageFromArray(new_src_arr)
                        sitk.WriteImage(out, org_name)
                        patch_name.append(record_name)
                        counter += 1

    return pre_ind_list, next_ind_list, radials_list, patch_name, error_count


def create_patch_images(max_points,path_name,spacing_path,gap_size):
    datasets = sorted(glob(path_name + '*'))

    for dataset in datasets:
        ds_idx = int(dataset[-2:])
        pre_ind_list, next_ind_list, radials_list, patch_name, error_count = creat_data(max_points, path_name, spacing_path,
                                                                           gap_size, ds_idx)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'pre_ind': pre_ind_list, 'next_ind': next_ind_list, 'radials': radials_list})
        print(dataframe.head())
        csv_name = "./patch_data/centerline_patch/no_offset/" + 'point_' + str(max_points) + '_gp_' + str(
            gap_size) + '/' + 'd' + str(ds_idx) + "_patch_info_%d.csv" % max_points
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'pre_ind', 'next_ind', 'radials'], sep=',')
        print("create patch info csv")
        print("down")

        if error_count > 0:
            # Сохранение информационного файла
            error_count_message: str = "Error point count that out of image borders: " + str(error_count)
            print(error_count_message)
            info_file_path: str = "./patch_data/centerline_patch/no_offset/" + 'point_' + str(max_points) + '_gp_' + str(gap_size) + '/info_' + 'd' + str(ds_idx) + '.txt'
            f = open(info_file_path, 'w')
            f.write(error_count_message)
            f.close()


max_points = 500
gap_size = 1

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.ini'))
path_name = config['PATH']['input_folder']

spacing_path = 'spacing_info.csv'
create_patch_images(max_points,path_name,spacing_path,gap_size)