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
from numpy import nanmax, argmax, unravel_index
from scipy.spatial.distance import pdist, squareform
np.random.seed(4)
from utils import resample, get_shell, get_proximity


def creat_data(path_name,spacing_path,gap_size,save_num):
    spacing_info = np.loadtxt(spacing_path,
                              delimiter=",", dtype=np.float32)
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
    re_spacing_img, curr_spacing, resize_factor = resample(src_array, np.array([spacing_z, spacing_x, spacing_y]),
                                                           np.array([1, 1, 1]))

    max_z, max_y, max_x = re_spacing_img.shape

    ds_dir = os.path.dirname(image_pre_fix)
    vessel_paths = sorted(glob(os.path.join(ds_dir, 'vessel*')))

    vessel_points = []
    for vessel in vessel_paths:
        reference_path = os.path.join(vessel, 'pointS.txt')
        txt_data = np.loadtxt(reference_path, dtype=np.float32)
        vessel_points.append(txt_data)

    distance = pdist(vessel_points)
    distance = squareform(distance)
    n, [point1_idx, point2_idx] = nanmax(distance), unravel_index(argmax(distance), distance.shape)
    ostia_points = [vessel_points[point1_idx], vessel_points[point2_idx]]

    print('ostia points:',ostia_points)

    max_range = 17
    max_points = 100

    counter = 0
    error_count: int = 0
    for op in ostia_points:
        for k in range(max_range):
            if k==0:
                    center_x = op[0]
                    center_y = op[1]
                    center_z = op[2]

                    center_x_pixel = int(round(center_x))
                    center_y_pixel = int(round(center_y))
                    center_z_pixel = int(round(center_z))

                    target_point = np.array([center_x_pixel, center_y_pixel, center_z_pixel])
                    print("new center:", target_point)

                    min_dis = np.linalg.norm(target_point - op)
                    print('min dis:', min_dis)
                    curr_proximity = get_proximity(min_dis,cutoff_value=16)
                    print('proximity:', curr_proximity)
                    cut_size = 9

                    left_x = center_x_pixel - cut_size
                    right_x = center_x_pixel + cut_size
                    left_y = center_y_pixel - cut_size
                    right_y = center_y_pixel + cut_size
                    left_z = center_z_pixel - cut_size
                    right_z = center_z_pixel + cut_size

                    if (right_z+1)<len(re_spacing_img) and left_z>=0 and (right_y+1)<max_y and left_y>=0 and (right_x+1)<max_x  and left_x>=0 and curr_proximity>0:
                        new_src_arr = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                        for ind in range(left_z, right_z + 1):
                            src_temp = re_spacing_img[ind].copy()
                            new_src_arr[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
                        folder_path = './patch_data/ostia_patch/postive/' + 'gp_' + str(gap_size) + '/d' + str(i)
                        if not os.path.exists(folder_path):
                            os.makedirs(folder_path)
                        record_name = 'ostia_patch/postive/' + 'gp_' + str(gap_size) + '/d' + str(i) + '/' + 'd_' + str(
                            i) + '_' + 'x_' + str(center_x_pixel) + '_y_' + str(center_y_pixel) + '_z_' + str(center_z_pixel) + '.nii.gz'
                        print(record_name)
                        org_name = './patch_data/' + record_name
                        out = sitk.GetImageFromArray(new_src_arr)
                        sitk.WriteImage(out, org_name)

                        proximity_list.append(curr_proximity)
                        patch_name.append(record_name)
                        counter += 1
                        print('')
                    else:
                        print('out of bounder skip this block')
            else:
                x_list,y_list,z_list = get_shell(max_points,k)
                record_set = set()
                record_set.add((int(round(op[0])),int(round(op[1])),int(round(op[2]))))

                for m in range(len(x_list)):
                    # print('op:', op)
                    new_x = int(round(op[0] + x_list[m]))
                    new_y = int(round(op[1] + y_list[m]))
                    new_z = int(round(op[2] + z_list[m]))
                    check_temp = (new_x,new_y,new_z)
                    if check_temp not in record_set:
                        record_set.add(check_temp)
                        center_x_pixel = new_x
                        center_y_pixel = new_y
                        center_z_pixel = new_z

                        target_point = np.array([center_x_pixel, center_y_pixel, center_z_pixel])
                        print("new center:", target_point)

                        min_dis = np.linalg.norm(target_point - op)
                        print('min dis:', min_dis)
                        curr_proximity = get_proximity(min_dis,cutoff_value=16)
                        print('proximity:', curr_proximity)
                        cut_size = 9

                        left_x = center_x_pixel - cut_size
                        right_x = center_x_pixel + cut_size
                        left_y = center_y_pixel - cut_size
                        right_y = center_y_pixel + cut_size
                        left_z = center_z_pixel - cut_size
                        right_z = center_z_pixel + cut_size

                        if (right_z + 1) < len(re_spacing_img) and left_z >= 0 and (
                                right_y + 1) < max_y and left_y >= 0 and (right_x + 1) < max_x and left_x >= 0 and curr_proximity>0:
                            print('')
                            new_src_arr = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))
                            for ind in range(left_z, right_z + 1):
                                src_temp = re_spacing_img[ind].copy()
                                new_src_arr[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]

                            folder_path = './patch_data/ostia_patch/postive/' + 'gp_' + str(
                                gap_size) + '/d' + str(i)
                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)
                            record_name = 'ostia_patch/postive/' + 'gp_' + str(gap_size) + '/d' + str(
                                i) + '/' + 'd_' + str(
                                i) + '_' + 'x_' + str(center_x_pixel) + '_y_' + str(center_y_pixel) + '_z_' + str(
                                center_z_pixel) + '.nii.gz'
                            print(record_name)
                            org_name = './patch_data/' + record_name
                            out = sitk.GetImageFromArray(new_src_arr)
                            sitk.WriteImage(out, org_name)

                            proximity_list.append(curr_proximity)
                            patch_name.append(record_name)
                            counter += 1

                        else:
                            print('out of bounder skip this block')
    return patch_name, proximity_list, error_count

def create_patch_images(path_name,spacing_path,gap_size):
    datasets = sorted(glob(path_name + '*'))

    for dataset in datasets:
        ds_idx = int(dataset[-2:])
        patch_name, proximity_list, error_count = creat_data(path_name,spacing_path,gap_size,ds_idx)
        dataframe = pd.DataFrame(
            {'patch_name': patch_name, 'proximity': proximity_list})
        print(dataframe.head())
        csv_name = "./patch_data/ostia_patch/postive/"+ 'gp_' + str(gap_size)+'/'+'d'+str(ds_idx) + "_patch_info.csv"
        dataframe.to_csv(csv_name, index=False, columns=['patch_name', 'proximity'], sep=',')
        print("create patch info csv")
        print("down")

        if error_count > 0:
            # ???????????????????? ?????????????????????????????? ??????????
            error_count_message: str = "Error point count that out of image borders: " + str(error_count)
            print(error_count_message)
            info_file_path: str = "./patch_data/ostia_patch/postive/"+ 'gp_' + str(gap_size)+'/'+'d'+str(ds_idx) + '.txt'
            f = open(info_file_path, 'w')
            f.write(error_count_message)
            f.close()


config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.ini'))
path_name = config['PATH']['input_folder']

spacing_path = 'spacing_info.csv'
gap_size  = 1
create_patch_images(path_name,spacing_path,gap_size)