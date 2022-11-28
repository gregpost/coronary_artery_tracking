from xml.dom import minidom
import copy
import torch
import numpy as np
import SimpleITK as sitk
from setting import infer_model, device, spacing, re_spacing_img, resize_factor, max_points, prob_thr
from utils import get_spacing_res2, data_preprocess, prob_terminates, get_shell, get_angle
from setting import setting_info
from typing import List
from mps_reader import AddPointNodes


def SavePointListAsMPS(_points: List[List[int]], _filename: str,  _image: sitk.Image):
    """
    Сохраняет список точек в координатах NumPy в виде .mps файла.

    :param _points: Список точек в NumPy координатах.
    :param _filename: Имя .mps файла, который будет создан.
    :param _image: Исходное изображение.
    """

    document = minidom.Document()
    point_set_file: minidom.Element = document.createElement('point_set_file')
    document.appendChild(point_set_file)

    file_version: minidom.Element = document.createElement('file_version')
    file_version_value = document.createTextNode('0.1')
    file_version.appendChild(file_version_value)
    point_set_file.appendChild(file_version)

    point_set: minidom.Element = document.createElement('point_set')
    point_set_file.appendChild(point_set)

    time_series: minidom.Element = document.createElement('time_series')
    point_set.appendChild(time_series)

    time_series_id: minidom.Element = document.createElement('time_series_id')
    time_series_id_value = document.createTextNode('0')
    time_series_id.appendChild(time_series_id_value)
    time_series.appendChild(time_series_id)

    AddPointNodes(_points, time_series, document)

    with open(_filename, 'w') as file:
        document_str = document.toprettyxml()
        file.write(document_str)


def infer(patch: np.ndarray):
    """
    :param start: Initial point
    :return: Moving position, the index of maximum confidence direction, Current termination probability
    """
    input_data = data_preprocess(patch)

    inputs = input_data.to(device)
    outputs = infer_model(inputs.float())

    outputs = outputs.view((len(input_data), max_points + 1))
    outputs_1 = outputs[:, :len(outputs[0]) - 1]
    outputs_2 = outputs[:, -1]

    outputs_1 = torch.nn.functional.softmax(outputs_1, 1)
    indexs = np.argsort(outputs_1.cpu().detach().numpy()[0])[::-1]
    curr_prob = prob_terminates(outputs_1, max_points).cpu().detach().numpy()[0]
    curr_r = outputs_2.cpu().detach().numpy()[0]
    sx, sy, sz = get_shell(max_points, curr_r)
    return [sx, sy, sz], indexs, curr_r, curr_prob


cube_path = '/home/skilpadd/Job/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/centerline_patch/val/d_0_v_1_patch_80_3.nii.gz'
cube_image = sitk.ReadImage(cube_path)
cube_array = sitk.GetArrayFromImage(cube_image)
center = np.array(cube_array.shape) // 2

s_all, indexs, curr_r, curr_prob = infer(cube_array)
sx, sy, sz = s_all

forward_x = sx[indexs[0]] + center[0]
forward_y = sy[indexs[0]] + center[1]
forward_z = sz[indexs[0]] + center[2]

forward_move_direction_x = sx[indexs[0]]
forward_move_direction_y = sy[indexs[0]]
forward_move_direction_z = sz[indexs[0]]

for i in range(1, len(indexs)):
    curr_angle = get_angle(np.array([sx[indexs[i]], sy[indexs[i]], sz[indexs[i]]]),
                           np.array([forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]))
    # To determine two initial opposing directions of the tracker, two local maxima d0 and d0′ separated by an angle ≥ 90°
    if curr_angle >= 90:
        backward_move_direction_x = copy.deepcopy(sx[indexs[i]])
        backward_move_direction_y = copy.deepcopy(sy[indexs[i]])
        backward_move_direction_z = copy.deepcopy(sz[indexs[i]])
        break

backward_x = backward_move_direction_x + center[0]
backward_y = backward_move_direction_y + center[1]
backward_z = backward_move_direction_z + center[2]

forward_point = np.array([forward_x, forward_y, forward_z])
backward_point = np.array([backward_x, backward_y, backward_z])

print('forward', forward_point)
print('backward', backward_point)

SavePointListAsMPS([forward_point], 'forward_point_patch_80_3.mps', cube_image)
SavePointListAsMPS([backward_point], 'backward_point_patch_80_3.mps', cube_image)
