import os
import copy
import torch
import numpy as np
import pandas as pd
from setting import re_spacing_img, resize_factor, spacing, device, infer_model, max_points, setting_info
from utils import get_angle, data_preprocess, get_spacing_res2, prob_terminates, get_shell

seed_points = pd.read_csv(os.path.join(setting_info["seeds_gen_info_to_save"], "seeds.csv"))


def infer(start: list):
    """
    :param start: Initial point
    :return: Moving position, the index of maximum confidence direction, Current termination probability
    """
    max_z = re_spacing_img.shape[0]
    max_x = re_spacing_img.shape[1]
    max_y = re_spacing_img.shape[2]

    cut_size = 9
    spacing_x = spacing[0]
    spacing_y = spacing[1]
    spacing_z = spacing[2]

    center_x_pixel = get_spacing_res2(start[0], spacing_x, resize_factor[1])
    center_y_pixel = get_spacing_res2(start[1], spacing_y, resize_factor[2])
    center_z_pixel = get_spacing_res2(start[2], spacing_z, resize_factor[0])

    left_x = center_x_pixel - cut_size
    right_x = center_x_pixel + cut_size
    left_y = center_y_pixel - cut_size
    right_y = center_y_pixel + cut_size
    left_z = center_z_pixel - cut_size
    right_z = center_z_pixel + cut_size

    new_patch = np.zeros((cut_size * 2 + 1, cut_size * 2 + 1, cut_size * 2 + 1))

    if not (
            left_x < 0 or right_x < 0 or left_y < 0 or right_y < 0 or left_z < 0 or right_z < 0 or left_x >= max_x or right_x >= max_x or left_y >= max_y or right_y >= max_y or left_z >= max_z or right_z >= max_z):
        for ind in range(left_z, right_z + 1):
            src_temp = re_spacing_img[ind].copy()
            new_patch[ind - left_z] = src_temp[left_y:right_y + 1, left_x:right_x + 1]
        input_data = data_preprocess(new_patch)

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
    else:
        return None


def search_first_node(start: list, prob_records: list):
    """
    :param start: Initial point
    :return: Next direction vector, Probability record, Current radius
    """
    s_all, indexs, curr_r, curr_prob = infer(start=start)
    start_x, start_y, start_z = start
    prob_records.pop(0)
    prob_records.append(curr_prob)
    sx, sy, sz = s_all
    forward_x = sx[indexs[0]] + start_x
    forward_y = sy[indexs[0]] + start_y
    forward_z = sz[indexs[0]] + start_z
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
    backward_x = backward_move_direction_x + start_x
    backward_y = backward_move_direction_y + start_y
    backward_z = backward_move_direction_z + start_z
    direction = {}
    direction["forward"] = [forward_x, forward_y, forward_z]
    direction["forward_vector"] = [forward_move_direction_x, forward_move_direction_y, forward_move_direction_z]
    direction["backward"] = [backward_x, backward_y, backward_z]
    direction["backward_vector"] = [backward_move_direction_x, backward_move_direction_y, backward_move_direction_z]
    return direction, prob_records, curr_r


def build_vessel_tree(seeds: np.ndarray):
    prob_records = [0] * 3
    seeds_unused = []

    for seed in seeds:
        direction, prob_records, curr_r = search_first_node(start=seed, prob_records=prob_records)
