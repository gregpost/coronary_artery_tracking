import numpy as np
import SimpleITK as sitk
from setting import src_array, spacing, seeds_model, ostia_model, device, setting_info
from utils import data_preprocess, resample, crop_heart
from mps_reader import SavePointListAsMPS, GetPointListFromMPS
import torch
from debug_manager import DebugManager

cube_path = '/home/skilpadd/Job/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/seeds_patch (copy)/positive/gp_100/d0/d_0_v_1_x_93_y_99_z_99.nii.gz'
cube_image = sitk.ReadImage(cube_path)
cube_array = sitk.GetArrayFromImage(cube_image)

input_data = data_preprocess(cube_array)
inputs = input_data.to(device)
seeds_outputs = seeds_model(inputs.float())
seeds_outputs = seeds_outputs.view((len(input_data)))  # view
seeds_proximity = seeds_outputs.cpu().detach().numpy()
print(seeds_proximity[0])

# d_0_v_0_x_60_y_59_z_90.nii.gz --> [68.25521]
# d_0_v_1_x_91_y_97_z_102.nii.gz --> [65.865616]
# d_0_x_28_y_9_z_104.nii.gz --> -0.6217545
