import torch
import SimpleITK as sitk
import torchio as tio
import numpy as np

image_path = '/home/skilpadd/Job/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/centerline_patch (copy)/no_offset/point_500_gp_1/d0/d_0_v_0_patch_2.nii.gz'
image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_path))

transforms = tio.Compose([
    tio.RandomSwap(),
    tio.RandomNoise(),
    tio.RandomAffine()
])

img = torch.from_numpy(image_array)
transformed = transforms(img.unsqueeze(0))
print(transformed)
print(type(transformed))