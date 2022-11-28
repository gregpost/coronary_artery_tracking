import os
from glob import glob
import SimpleITK as sitk
import pandas as pd
from mps_reader import SavePointListAsMPS

root_ds = 'train_data'
save_dir = 'seed_points_mps'
patch_dirs = 'patch_data/seeds_patch/positive/gp_100'
ds_dfs = sorted(glob(os.path.join(patch_dirs, 'd*_patch_info.csv')))
ds_points = {}

for ds_df in ds_dfs:
    df = pd.read_csv(ds_df)
    patch_names = df['patch_name']

    points = []
    for patch_name in patch_names:
        _, ds_idx, _, vessel_idx, _, x, _, y, _, z = os.path.basename(patch_name).split('.')[0].split('_')
        points.append([float(x), float(y), float(z)])
    ds_points[ds_idx] = points

for ds_idx in ds_points:
    ds_number = f'0{ds_idx}' if int(ds_idx) < 10 else ds_idx
    image_path = os.path.join(root_ds, f'dataset{ds_number}', f'image{ds_number}')
    image = sitk.ReadImage(image_path)
    x_spacing, y_spacing, z_spacing = image.GetSpacing()
    SavePointListAsMPS([[z / z_spacing, y / y_spacing, x / x_spacing] for x, y, z in ds_points[ds_idx]], os.path.join(save_dir, f'seedpoints_d_{ds_idx}.mps'), image)

