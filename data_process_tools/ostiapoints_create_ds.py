import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split


def combine_dfs(folder):
    """
    Возвращает список из csv файлов директории
    :param folder:
    :return: dfs
    """
    dfs = []
    for file in folder:
        df = pd.read_csv(os.path.abspath(file))
        dfs.append(df)

    return dfs


def create_ds(data_path):
    offset_dir = os.path.join(data_path, 'postive')
    no_offset_dir = os.path.join(data_path, 'negative')

    offset_path = os.path.join(data_path, offset_dir)
    no_offset_path = os.path.join(data_path, no_offset_dir)

    offset_ps_dir = os.listdir(offset_path)[0]
    no_offset_ps_dir = os.listdir(no_offset_path)[0]

    offset_ps_path = os.path.join(offset_path, offset_ps_dir)
    no_offset_ps_path = os.path.join(no_offset_path, no_offset_ps_dir)

    offset_csvs = glob.glob(os.path.join(offset_ps_path, '*.csv'))
    no_offset_csvs = glob.glob(os.path.join(no_offset_ps_path, '*.csv'))

    offset_dfs = combine_dfs(offset_csvs)
    no_offset_dfs = combine_dfs(no_offset_csvs)

    dfs = offset_dfs + no_offset_dfs
    df = pd.concat(dfs, axis=0, ignore_index=True)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    return train_df, test_df


def move_images(df, image_dir, dest_dir):
    df = df.copy()
    filenames = []

    for index, row in df.iterrows():
        print(f'ostiapoints index: {index}')
        filepath = os.path.join(image_dir, row['patch_name'])
        filename = os.path.basename(filepath)

        dest_path = os.path.join(dest_dir, filename)
        os.rename(filepath, dest_path)
        filenames.append(filename)

    df['patch_name'] = filenames
    return df


path_data = os.path.abspath(os.path.join('patch_data', 'ostia_patch'))
train_df, val_df = create_ds(path_data)

train_dir = os.path.join(path_data, 'train')
val_dir = os.path.join(path_data, 'val')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

train_df = move_images(train_df, 'patch_data', train_dir)
train_df = train_df[train_df['patch_name'] != '']

val_df = move_images(val_df, 'patch_data', val_dir)
val_df = val_df[val_df['patch_name'] != '']

train_df_name = os.path.join(path_data, 'train.csv')
val_df_name = os.path.join(path_data, 'val.csv')

train_df.to_csv(train_df_name, index=False)
val_df.to_csv(val_df_name, index=False)
