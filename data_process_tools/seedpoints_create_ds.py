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
    neg_dir = os.path.join(data_path, 'negative')
    pos_dir = os.path.join(data_path, 'positive')

    pos_ps_dir = os.listdir(pos_dir)[0]
    neg_ps_dir = os.listdir(neg_dir)[0]

    pos_ps_path = os.path.join(pos_dir, pos_ps_dir)
    neg_ps_path = os.path.join(neg_dir, neg_ps_dir)

    pos_csvs = glob.glob(os.path.join(pos_ps_path, '*.csv'))
    neg_csvs = glob.glob(os.path.join(neg_ps_path, '*.csv'))

    pos_dfs = combine_dfs(pos_csvs)
    neg_dfs = combine_dfs(neg_csvs)

    dfs = pos_dfs + neg_dfs
    df = pd.concat(dfs, axis=0, ignore_index=True)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    return train_df, test_df


def move_images(df, image_dir, dest_dir):
    df = df.copy()
    filenames = []

    for index, row in df.iterrows():
        print(f'seedpoints index: {index}')
        filepath = os.path.join(image_dir, row['patch_name'])
        filename = os.path.basename(filepath)

        dest_path = os.path.join(dest_dir, filename)
        os.rename(filepath, dest_path)
        filenames.append(filename)

    df['patch_name'] = filenames
    return df


path_data = os.path.abspath(os.path.join('patch_data', 'seeds_patch'))
train_df, val_df = create_ds(path_data)

train_dir = os.path.join(path_data, 'train')
val_dir = os.path.join(path_data, 'val')

if not os.path.exists(train_dir):
    os.mkdir(train_dir)

if not os.path.exists(val_dir):
    os.mkdir(val_dir)

train_df = move_images(train_df, 'patch_data', train_dir)
val_df = move_images(val_df, 'patch_data', val_dir)

train_df_name = os.path.join(path_data, 'train.csv')
val_df_name = os.path.join(path_data, 'val.csv')

train_df.to_csv(train_df_name, index=False)
val_df.to_csv(val_df_name, index=False)
