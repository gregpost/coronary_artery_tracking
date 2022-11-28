# -*- coding: UTF-8 -*-
# @Time    : 14/05/2020 17:56
# @Author  : BubblyYi
# @FileName: seeds_train_tools.py
# @Software: PyCharm
import os
import sys
sys.path.append('..')
from models.seedspoints_net import SeedspointsNet
from seeds_net_data_provider_aug import DataGenerater
from seeds_trainner import Trainer
import torch

def get_dataset(save_num = 0):
    # Replace these paths to the path where you store the data
    train_data_info_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/seeds_patch/train_save_d"+str(save_num)+"_train.csv"
    train_pre_fix_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/seeds_patch"
    train_flag = 'train'
    train_transforms = None
    target_transform = None
    train_dataset = DataGenerater(train_data_info_path, train_pre_fix_path, train_transforms, train_flag, target_transform)

    val_data_info_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/seeds_patch/train_save_d"+str(save_num)+"_val.csv"
    val_pre_fix_path = "/Coronary-Artery-Tracking-via-3D-CNN-Classification/data_process_tools/patch_data/seeds_patch"
    val_flag = 'val'
    test_valid_transforms = None
    target_transform = None
    val_dataset = DataGenerater(val_data_info_path, val_pre_fix_path, test_valid_transforms, val_flag, target_transform)

    return train_dataset, val_dataset


def get_ds(train_df_path, train_prefix_path, val_df_path, val_prefix_path):
    train_flag = 'train'
    train_transforms = None
    target_transform = None
    train_ds = DataGenerater(train_df_path, train_prefix_path, train_transforms, train_flag, target_transform)

    val_flag = 'val'
    test_valid_transforms = None
    target_transform = None
    val_ds = DataGenerater(val_df_path, val_prefix_path, test_valid_transforms, val_flag, target_transform)

    return train_ds, val_ds


if __name__ == '__main__':
    data_path = os.path.join('..', 'data_process_tools', 'patch_data', 'seeds_patch')

    # Here we use 8 fold cross validation, save_num means to use dataset0x as the validation set
    save_num = 1
    # train_dataset, val_dataset = get_dataset(save_num)

    train_df_path = os.path.join(data_path, 'train.csv')
    train_img_prefix = os.path.join(data_path, 'train')

    val_df_path = os.path.join(data_path, 'val.csv')
    val_img_prefix = os.path.join(data_path, 'val')

    train_dataset, val_dataset = get_ds(train_df_path, train_img_prefix, val_df_path, val_img_prefix)

    curr_model_name = "seedspoints_net"
    model = SeedspointsNet()

    inital_lr = 0.001
    start_epoch = 0
    save_fold = "../checkpoint/seeds_checkpoints/"
    load_last_weights = True
    curr_loss = sys.float_info.max

    if start_epoch != 0:
        print('Loading saved model...')
        model_checkpoint_path = save_fold + "/" + curr_model_name + "_model_s" + str(save_num) + ".pkl"
        seeds_checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(seeds_checkpoint['net_dict'])

        inital_lr = seeds_checkpoint['initial_lr']
        curr_loss = seeds_checkpoint['curr_loss']
    elif load_last_weights:
        print('Loading last weights...')
        model_checkpoint_path = save_fold + "/" + curr_model_name + "_model_s" + str(save_num) + ".pkl"
        seeds_checkpoint = torch.load(model_checkpoint_path)
        model.load_state_dict(seeds_checkpoint['net_dict'])

    batch_size = 1024
    num_workers = os.cpu_count()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=inital_lr,weight_decay=0.001)

    trainer = Trainer(batch_size,
                      num_workers,
                      train_dataset,
                      val_dataset,
                      model,
                      curr_model_name,
                      optimizer,
                      criterion,
                      start_epoch=start_epoch,
                      max_epoch=100,
                      save_num=save_num,
                      best_test_loss=curr_loss,
                      initial_lr=inital_lr)

    trainer.run_train()
