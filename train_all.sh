#! /bin/bash
cd centerline_train_tools/
CUDA_VISIBLE_DEVICES=0 /home/kmt/hdd/coronary_artery_tracking/venv/bin/python3 centerline_train_tools.py
cd ..
cd seedspoints_train_tools/
CUDA_VISIBLE_DEVICES=0 /home/kmt/hdd/coronary_artery_tracking/venv/bin/python3 seeds_train_tools.py
cd ..
cd ostiapoints_train_tools
CUDA_VISIBLE_DEVICES=0 /home/kmt/hdd/coronary_artery_tracking/venv/bin/python3 ostia_train_tools.py