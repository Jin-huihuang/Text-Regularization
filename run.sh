#!/bin/bash

data_dir=/data/

# PACS
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS --dataset PACS --deterministic \
--trial_seed 0 --checkpoint_freq 100 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.1 --lr 5e-5

# VLCS
CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS --dataset VLCS --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.01 --lr 1e-6

# OfficeHome
CUDA_VISIBLE_DEVICES=0 python train_all.py OH --dataset OfficeHome --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 1 --lr 5e-5

# TerraIncognita
CUDA_VISIBLE_DEVICES=0 python train_all.py TR0 --dataset TerraIncognita --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm Contrast \
--contrast_w 1 --lr 5e-5

# DomainNet
CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet --dataset DomainNet --deterministic \
--trial_seed 0 --checkpoint_freq 1000 --steps 15001 --data_dir $data_dir --algorithm Contrast \
--contrast_w 0.1 --lr 5e-5