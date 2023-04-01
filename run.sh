#!/bin/bash

data_dir=/data/

# PACS
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS --dataset PACS --deterministic \
--trial_seed 0 --checkpoint_freq 100 --steps 5000 --data_dir $data_dir --use_buffers True \
--algorithm Contrast --cls_w 1 --contrast_w 0.1 --lr 5e-5 --resnet_dropout 0 --weight_decay 0 \
--Text RN50 --Linear_cls True --epsilon 0.01

# VLCS
CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS --dataset VLCS --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir /data/ --use_buffers True \
--algorithm Contrast --cls_w 1 --contrast_w 0.01 --lr 1e-6 --resnet_dropout 0.4 --weight_decay 0 \
--Text RN50 --Linear_cls True --epsilon 0.1

# 
