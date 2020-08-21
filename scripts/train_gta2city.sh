#!/usr/bin/bash
set -ex
ls
export PYTHONPATH=`pwd`
DLS_DATA_URL='/share/data/'
DLS_TRAIN_TRAIN='./snapshots/'
echo ${DLS_DATA_URL}
CUDA_VISIBLE_DEVICES=2 python train_gta2city.py \
            --method='GTA5KLASA' \
            --backbone='resnet'\
            --data-dir=${DLS_DATA_URL}gta5 \
            --data-list='./datasets/gta5_list/train.txt' \
            --data-dir-target=${DLS_DATA_URL}cityscapes \
            --data-list-target='./datasets/cityscapes_list/train.txt' \
            --snapshot-dir=${DLS_TRAIN_TRAIN} \
            --resume='pretrained/GTA5100k_25000.pth' \
            --batch-size=1 \
            --num-steps=150000 \
            --num-steps-stop=150000 \
            --lambda-adv-target1=0.001 \
