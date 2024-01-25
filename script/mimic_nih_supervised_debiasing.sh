#!/bin/bash

seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    CUDA_VISIBLE_DEVICES=$1 python run.py --data MIMIC_NIH --mode ERM \
        --data_dir /home/user/hdisk \
        --batch_size 128 --ERM_epochs 4 --seed $s --lambda_offdiag 10 --lr_ERM 3e-4
        #--oversample_pth "expr/checkpoint/UTKFace_age_SimCLR_lambda_0.3_seed_"$s"/wrong_idx.pth"
done

