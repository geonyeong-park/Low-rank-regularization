#!/bin/bash

seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0.5 --simclr_epochs 100 --linear_iters 3000

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
        --mode oversample --lambda_upweight 5 \
        --oversample_pth "expr/checkpoint/UTKFace_gender_SimCLR_lambda_0.5_seed_"$s"/wrong_idx.pth"
done

