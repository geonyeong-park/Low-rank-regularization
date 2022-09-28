#!/bin/bash

seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0.3 --simclr_epochs 100 --linear_iters 3000

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
        --mode oversample --lambda_upweight 10 \
        --oversample_pth "expr/checkpoint/UTKFace_age_SimCLR_lambda_0.3_seed_"$s"/wrong_idx.pth"
done

