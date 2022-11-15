#!/bin/bash

seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --bias_attr gender --target_attr makeup \
        --seed $seed --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --exp_name "shallow" --arch conv

    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --bias_attr gender --target_attr makeup \
        --seed $seed --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset

    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --bias_attr gender --target_attr makeup \
        --seed $seed --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --mode oversample --oversample_pth "expr/checkpoint/celebA_shallow_makeup_SimCLR_lambda_0.0_seed_"$s"/wrong_idx.pth" \
        --lambda_upweight 8
done

