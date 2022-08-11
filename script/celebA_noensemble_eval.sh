#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/celebA_makeup_lambda_0.0_seed_$2/wrong_idx.pth"

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/celebA_makeup_lambda_0.01_seed_$2/wrong_idx.pth"
