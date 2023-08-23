#!/bin/bash

seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    echo "seed: $s"
    #CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup --bias_attr gender \
    #    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    #    --seed $s


    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup --bias_attr gender \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --seed $s \
        --mode oversample --lambda_upweight 8 \
        --oversample_pth "expr/checkpoint/celebA_makeup_SimCLR_lambda_0.01_seed_$s/wrong_idx.pth"
done
