#!/bin/bash

simclr_seed=(51 61 71 81)

for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    seed=${simclr_seed[$i]}

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
        --seed $seed --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        --save_every 5 --wd 1.

    #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    #    --seed $seed --lambda_offdiag 0. --simclr_epochs 5 --linear_iters 3000 --mode_CL SimCLR

    #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    #    --seed $seed --lambda_offdiag 0. --simclr_epochs 10 --linear_iters 3000 --mode_CL SimCLR

    #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    #    --seed $seed --lambda_offdiag 0. --simclr_epochs 15 --linear_iters 3000 --mode_CL SimCLR
done

