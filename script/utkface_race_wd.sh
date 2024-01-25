#!/bin/bash

#weight_decay=(0.0001 0.0005 0.001 0.005 0.01 0.1)
weight_decay=(0.05 0.5)
simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    for ((j = 0; j < ${#weight_decay[@]}; ++j)); do
        seed=${simclr_seed[$i]}
        wd=${weight_decay[$j]}
        echo "seed: $seed, tau: $wd"

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
            --seed $seed --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
            --exp_name "wd$wd" --wd $wd
    done
done


