#!/bin/bash

simclr_seed=(500 600 700 800)
weight_decay=(0.0001 0.0005 0.001 0.005 0.01 0.05)
# 0.005부터 collapse

for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    for ((j = 0; j < ${#weight_decay[@]}; ++j)); do
        seed=${simclr_seed[$i]}
        wd=${weight_decay[$j]}

        CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --bias_attr gender --target_attr makeup \
            --seed $seed --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 --mode_CL SimCLR \
            --wd $wd --exp_name "wd$wd"
    done
done
