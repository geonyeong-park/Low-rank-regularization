#!/bin/bash

simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    seed=${simclr_seed[$i]}
    echo "seed: $seed"

    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --bias_attr gender --target_attr makeup \
        --seed $seed --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 --mode_CL SimCLR \
        --data_dir /home/pky/research_new/dataset \
        --exp_name "tau0.07"
done

