#!/bin/bash

#temp=(0.01 0.07 0.1 0.5 1. 1.5 5.)
temp=(1.5 5.)
simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    for ((j = 0; j < ${#temp[@]}; ++j)); do
        seed=${simclr_seed[$i]}
        tau=${temp[$j]}
        echo "seed: $seed, tau: $tau"

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
            --seed $seed --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
            --data_dir /home/pky/research_new/dataset \
            --exp_name "tau$tau" --temperature $tau --phase test

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender --mode_CL SimCLR \
            --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
            --mode oversample --lambda_upweight 10 --exp_name "tau0.07" \
            --data_dir /home/pky/research_new/dataset \
            --oversample_pth "expr/checkpoint/UTKFace_tau5._age_SimCLR_lambda_0.0_seed_$s/wrong_idx.pth"


    done
done


