#!/bin/bash

vicreg_seed=(700 710 720 730)
simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#vicreg_seed[@]}; ++i)); do
    main_s=${simclr_seed[$i]}
    weight_s=${vicreg_seed[$i]}

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
        --seed $main_s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        --mode oversample --lambda_upweight 10 \
        --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.0_seed_$weight_s/wrong_idx.pth"

    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
        --seed $main_s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        --mode oversample --lambda_upweight 10 \
        --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.1_seed_$weight_s/wrong_idx.pth"
done

