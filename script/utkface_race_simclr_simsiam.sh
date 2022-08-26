#!/bin/bash

simsiam_seed=(222 333 444 555)
simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#simsiam_seed[@]}; ++i)); do
    main_s=${simclr_seed[$i]}
    weight_s=${simsiam_seed[$i]}


    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
        --seed $main_s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        --mode oversample --lambda_upweight 4 \
        --oversample_pth "expr/checkpoint/UTKFace_gender_SimSiam_lambda_0.002_seed_"$weight_s"/wrong_idx.pth"
done


#CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
#    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 \
#    --mode oversample --lambda_list 0. 0.01 0.02 0.03 0.04 0.05 --cutoff 0.68 --lambda_upweight 8
