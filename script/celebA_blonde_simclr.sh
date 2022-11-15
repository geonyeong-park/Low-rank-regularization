#!/bin/bash


main_seed=(104 103 102 101)
wrong_seed=(1004 1003 1002 1001)


for ((i = 0; i < ${#main_seed[@]}; ++i)); do
#for s in ${main_seed[@]}; do
    s=${main_seed[$i]}
    ws=${wrong_seed[$i]}
    echo "seed: $s"
    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr blonde --bias_attr gender --mode_CL SimCLR \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --arch resnet50

    #CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr blonde --bias_attr gender --mode_CL SimCLR \
    #    --lambda_offdiag 0.03 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    #    --data_dir /home/pky/research_new/dataset \
    #    --seed $s

    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr blonde --bias_attr gender --mode_CL SimCLR \
        --lambda_offdiag 0.0 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --data_dir /home/pky/research_new/dataset \
        --seed $s --mode oversample --oversample_pth "expr/checkpoint/celebA_blonde_SimCLR_lambda_0.03_seed_"$ws"/wrong_idx.pth" \
        --lambda_upweight 15 --arch resnet50
done

