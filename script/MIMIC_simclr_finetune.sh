#!/bin/bash

#seed=(14 13 12 11)
seed=(11 12 13 14)


for s in ${seed[@]}; do
    echo "seed: $s"
    : '
    CUDA_VISIBLE_DEVICES=$1 python run.py --data MIMIC_NIH --mode_CL SimCLR \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 100 --linear_iters 3000 \
        --data_dir /home/user/hdisk \
        --seed $s --finetune

    CUDA_VISIBLE_DEVICES=$1 python run.py --data MIMIC_NIH --mode_CL SimCLR \
        --lambda_offdiag 0.005 --batch_size 128 --simclr_epochs 100 --linear_iters 3000 \
        --data_dir /home/user/hdisk \
        --seed $s --finetune --temperature 0.2 --exp_name "temp_0.2"
    '
    CUDA_VISIBLE_DEVICES=$1 python run.py --data MIMIC_NIH --mode_CL SimCLR \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 100 --linear_iters 3000 \
        --data_dir /home/user/hdisk \
        --seed $s --finetune \
        --mode oversample --lambda_upweight 5 \
        --oversample_pth "expr/checkpoint/MIMIC_NIH_temp_0.2__SimCLR_lambda_0.005_seed_"$s"/wrong_idx.pth" \
        --lr_clf 0.0001 --lr_simclr 0.0001 \
        --optimizer SGD --wd 0.0005
done


#CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
#    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 \
#    --mode oversample --lambda_list 0. 0.01 0.02 0.03 0.04 0.05 --cutoff 0.68 --lambda_upweight 8
