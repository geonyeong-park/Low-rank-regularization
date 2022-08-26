#!/bin/bash

lambdas=(0.0 0.01 0.02 0.008 0.006)


for ld in ${lambdas[@]}; do
    echo "lambda_offdiag: $ld"
    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup --bias_attr gender --mode_CL vicReg \
        --lambda_offdiag $ld --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --seed $2

    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup --bias_attr gender --mode_CL vicReg \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --seed $2 \
        --mode oversample --lambda_upweight 8 \
        --oversample_pth "expr/checkpoint/celebA_makeup_vicReg_lambda_"$ld"_seed_$2/wrong_idx.pth"
done


#CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
#    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 \
#    --mode oversample --lambda_list 0. 0.01 0.02 0.03 0.04 0.05 --cutoff 0.68 --lambda_upweight 8
