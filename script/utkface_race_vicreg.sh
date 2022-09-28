#!/bin/bash

seed=(1004 1005 1006 1007)
#seed=(1006 1007)
#lambdas=(0.0 0.1 0.2 0.3 0.5)
lambdas=(0.05)


for s in ${seed[@]}; do
    for ld in ${lambdas[@]}; do
        echo "lambda_offdiag: $ld"

        #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race --mode_CL vicReg \
        #    --data_dir /home/pky/research_new/dataset \
        #    --seed $s --lambda_offdiag $ld --simclr_epochs 100 --linear_iters 3000 \

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race --mode_CL vicReg \
            --seed $s --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
            --mode oversample --lambda_upweight 5 \
            --data_dir /home/pky/research_new/dataset \
            --oversample_pth "expr/checkpoint/UTKFace_gender_vicReg_lambda_"$ld"_seed_"$s"/wrong_idx.pth"
    done
done


#CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
#    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 \
#    --mode oversample --lambda_list 0. 0.01 0.02 0.03 0.04 0.05 --cutoff 0.68 --lambda_upweight 8
