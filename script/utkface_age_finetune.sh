#!/bin/bash

lambdas=(0.0 0.1 0.3 0.5)
simclr_seed=(500 600 700 800)


for ((i = 0; i < ${#simclr_seed[@]}; ++i)); do
    for ((j = 0; j < ${#lambdas[@]}; ++j)); do
        seed=${simclr_seed[$i]}
        ld=${lambdas[$j]}
        echo "seed: $seed, ld: $ld"

        #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
        #    --seed $seed --lambda_offdiag $ld --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        #    --data_dir /home/pky/research_new/dataset

        #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
        #    --seed $seed --lambda_offdiag $ld --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        #    --data_dir /home/pky/research_new/dataset \
        #    --finetune --finetune_only_once --lr_simclr 0.0001 --lr_clf 0.0001

        #CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
        #    --seed $seed --lambda_offdiag $ld --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
        #    --data_dir /home/pky/research_new/dataset \
        #    --finetune

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
            --seed $seed --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 --mode_CL SimCLR \
            --data_dir /home/pky/research_new/dataset \
            --finetune --mode oversample \
            --oversample_pth expr/checkpoint/UTKFace_age_SimCLR_lambda_0.0_seed_$seed/wrong_idx.pth \
            --lambda_upweight 10 --lr_simclr 0.0001 --lr_clf 0.0001
    done
done


