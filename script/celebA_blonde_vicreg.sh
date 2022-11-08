#!/bin/bash


seed=(1004 1003 1002 1001)


for s in ${seed[@]}; do
    echo "seed: $s"
    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr blonde --bias_attr gender --mode_CL vicReg \
        --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
        --data_dir /home/pky/research_new/dataset \
        --seed $s
done

