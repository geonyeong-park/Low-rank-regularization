#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python run.py --data stl10mnist --simclr_epochs 20 \
    --linear_iters 3000 --seed 111 --lambda_offdiag 0. \
    --batch_size 128 \
    --num_unique_mnist $2 --mode_CL $3

for ((i = 1; i < 21; ++i)); do
    echo "[STL10MNIST] pretrained epochs $i, number of unique mnist $2"
    echo "GPU: $1"

    CUDA_VISIBLE_DEVICES=$1 python run.py --data stl10mnist --simclr_epochs $i \
        --linear_iters 3000 --seed 111 --lambda_offdiag 0. \
        --batch_size 128 \
        --num_unique_mnist $2 --mode_CL $3

    #CUDA_VISIBLE_DEVICES=$1 python run.py --data stl10mnist --simclr_epochs 10 \
    #    --linear_iters 3000 --seed 7777 --lambda_offdiag 0. \
    #    --batch_size 128 \
    #    --bias_ratio $r --lambda_upweight $l \
    #    --mode oversample --oversample_pth "expr/checkpoint/stl10mnist__lambda_0.0_seed_7777/debias_idx_$r.pth"
done
