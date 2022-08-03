#!/bin/bash

ratio=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ld=(1. 1.25 1.33 1.66 2. 2.5 3.33 5. 10.)


for ((i = 0; i < ${#ratio[@]}; ++i)); do
    r=${ratio[$i]}
    l=${ld[$i]}

    echo "[STL10MNIST] ratio $r, lambda_upweight $l"
    echo "GPU: $1"

    CUDA_VISIBLE_DEVICES=$1 python run.py --data stl10mnist --simclr_epochs 10 \
        --linear_iters 3000 --seed 7777 --lambda_offdiag 0. \
        --batch_size 128 \
        --bias_ratio $r
    CUDA_VISIBLE_DEVICES=$1 python run.py --data stl10mnist --simclr_epochs 10 \
        --linear_iters 3000 --seed 7777 --lambda_offdiag 0. \
        --batch_size 128 \
        --bias_ratio $r --lambda_upweight $l \
        --mode oversample --oversample_pth "expr/checkpoint/stl10mnist__lambda_0.0_seed_7777/debias_idx_$r.pth"
done
