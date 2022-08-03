#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0.01 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0.02 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0.03 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0.04 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0.05 --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2

CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup \
    --lambda_offdiag 0. --batch_size 128 --simclr_epochs 20 --linear_iters 5000 \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 \
    --mode oversample --lambda_list 0. 0.01 0.02 0.03 0.04 0.05 --cutoff 0.68 --lambda_upweight 8
