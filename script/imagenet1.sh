#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0. --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.01 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.03 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777


CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.05 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.1 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.3 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777

CUDA_VISIBLE_DEVICES=$1 python run.py --lambda_offdiag 0.5 --data imagenet --simclr_epochs 100 \
    --linear_iters 10000 --batch_size 128 --seed 7777
