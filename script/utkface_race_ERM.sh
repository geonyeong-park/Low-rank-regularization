#!/bin/bash

seed=(111 222 333 444)


for s in ${seed[@]}; do
    echo "[ERM] Seed: $s"
    CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --target_attr race --bias_attr gender --mode ERM \
        --data_dir /home/pky/research_new/dataset \
        --ERM_epochs 20 --seed $s
done
