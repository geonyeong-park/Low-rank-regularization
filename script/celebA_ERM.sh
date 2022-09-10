#!/bin/bash

seed=(111 222 333 444)


for s in ${seed[@]}; do
    echo "[ERM] Seed: $s"
    CUDA_VISIBLE_DEVICES=$1 python run.py --data celebA --target_attr makeup --bias_attr gender --mode ERM \
        --data_dir /home/pky/research_new/dataset \
        --batch_size 128 --ERM_epochs 20 --seed $s
done
