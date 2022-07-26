#!/bin/bash

epoch=(20 40 60 80 100)
lr=(0.0003 0.0006 0.001)


for e in ${epoch[@]}; do
    for l in ${lr[@]}; do
        echo "[Ours] epoch $e, lr $l"
        echo "GPU: $1"
        echo "Data: UTKFace"
        echo "Bias attribution: $2"

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr $2 \
            --seed 7777 --lambda_offdiag 0.1 --lr_simclr $l --simclr_epochs $e
    done
done
