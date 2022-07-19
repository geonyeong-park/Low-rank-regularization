#!/bin/bash

temperature=(0.05 0.07 0.08 0.09 1.0)
lambd=(0. 0.1 0.2 0.3 0.4 0.5)


for t in ${temperature[@]}; do
    for l in ${lambd[@]}; do
        echo "[Ours] temperature $t, lambda_offdiag $l"
        echo "GPU: $1"
        echo "Data: UTKFace"
        echo "Bias attribution: $2"

        CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr $2 \
            --seed 6666 --lambda_offdiag $l --temperature $t
    done
done
