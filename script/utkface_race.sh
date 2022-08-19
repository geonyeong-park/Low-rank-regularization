CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/UTKFace_gender_lambda_0.0_seed_$2/wrong_idx.pth"

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/UTKFace_gender_lambda_0.1_seed_$2/wrong_idx.pth"

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/UTKFace_gender_lambda_0.3_seed_$2/wrong_idx.pth"

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 8 \
    --oversample_pth "expr/checkpoint/UTKFace_gender_lambda_0.5_seed_$2/wrong_idx.pth"

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr gender --target_attr race \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 8 \
    --lambda_list 0. 0.1 0.3 0.5 1.0 --cutoff 0.47
