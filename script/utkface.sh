CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0.1 --simclr_epochs 100 --linear_iters 3000 

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0.3 --simclr_epochs 100 --linear_iters 3000 

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0.5 --simclr_epochs 100 --linear_iters 3000 

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 1. --simclr_epochs 100 --linear_iters 3000 

CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.0_seed_$2/wrong_idx.pth" 


CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.1_seed_$2/wrong_idx.pth" 


CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.3_seed_$2/wrong_idx.pth" 


CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.5_seed_$2/wrong_idx.pth" 


CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --lambda_list 0. 0.1 0.3 0.5 1.0 --cutoff 0.47 


