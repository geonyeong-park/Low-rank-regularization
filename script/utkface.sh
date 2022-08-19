#CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 

#CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age \
#    --data_dir /home/pky/research_new/dataset \
#    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
#    --mode oversample --lambda_upweight 10 \
#    --oversample_pth "expr/checkpoint/UTKFace_age_lambda_0.0_seed_$2/wrong_idx.pth" 


CUDA_VISIBLE_DEVICES=$1 python run.py --data UTKFace --bias_attr age --target_attr gender \
    --data_dir /home/pky/research_new/dataset \
    --seed $2 --lambda_offdiag 0. --simclr_epochs 100 --linear_iters 3000 \
    --mode oversample --lambda_upweight 10 \
    --lambda_list 0. 0.1 0.3 --cutoff 0.6


