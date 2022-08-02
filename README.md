# Low-rank-regularization

## 1. SimCLR + Linear evaluation
```
python run.py --lambda_offdiag $ld. --data UTKFace --bias_attr age \
--simclr_epochs 100 --linear_iters 3000 --seed 1004
```
- `$ld` == 0: Vanilla SimCLR
- `$ld` > 0: Pretrain encoder with rank regularization
  - 2-1. Debiased linear evaluation w/o ensemble trick: run the above cmd with '$ld'=0., 0.1
  - 2-2. Debiased linear evaluation w/ ensemble trick: For ensemble tricks, please run the above cmd with '$ld'=0., 0.1, 0.3, 0.5, 1.0


## 2. SimCLR + Debiased linear evaluation

## 2-1. Debiased linear evaluation w/o ensemble trick
```
python run.py --lambda_offdiag 0. --data UTKFace --bias_attr age \
--simclr_epochs 100 --linear_iters 3000 --seed 1004 \
--mode oversample --oversample_pth expr/checkpoint/UTKFace_age_lambda_0.1_seed_1004/wrong_idx.pth \
--lambda_upweight 10
```
- ```oversample_pth``` indicates a correct/incorrect classification results of training samples. 
Incorrect samples will be oversampled as ```lambda_upweight``` during linear evaluation.

## 2-2. Debiased linear evaluation w/ ensemble trick
```
python run.py --lambda_offdiag 0. --data UTKFace --bias_attr age \
--simclr_epochs 100 --linear_iters 3000 --seed 1004 \
--mode oversample --lambda_list 0. 0.1 0.3 0.5 1.0 --cutoff 0.47 \
--lambda_upweight 10
```
- ```cutoff``` indicates a threshold value to determine the incorrect samples to be oversampled.
