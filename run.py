import argparse
import torch
import torch.backends.cudnn as cudnn
from training.SimCLRSolver import SimCLRSolver
from training.LinearEvalSolver import LinearEvalSolver
from training.OversampleSolver import OversampleSolver
from training.ERMSolver import ERMSolver
from utils import setup, save_config


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    args = setup(args)

    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.mode == 'oversample':
        solver = OversampleSolver(args)

        if args.phase == 'train':
            solver.train_with_oversampling()
        else:
            solver.evaluate()

    elif args.mode == 'SimCLR':
        solver = LinearEvalSolver(args)

        if args.phase == 'train':
            solver.train()
        else:
            solver.evaluate()

    elif args.mode == 'ERM':
        solver = ERMSolver(args)

        if args.phase == 'train':
            solver.train()
        else:
            solver.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')

    #parser.add_argument('--exp_name', default=None, help='additional exp tag. See setup()')

    parser.add_argument('--mode', default='SimCLR', choices=['SimCLR', 'oversample', 'ERM'],
                        help='Vanilla SimCLR / Oversample failed samples for debiased linear evaluation / Vanilla ERM')
    parser.add_argument('--oversample_pth', default=None, help='denoting which samples to be oversampled', type=str)
    parser.add_argument('--phase', default='train', choices=['train', 'test'], type=str)

    # data
    parser.add_argument('--data_dir', default='/home/user/research/dataset',
                        help='path to dataset')
    parser.add_argument('--data', default='UTKFace',
                        help='dataset name', choices=['UTKFace', 'celebA', 'bffhq', 'stl10mnist', 'imagenet'])
    parser.add_argument('--bias_attr', default='age', choices=['race', 'age', 'gender'],
                        type=str, help='For UTKFace')
    parser.add_argument('--target_attr', default='blonde', choices=['blonde', 'makeup', 'race', 'gender'],
                        type=str, help='For celebA')
    parser.add_argument('--bias_ratio', default=0.1,
                        type=float, help='For stl10mnist')
    parser.add_argument('--imagenetA_dir', default='/home/user/research/dataset/ImageNet-A')

    # arch
    parser.add_argument('--arch', default='resnet18',
                        choices=['resnet18', 'resnet50'])
    parser.add_argument('--simclr_dim', default=128, type=int,
                        help='feature dimension (default: 128)')

    # train
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--simclr_epochs', default=100, type=int, metavar='N',
                        help='number of simclr pretraining epochs to run')
    parser.add_argument('--ERM_epochs', default=30, type=int, metavar='N',
                        help='number of ERM epochs to run')
    parser.add_argument('--linear_iters', default=1000, type=int, metavar='N',
                        help='number of linear evaluation iterations to run')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_simclr', default=0.0003, type=float, help='lr for SimCLR pretraining')
    parser.add_argument('--lr_clf', default=0.0003, type=float, help='lr for SimCLR linear eval')
    parser.add_argument('--lr_ERM', default=0.001, type=float, help='lr for ERM')
    parser.add_argument('--lr_decay_offset', default=10, type=int, help='lr decay is not used in SimCLR')
    parser.add_argument('--lr_decay_gamma', default=0.1, type=float, help='For ERM')
    parser.add_argument('--lambda_offdiag', default=0.1, type=float, help='rank regularization')
    parser.add_argument('--lambda_upweight', default=5, type=float, help='oversampling bias-free samples')
    parser.add_argument('--lambda_list', default=None, nargs='+', type=float,
                        help='Lambda_offdiags for ensemble trick')
    parser.add_argument('--cutoff', default=0.75, type=float,
                        help='Threshold for pseudo bias label')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)

    # misc
    parser.add_argument('--seed', default=1004, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--num_workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--eval_every', default=300, type=int,
                        help='Evaluate every n iters')
    parser.add_argument('--save_every', default=20, type=int,
                        help='Save pretrained models every n epochs')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--log_dir', type=str, default='expr/log')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoint')

    ################
    # contrastives
    ################
    parser.add_argument("--lambda_simsiam", type=float, default=1,
                        help='SimSiam loss coefficient')
    parser.add_argument("--lambda_vicReg_pos", type=float, default=25.0,
                        help='Positive pair, invariance regularization loss coefficient')
    parser.add_argument("--lambda_vicReg_std", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--lambda_vicReg_cov", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument('--mode_CL', default='SimCLR', choices=['SimCLR', 'SimSiam', 'vicReg'])

    main()
