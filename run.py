import argparse
import torch
import torch.backends.cudnn as cudnn
from training.SimCLRSolver import SimCLRSolver
from training.LinearEvalSolver import LinearEvalSolver
from training.OversampleSolver import OversampleSolver
from utils import setup


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda')
    args = setup(args)

    cudnn.deterministic = True
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    if args.oversample:
        solver = OversampleSolver(args)

        if args.phase == 'train':
            solver.train_with_oversampling()
        else:
            solver.evaluate()

    else:
        solver = LinearEvalSolver(args)

        if args.phase == 'train':
            solver.train()
        else:
            solver.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')

    parser.add_argument('--exp_name', default=None, help='additional exp tag. See setup()')
    parser.add_argument('--oversample', default=False, action='store_true',
                        help='Oversample failed samples for debiased linear evaluation')
    parser.add_argument('--oversample_pth', default=None, help='denoting which samples to be oversampled', type=str)
    parser.add_argument('--phase', default='train', choices=['train', 'test'], type=str)

    # data
    parser.add_argument('--data_dir', default='/home/user/research/dataset',
                        help='path to dataset')
    parser.add_argument('--data', default='UTKFace',
                        help='dataset name', choices=['UTKFace', 'celebA'])
    parser.add_argument('--bias_attr', default='race', choices=['race', 'age'],
                        type=str, help='For UTKFace')
    parser.add_argument('--target_attr', default='blonde', choices=['blonde', 'makeup'],
                        type=str, help='For celebA')

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
    parser.add_argument('--linear_epochs', default=30, type=int, metavar='N',
                        help='number of linear evaluation epochs to run')
    parser.add_argument('--batch_size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr_simclr', '--learning-rate', default=0.0003, type=float)
    parser.add_argument('--lambda_offdiag', default=0.1, type=float, help='rank regularization')
    parser.add_argument('--lambda_upweight', default=20, type=float, help='oversampling bias-free samples')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'], type=str)

    # misc
    parser.add_argument('--seed', default=7777, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--fp16-precision', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--num_workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--eval_every', default=5, type=int,
                        help='Evaluate every n epochs')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

    parser.add_argument('--log_dir', type=str, default='expr/log')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoint')

    main()
