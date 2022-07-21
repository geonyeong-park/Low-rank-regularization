from torch.utils.tensorboard import SummaryWriter
from os.path import join as ospj
import os

import torch
from data_aug.data_loader import get_original_loader, get_val_loader, InputFetcher
from training.LinearEvalSolver import LinearEvalSolver


class OversampleSolver(LinearEvalSolver):
    def __init__(self, args):
        super(OversampleSolver, self).__init__(args)
        self.writer = SummaryWriter(ospj(args.log_dir, 'debiased_eval'))

    def train_with_oversampling(self):
        if self.args.oversample_pth is not None:
            pth = self.args.oversample_pth
            if not os.path.exists(pth):
                raise ValueError(f'{pth} does not exists')
        else:
            pth = ospj(self.args.checkpoint_dir, 'wrong_index.pth')

        try:
            self._load_checkpoint(self.args.simclr_epochs, 'biased_simclr')
            assert os.path.exists(pth)
            print('Pretrained SimCLR ckpt exists. Move onto linear evaluation')
        except:
            print('WARNING: Either pretrained model or wrong_index.pth does not exists')
            self.train()
            print('Now you have both pretrained model and wrong_index.pth. \
                  Run this code again.')
            return

        wrong_label = torch.load(pth)
        print(f'Number of wrong/total samples: {wrong_label.sum()}/{wrong_label.size(0)}')
        upweight = torch.ones_like(wrong_label)
        upweight[wrong_label == 1] = self.args.lambda_upweight
        upweight_loader = get_original_loader(self.args, sampling_weight=upweight, simclr_aug=False)

        self.linear_evaluation(upweight_loader, token='debiased_linear')

    def evaluate(self):
        fetcher_val = self.loaders.test
        self._load_checkpoint(self.args.linear_epochs, 'debiased_linear')
        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0)
