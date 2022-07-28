from torch.utils.tensorboard import SummaryWriter
from os.path import join as ospj
import os
import logging

import torch
from data_aug.data_loader import get_original_loader, get_val_loader, InputFetcher
from training.LinearEvalSolver import LinearEvalSolver


class OversampleSolver(LinearEvalSolver):
    def __init__(self, args):
        super(OversampleSolver, self).__init__(args)
        self.writer = SummaryWriter(ospj(args.log_dir, 'debiased_eval'))

    def make_pseudo_label(self):
        score_file_name = lambda ld: ospj(self.args.score_file_template(ld), 'score_idx.pth')
        score_file = score_file_name(0.) # It must exists
        score = torch.load(score_file)

        for ld in self.args.lambda_list:
            if ld == 0.:
                continue
            else:
                new_score = torch.load(score_file_name(ld))
                score += new_score
        score /= len(self.args.lambda_list)
        pseudo_label = (score > self.args.cutoff).float()

        wrong_idx_path = ospj(self.args.checkpoint_dir, 'wrong_index.pth')
        torch.save(pseudo_label, wrong_idx_path)

        debias_idx_path = ospj(self.args.checkpoint_dir, 'debias_idx.pth')
        debias_label = torch.load(debias_idx_path)

        self.pseudo_label_precision_recall(pseudo_label, debias_label)

    def train_with_oversampling(self):
        """
        EpochEnsemble
        1. For each lambda_offdiag, load bias-score file
        2. Take average of score files, and make a pseudo bias label
        3. Run debiased linear evaluation
        """

        assert self.args.lambda_offdiag == 0 # Assert the main encoder is pretrained w/o rank regularization

        final_index_pth = ospj(self.args.checkpoint_dir, 'wrong_index.pth')
        if os.path.exists(final_index_pth):
            print('Bias label exists. Move onto linear evaluation')
        else:
            self.make_pseudo_label()
            print('Saved pseudo bias label')

        if self.args.oversample_pth is not None: # Only for manual pseudo_label experiments
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
            raise ValueError('Either pretrained SimCLR or pseudo bias label does not exist')

        wrong_label = torch.load(pth)
        print(f'Number of wrong/total samples: {wrong_label.sum()}/{wrong_label.size(0)}')
        upweight = torch.ones_like(wrong_label)
        upweight[wrong_label == 1] = self.args.lambda_upweight
        upweight_loader = get_original_loader(self.args, sampling_weight=upweight, simclr_aug=False)
        upweight_fetcher = InputFetcher(upweight_loader)

        self.linear_evaluation(upweight_fetcher, token='debiased_linear')

    def evaluate(self):
        fetcher_val = self.loaders.test
        self._load_checkpoint(self.args.linear_iters, 'debiased_linear')
        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0)
