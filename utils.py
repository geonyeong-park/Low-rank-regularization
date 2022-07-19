import os
from os.path import join as ospj
import shutil

import torch
import yaml
import numpy as np
import json


def setup(args):
    args.data_dir = ospj(args.data_dir, args.data)

    fname = args.exp_name
    if args.data == 'UTKFace':
        attr = args.bias_attr
    elif args.data == 'celebA':
        attr = args.target_attr

    if fname is None:
        fname = f'{args.data}_{attr}_lambda_{args.lambda_offdiag}_seed_{args.seed}'
    else:
        fname = f'{args.data}_{attr}_lambda_{args.lambda_offdiag}_{fname}_seed_{args.seed}'

    args.log_dir = ospj(args.log_dir, fname)
    os.makedirs(args.log_dir, exist_ok=True)

    args.checkpoint_dir = ospj(args.checkpoint_dir, fname)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    return args


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float))

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()


class CheckpointIO(object):
    def __init__(self, fname_template, **kwargs):
        os.makedirs(os.path.dirname(fname_template), exist_ok=True)
        self.fname_template = fname_template
        self.module_dict = kwargs

    def register(self, **kwargs):
        self.module_dict.update(kwargs)

    def save_label(self, label_dict):
        print('Saving domain label configuration...')
        torch.save(label_dict, self.fname_template)

    def load_label(self):
        return torch.load(self.fname_template)

    def save(self, step, token):
        fname = self.fname_template.format(step, token)
        print('Saving checkpoint into %s...' % fname)
        outdict = {}
        for name, module in self.module_dict.items():
            outdict[name] = module.state_dict()
        torch.save(outdict, fname)

    def load(self, step, token, which=None, return_fname=False):
        fname = self.fname_template.format(step, token)
        if not os.path.exists(fname): print(f'WARNING: {fname} does not exist!')
        if return_fname: return fname
        print('Loading checkpoint from %s...' % fname)
        if torch.cuda.is_available():
            module_dict = torch.load(fname)
        else:
            module_dict = torch.load(fname, map_location=torch.device('cpu'))

        if which:
            module = self.module_dict[which]
            module.load_state_dict(module_dict[which])
            return

        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name])
