import os
from os.path import join as ospj
import shutil

import torch
import yaml
import numpy as np
import json


def setup(args):
    args.data_dir = ospj(args.data_dir, args.data)

    if args.data == 'UTKFace':
        attr = args.bias_attr
    elif args.data == 'celebA':
        attr = args.target_attr
    elif args.data == 'bffhq' or args.data == 'imagenet':
        attr = ''
    elif args.data == 'stl10mnist':
        attr = args.num_unique_mnist

    if args.exp_name is not None:
        exp = f'{args.exp_name}_'
    else:
        exp = ''


    if args.mode != 'ERM':
        ckpt_tmp = args.checkpoint_dir
        fname_template = lambda ld: ospj(ckpt_tmp, f'{args.data}_{exp}{attr}_{args.mode_CL}_lambda_{ld}_seed_{args.seed}')
        fname = f'{args.data}_{exp}{attr}_{args.mode_CL}_lambda_{args.lambda_offdiag}_seed_{args.seed}'
    else:
        fname = f'{args.data}_{exp}{attr}_ERM_seed_{args.seed}'

    args.log_dir = ospj(args.log_dir, fname)
    os.makedirs(args.log_dir, exist_ok=True)


    args.checkpoint_dir = ospj(args.checkpoint_dir, fname)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.mode != 'ERM':
        args.score_file_template = fname_template

    return args


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config(args):
    with open(os.path.join(args.log_dir, 'args.txt'), 'a') as f:
        json.dump(args.__dict__, f, indent=2)


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

    def save(self, pth):
        acc = (self.cum / self.cnt)
        torch.save(acc, pth)


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
            module.load_state_dict(module_dict[which], strict=False)
            return

        for name, module in self.module_dict.items():
            module.load_state_dict(module_dict[name], strict=False)
