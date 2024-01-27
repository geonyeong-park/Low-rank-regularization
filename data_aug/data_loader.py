import os
from os.path import join as ospj
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from torch.utils import data
from munch import Munch
from data_aug.utkface import get_utk_face
from data_aug.celebA import get_celeba
from data_aug.bffhq import get_bFFHQ
from data_aug.merge_dataset import get_stl10mnist
from data_aug.imagenet import get_imagenet
from data_aug.mimic import get_mimic
from data_aug.mimic_nih import get_mimic_nih

from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset


def get_original_loader(args, return_dataset=False, sampling_weight=None, simclr_aug=True,
                        finetune=False, inverted_sampling=False, shuffle=True, return_num_data=False, finetune_ratio=0.):
    dataset_name = args.data
    if dataset_name == 'UTKFace':
        dataset = get_utk_face(args.data_dir, bias_attr=args.bias_attr, target_attr=args.target_attr, split='train',
                               simclr_aug=simclr_aug, img_size=64, bias_rate=0.9,)
    elif dataset_name == 'celebA':
        dataset = get_celeba(args.data_dir, target_attr=args.target_attr, split='train',
                             simclr_aug=simclr_aug, img_size=224)
    elif dataset_name == 'bffhq':
        dataset = get_bFFHQ(args.data_dir, split='train', simclr_aug=simclr_aug)
    elif dataset_name == 'stl10mnist':
        dataset = get_stl10mnist(args.data_dir, split='train' if not simclr_aug else 'unlabeled', simclr_aug=simclr_aug,
                                 num_unique_mnist=args.num_unique_mnist)
    elif dataset_name == 'imagenet':
        dataset = get_imagenet(ospj(args.data_dir, 'train'), train=True, simclr_aug=simclr_aug)
    elif dataset_name == 'MIMIC_CXR':
        dataset = get_mimic(args.data_dir, split='train', simclr_aug=simclr_aug)
    elif dataset_name == 'MIMIC_NIH':
        dataset = get_mimic_nih(args.data_dir, split='train', simclr_aug=simclr_aug)
    else:
        raise ValueError

    if finetune:
        assert finetune_ratio != 0
        indices_file = ospj(args.checkpoint_dir, f'subset_indices_{finetune_ratio}.npy')

        if os.path.exists(indices_file):
            print(f'{indices_file} exists')
            indices = np.load(indices_file)
            if inverted_sampling:
                num_data = len(dataset)
                total_indices = np.arange(num_data)
                inverted_indices = [ind for ind in total_indices if ind not in indices]
                indices = inverted_indices
            dataset = Subset(dataset, indices)
        else:
            num_data = len(dataset)
            indices = np.random.choice(num_data, int(finetune_ratio*num_data))
            dataset = Subset(dataset, indices)
            print('*'*50)
            print(f'Sample subset of training samples for finetuning: length = {len(dataset)}\n')
            np.save(indices_file, indices)
            print(f'Saved indices in {indices_file}')
            print('*'*50)

    if return_dataset:
        return dataset
    else:
        if sampling_weight is not None:
            # replacement = False if sampling_weight.sum() > args.batch_size else True
            sampler = WeightedRandomSampler(sampling_weight, args.batch_size, replacement=True)
            return data.DataLoader(dataset=dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.num_workers,
                                   sampler=sampler,
                                   pin_memory=True,
                                   drop_last=simclr_aug)
        else:
            return data.DataLoader(dataset=dataset,
                                   batch_size=args.batch_size,
                                   shuffle=shuffle,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=simclr_aug)

def get_val_loader(args, split='valid'):
    dataset_name = args.data
    if dataset_name == 'UTKFace':
        dataset = get_utk_face(args.data_dir, bias_attr=args.bias_attr, target_attr=args.target_attr, split=split,
                               simclr_aug=False, img_size=64, bias_rate=0.,)
    elif dataset_name == 'celebA':
        if split == 'valid':
            split = 'train_valid'
        elif split == 'test':
            split = 'valid'
        dataset = get_celeba(args.data_dir, target_attr=args.target_attr, split=split,
                             simclr_aug=False, img_size=224)
    elif dataset_name == 'bffhq':
        dataset = get_bFFHQ(args.data_dir, split=split, simclr_aug=False)
    elif dataset_name == 'stl10mnist':
        dataset = get_stl10mnist(args.data_dir, split='test', simclr_aug=False,
                                 bias_ratio=0.1)
    elif dataset_name == 'imagenet':
        assert split in ['biased', 'unbiased', 'ImageNet-A']
        if split == 'biased' or split == 'unbiased':
            dataset = get_imagenet(ospj(args.data_dir, 'val'), train=False, simclr_aug=False)
        else:
            dataset = get_imagenet(args.imagenetA_dir, train=False, simclr_aug=False,
                                   val_data='ImageNet-A')
    elif dataset_name == 'MIMIC_CXR':
        dataset = get_mimic(args.data_dir, split=split, simclr_aug=False)
    elif dataset_name == 'MIMIC_NIH':
        dataset = get_mimic_nih(args.data_dir, split='valid', simclr_aug=False)
    else:
        raise ValueError

    return data.DataLoader(dataset=dataset,
                           batch_size=args.batch_size,
                           shuffle=False,
                           num_workers=args.num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader):
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _fetch(self):
        try:
            x, target, bias, index = next(self.iter) # attr: (class_label, bias_label)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, target, bias, index = next(self.iter) # attr: (class_label, bias_label)
        return x, target, bias, index

    def __len__(self):
        return len(self.loader)

    def __next__(self):
        x, y, bias_label, index = self._fetch()
        bias_label = bias_label.to(self.device)
        inputs = Munch(index=index, images=x, labels=y, bias_label=bias_label)

        return Munch({k: v.to(self.device) for k, v in inputs.items()})

