import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torch.utils.data.dataset import Dataset

from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import MNIST, STL10
from torchvision import transforms as T
import logging

from data_aug.view_generator import ContrastiveLearningViewGenerator


class STL10MNIST(Dataset):
    def __init__(self, root, train='unlabeled', transform=None,
                 download=False, num_unique_mnist=60000, bias_ratio=0.9):

        self.num_unique_mnist = num_unique_mnist
        self.n_confusing_labels = 9
        self.bias_ratio = bias_ratio
        self.transform = transform
        self.train = True if train != 'test' else False
        self.random = True if train != 'test' else False

        self.stl = datasets.STL10(root, split=train, transform=None, download=download)
        self.stl.data = np.transpose(self.stl.data, (0, 2, 3, 1))
        self.mnist = MNIST(root, train=True, download=download)
        self.mnist.data = transforms.Resize(32)(self.mnist.data)

        indices = np.arange(len(self.mnist.data))
        np.random.shuffle(indices)

        self.mnist_data = np.expand_dims(self.mnist.data[indices], -1)
        self.mnist_data[self.mnist_data < 100] = 0
        self.mnist_data[self.mnist_data >= 100] = 1
        self.mnist_data = np.tile(self.mnist_data, (1, 1, 1, 3))
        self.mnist_targets = self.mnist.targets[indices]
        self.mnist_data = self.mnist_data[:self.num_unique_mnist]
        self.mnist_targets = self.mnist_targets[:self.num_unique_mnist]

        if train == 'unlabeled': # Randomly shuffle
            self.data, self.biased_targets = self.make_data(self.num_unique_mnist, len(self.stl))
            self.targets = -torch.ones_like(self.biased_targets).long() # dummy
        elif train == 'test':
            self.data, self.biased_targets = self.make_data(self.num_unique_mnist, len(self.stl))
            self.targets = torch.LongTensor(self.stl.labels)
        elif train == 'train':
            self.data, self.biased_targets = self.make_data(self.num_unique_mnist, len(self.stl))
            self.targets = torch.LongTensor(self.stl.labels)
            #self.data, self.targets, self.biased_targets = self.make_biased_data()

    def _shuffle(self, iteratable):
        if self.random:
            np.random.shuffle(iteratable)

    def make_data(self, num_unique, num_data):
        rand_ind = np.random.randint(num_unique, size=num_data)
        mnist_data = self.mnist_data[rand_ind]
        targets = self.mnist_targets[rand_ind]
        stl_data = self.stl.data[:num_data]

        data = self._compose_stl10_mnist(stl_data, mnist_data)
        return data, targets

    def make_biased_data(self):
        """Build biased MNIST.
        """
        bias_indices = {label: torch.LongTensor() for label in range(10)}
        for label in range(10):
            self._update_bias_indices(bias_indices, label)

        data = np.empty((0, 96, 96, 3))
        targets = np.empty((0))
        biased_targets = []

        for bias_label, indices in bias_indices.items():
            _data, _targets = self._make_biased_data(indices, bias_label)
            data = np.concatenate([data, _data])
            targets = np.concatenate([targets, _targets])
            biased_targets.extend([bias_label] * len(indices))

        biased_targets = np.array(biased_targets)
        return data, targets, biased_targets

    def _update_bias_indices(self, bias_indices, label):
        if self.n_confusing_labels > 9 or self.n_confusing_labels < 1:
            raise ValueError(self.n_confusing_labels)

        indices = np.where((self.targets == label).numpy())[0]
        self._shuffle(indices)
        indices = torch.LongTensor(indices)

        n_samples = len(indices)
        n_correlated_samples = int(n_samples * self.bias_ratio)
        n_decorrelated_per_class = int(np.ceil((n_samples - n_correlated_samples) / (self.n_confusing_labels)))

        correlated_indices = indices[:n_correlated_samples]
        bias_indices[label] = torch.cat([bias_indices[label], correlated_indices])

        decorrelated_indices = torch.split(indices[n_correlated_samples:], n_decorrelated_per_class)

        other_labels = [_label % 10 for _label in range(label + 1, label + 1 + self.n_confusing_labels)]
        self._shuffle(other_labels)

        for idx, _indices in enumerate(decorrelated_indices):
            _label = other_labels[idx]
            bias_indices[_label] = torch.cat([bias_indices[_label], _indices])

    def _make_biased_data(self, indices, bias_label):
        targets = self.targets[indices]
        data = self.stl.data[indices]
        bias_indices = np.where((self.mnist_targets == bias_label).numpy())[0]
        self._shuffle(bias_indices)
        bias_indices = bias_indices[:len(indices)]
        bias_data = self.mnist_data[bias_indices]
        biased_data = self._compose_stl10_mnist(data, bias_data)
        return biased_data, targets

    def _compose_stl10_mnist(self, stl_data, mnist_data):
        """
        template = np.zeros_like(stl_data)
        template[:, 10:42, 10:42, :] = mnist_data
        template[:, 54:86, 54:86, :] = mnist_data
        """
        template = np.tile(mnist_data, (1, 3, 3, 1))
        data = np.clip(stl_data * (1 - template) + 255 * template, 0, 255)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.astype(np.uint8), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target, int(self.biased_targets[index]), index


def get_stl10mnist(root, split='train', simclr_aug=True,
                   img_size=224, num_unique_mnist=60000, bias_ratio=0.9):
    logging.info(f'get_stl10mnist - split:{split}, aug: {simclr_aug}')
    if split == 'train' and simclr_aug:
        split = 'unlabeled'
    elif split == 'train' and not simclr_aug:
        split = 'train'

    if split != 'test':
        if simclr_aug:
            transform = T.Compose([
                T.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
                T.RandomHorizontalFlip(),
                T.RandomApply([
                    T.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                T.RandomGrayscale(p=0.2),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        else:
            transform = T.Compose(
                [
                    T.Resize((img_size, img_size)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )
    else:
        transform = T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    if simclr_aug:
        transform = ContrastiveLearningViewGenerator(transform)

    dataset = STL10MNIST(
        root=root,
        train=split,
        transform=transform,
        num_unique_mnist=num_unique_mnist,
        bias_ratio=bias_ratio
    )


    return dataset
