import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd
import logging
from data_aug.view_generator import ContrastiveLearningViewGenerator
from torchvision import transforms


class bFFHQDataset(Dataset):
    def __init__(self, root, split='train', transform=None, conflict_pct=0.5):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root
        self.conflict_token = f'{conflict_pct}pct'

        if split=='train':
            self.header_dir = os.path.join(root, self.conflict_token)
            self.align = glob(os.path.join(self.header_dir, 'align', "*", "*"))
            self.conflict = glob(os.path.join(self.header_dir, 'conflict',"*", "*"))

            self.data = self.align + self.conflict

        elif split=='test':
            self.data = glob(os.path.join(root, 'test', "*"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, attr[0], attr[1], index # attr=(class_label, bias_label)


def get_bFFHQ(root, split='train', simclr_aug=False, img_size=128):
    logging.info(f'get_bFFHQ - split: {split}, aug: {simclr_aug}')
    size_dict = {64: 72, 128: 144, 224: 256}
    load_size = size_dict[img_size]

    if split == 'train':
        if simclr_aug:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])

    if simclr_aug:
        transform = ContrastiveLearningViewGenerator(transform)

    dataset = bFFHQDataset(root, transform=transform, split=split)
    return dataset
