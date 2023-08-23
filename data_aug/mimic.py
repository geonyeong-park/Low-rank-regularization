import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd
import logging
from data_aug.view_generator import ContrastiveLearningViewGenerator
from torchvision import transforms as T
import random
from PIL import ImageFilter


class MIMIC_Dataset(Dataset):
    def __init__(self, root, split, transform, **kwargs):
        """
        mimic_path (str):
            path to MIMIC CXRs (with subfolders p10, p11, ... p19)
        df (pd.DataFrame):
            DataFrame with columns 'dicom_id', 'subject_id',
            'study_id', and 'label'.
        """
        self.mimic_path = root
        whole_df = pd.read_csv(os.path.join(root, "mimic_bias_ratio_10%.csv"))
        self.df = whole_df[whole_df['split'] == split]
        self.transforms = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        subject_id = row['subject_id']
        study_id = row['study_id']
        dicom_id = row['dicom_id']

        img_path = f"{self.mimic_path}/files/p{str(subject_id)[:2]}/p{subject_id}/s{study_id}/{dicom_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        bias_label = row['bias_label']
        label = row['label']

        return img, label, bias_label, idx


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=1.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_mimic(root, split='train', simclr_aug=True, img_size=224):
    logging.info(f'get_celeba - split:{split}, aug: {simclr_aug}')

    if split == 'train':
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
            transform = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
                T.RandomRotation(degrees=(-5, 5)),
                T.RandomAutocontrast(p=0.3),
                T.RandomEqualize(p=0.3),
                GaussianBlur(),
                T.ToTensor(),
            ])
    else:
        transform = T.Compose(
            [
                T.Resize((224,224)),
                T.ToTensor(),
            ]
        )

    if simclr_aug:
        transform = ContrastiveLearningViewGenerator(transform)

    dataset = MIMIC_Dataset(
        root=root,
        split=split,
        transform=transform
    )
    return dataset
