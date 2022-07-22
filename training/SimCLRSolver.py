import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging
import sys

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, CheckpointIO, MultiDimAverageMeter

from models.build_models import build_model, num_classes, last_dim
from data_aug.data_loader import get_original_loader, get_val_loader, InputFetcher


class SimCLRSolver(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes[args.data]
        self.attr_dims = [self.num_classes, self.num_classes]

        self.nets = build_model(args)
        # below setattrs are to make networks be children of Solver, e.g., for self.to(self.device)
        for name, module in self.nets.items():
            setattr(self, name, module)

        self.optims = Munch() # Used in pretraining
        for net in self.nets.keys():
            if args.optimizer == 'Adam':
                self.optims[net] = torch.optim.Adam(
                    self.nets[net].parameters(),
                    args.lr_simclr,
                    weight_decay=args.weight_decay
                )
            elif args.optimizer == 'SGD':
                self.optims[net] = torch.optim.SGD(
                    self.nets[net].parameters(),
                    lr=args.lr_simclr,
                    momentum=0.9,
                    weight_decay=args.weight_decay
                )

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'), **self.nets),
        ]
        logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'),
                            level=logging.INFO)

        # BUILD LOADERS
        self.loaders = Munch(train_simclr=get_original_loader(args),
                             train_linear=get_original_loader(args, simclr_aug=False),
                             val=get_val_loader(args, split='valid'),
                             test=get_val_loader(args, split='test'))

        self.scheduler = Munch()
        for net in self.nets.keys():
            self.scheduler[net] = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optims[net], T_max=len(self.loaders.train_simclr), eta_min=0, last_epoch=-1)

        self.writer = SummaryWriter(args.log_dir)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.normalize = nn.BatchNorm1d(last_dim[args.arch], affine=False)

        self.to(self.device)

    def _reset_grad(self):
        def _recursive_reset(optims_dict):
            for _, optim in optims_dict.items():
                if isinstance(optim, dict):
                    _recursive_reset(optim)
                else:
                    optim.zero_grad()
        return _recursive_reset(self.optims)

    def _save_checkpoint(self, step, token):
        for ckptio in self.ckptios:
            ckptio.save(step, token)

    def _load_checkpoint(self, step, token, which=None, return_fname=False):
        for ckptio in self.ckptios:
            ckptio.load(step, token, which, return_fname)

    def info_nce_loss(self, features):
        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def off_diagonal(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def contrastive_train(self):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.simclr_epochs} epochs.")

        for epoch_counter in range(self.args.simclr_epochs):
            for images, _, _, _ in tqdm(self.loaders.train_simclr):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    aux = self.nets.encoder(images, simclr=True, penultimate=True)
                    features_simclr = aux['simclr']
                    logits, labels = self.info_nce_loss(features_simclr)
                    loss_nce = self.criterion(logits, labels)

                    # ---------------------------------------------------------
                    # Covariance regularization
                    features_penul = aux['penultimate']
                    c = self.normalize(features_penul).T @ self.normalize(features_penul)
                    c.div_(self.args.batch_size)
                    loss_offdiag = self.off_diagonal(c).pow_(2).sum() / features_penul.size(1) ** 2
                    # ---------------------------------------------------------

                    loss = loss_nce - self.args.lambda_offdiag * loss_offdiag


                self.optims.encoder.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optims.encoder)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    self.writer.add_scalar('loss_nce', loss_nce, global_step=n_iter)
                    self.writer.add_scalar('loss_offdiag', loss_offdiag, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.encoder.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= self.args.lr_decay_offset:
                self.scheduler.encoder.step()
            msg = f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            logging.info(msg)
            print(msg)

        logging.info("Training has finished.")
        # save model checkpoints
        self._save_checkpoint(step=epoch_counter+1, token='biased_simclr')

        logging.info(f"Model checkpoint and metadata has been saved at {self.args.log_dir}.")
