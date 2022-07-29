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


class ERMSolver(nn.Module):
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
                    args.lr_ERM,
                    weight_decay=args.weight_decay
                )
            elif args.optimizer == 'SGD':
                self.optims[net] = torch.optim.SGD(
                    self.nets[net].parameters(),
                    args.lr_ERM,
                    momentum=0.9,
                    weight_decay=args.weight_decay
                )

        self.ckptios = [
            CheckpointIO(ospj(args.checkpoint_dir, '{:06d}_{}_nets.ckpt'), **self.nets),
        ]
        logging.basicConfig(filename=os.path.join(args.log_dir, 'training.log'),
                            level=logging.INFO)

        # BUILD LOADERS
        self.loaders = Munch(train=get_original_loader(args, simclr_aug=False),
                             val=get_val_loader(args, split='valid'),
                             test=get_val_loader(args, split='test'))

        self.scheduler = Munch()
        milestones = [int(args.ERM_epochs/3), int(args.ERM_epochs/3)]
        self.scheduler.classifier = torch.optim.lr_scheduler.MultiStepLR(
            self.optims[net], milestones=milestones, gamma=args.lr_decay_gamma
        )

        self.writer = SummaryWriter(args.log_dir)
        self.criterion = torch.nn.CrossEntropyLoss()

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

    def validation(self, fetcher):
        self.nets.classifier.eval()

        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)

        total_correct, total_num = 0, 0

        for images, labels, bias, _ in tqdm(fetcher):
            label = labels.to(self.device)
            images = images.to(self.device)
            bias = bias.to(self.device)

            with torch.no_grad():
                logit = self.nets.classifier(images)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

                total_correct += correct.sum()
                total_num += correct.shape[0]

            attr = torch.cat((labels.view(-1,1).to(self.device), bias.view(-1,1).to(self.device)), dim=1)
            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        #print(attrwise_acc_meter.cum.view(self.attr_dims[0], -1))
        #print(attrwise_acc_meter.cnt.view(self.attr_dims[0], -1))

        total_acc = total_correct / float(total_num)
        accs = attrwise_acc_meter.get_mean()

        self.nets.classifier.train()
        return total_acc, accs

    def report_validation(self, valid_attrwise_acc, valid_acc, step=0):
        eye_tsr = torch.eye(self.attr_dims[0]).long()
        valid_acc_align = valid_attrwise_acc[eye_tsr == 1].mean().item()
        valid_acc_conflict = valid_attrwise_acc[eye_tsr == 0].mean().item()

        all_acc = dict()
        for acc, key in zip([valid_acc, valid_acc_align, valid_acc_conflict],
                            ['Acc/total', 'Acc/align', 'Acc/conflict']):
            all_acc[key] = acc
            self.writer.add_scalar(key, acc, global_step=step)
        log = f"(Validation) Iteration [{step}], "
        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_acc.items()])
        logging.info(log)
        print(log)

    def train(self):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start ERM training for {self.args.ERM_epochs} epochs.")

        for epoch_counter in range(self.args.ERM_epochs):
            for images, labels, _, _ in tqdm(self.loaders.train):
                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    logits = self.nets.classifier(images)
                    loss = self.criterion(logits, labels)

                self.optims.classifier.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optims.classifier)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1 = accuracy(logits, labels, topk=(1, ))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.classifier.get_lr()[0], global_step=n_iter)

                n_iter += 1

                self.scheduler.classifier.step()

            msg = f"Epoch: {epoch_counter}\tLR: {self.scheduler.classifier.get_lr()[0]}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            logging.info(msg)
            print(msg)

            if (epoch_counter + 1) % self.args.eval_every == 0:
                self._save_checkpoint(step=epoch_counter+1, token='biased_ERM')

            if (n_iter+1) % self.args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(self.loaders.val)
                self.report_validation(valid_attrwise_acc, total_acc, n_iter+1)
                msg = f"Iter: {n_iter+1}\tLoss: {loss}\tAccuracy: {total_acc}"
                logging.info(msg)
                print(msg)



        logging.info("Training has finished.")
        # save model checkpoints
        self._save_checkpoint(step=epoch_counter+1, token='biased_ERM')

        logging.info(f"Model checkpoint and metadata has been saved at {self.args.log_dir}.")

    def evaluate(self):
        fetcher_val = self.loaders.test
        self._load_checkpoint(self.args.ERM_epochs, 'biased_ERM')
        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0)
