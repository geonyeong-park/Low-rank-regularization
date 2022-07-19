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
from training.SimCLRSolver import SimCLRSolver


class LinearEvalSolver(SimCLRSolver):
    def __init__(self, args):
        super(LinearEvalSolver, self).__init__(args)
        self.writer = SummaryWriter(ospj(args.log_dir, 'linear_eval'))

    def validation(self, fetcher, submunch):
        submunch.encoder.eval()
        submunch.classifier.eval()

        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)

        total_correct, total_num = 0, 0

        for images, labels, bias, _ in tqdm(fetcher):
            label = labels.to(self.device)
            images = images.to(self.device)
            bias = bias.to(self.device)

            with torch.no_grad():
                aux = submunch.encoder(images, penultimate=True)
                features = aux['penultimate']
                logit = submunch.classifier(features)
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

        submunch.encoder.train()
        submunch.classifier.train()
        return total_acc, accs

    def report_validation(self, valid_attrwise_acc, valid_acc, step=0):
        eye_tsr = torch.eye(self.attr_dims[0]).long()
        valid_acc_align = valid_attrwise_acc[eye_tsr == 1].mean().item()
        valid_acc_conflict = valid_attrwise_acc[eye_tsr == 0].mean().item()

        all_acc = dict()
        for acc, key in zip([valid_acc, valid_acc_align, valid_acc_conflict],
                            ['Acc/total', 'Acc/align', 'Acc/conflict']):
            all_acc[key] = acc
        log = f"(Validation) Iteration [{step+1}], "
        log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_acc.items()])
        logging.info(log)
        print(log)

    def linear_evaluation(self, submunch, token):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start Linear evaluation for {self.args.linear_epochs} epochs.")

        for epoch_counter in range(self.args.linear_epochs):
            for images, labels, _, _ in tqdm(self.loaders.train_linear):

                images = images.to(self.args.device)
                labels = labels.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    aux = submunch.encoder(images, freeze=True, penultimate=True)
                    features_penul = aux['penultimate']
                    logits = submunch.classifier(features_penul)
                    loss = self.criterion(logits, labels)

                submunch.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(submunch.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    top1 = accuracy(logits, labels, topk=(1, ))
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', submunch.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                submunch.scheduler.step()

            if (epoch_counter+1) % self.args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(self.loaders.val, submunch)
                self.report_validation(valid_attrwise_acc, total_acc, n_iter+1)

            msg = f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}"
            logging.info(msg)
            print(msg)

        logging.info("Training has finished.")
        # save model checkpoints
        self._save_checkpoint(step=epoch_counter+1, token=token)

        logging.info(f"Model checkpoint and metadata has been saved at {self.args.log_dir}.")

    def train(self):
        try:
            self._load_checkpoint(self.args.simclr_epochs, 'biased_simclr')
            print('Pretrained SimCLR ckpt exists. Move onto linear evaluation')
        except:
            print('Start SimCLR pretraining...')
            self.train_fb()
            print('Finished pretraining. Move onto linear evaluation')

        self.train_cb()

    def train_cb(self):
        submunch = Munch(encoder=self.nets.fb,
                         classifier=self.nets.cb,
                         optimizer=self.optims.cb,
                         scheduler=self.scheduler.cb)
        self.linear_evaluation(submunch, 'biased_linear')


