import os
from os.path import join as ospj
import logging

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, CheckpointIO, MultiDimAverageMeter

from data_aug.data_loader import get_original_loader, get_val_loader, InputFetcher
from training.SimCLRSolver import SimCLRSolver


class LinearEvalSolver(SimCLRSolver):
    def __init__(self, args):
        super(LinearEvalSolver, self).__init__(args)
        self.writer = SummaryWriter(ospj(args.log_dir, 'linear_eval'))

    def validation(self, fetcher):
        self.nets.encoder.eval()
        self.nets.classifier.eval()

        attrwise_acc_meter = MultiDimAverageMeter(self.attr_dims)

        total_correct, total_num = 0, 0

        for images, labels, bias, _ in tqdm(fetcher):
            label = labels.to(self.device)
            images = images.to(self.device)
            bias = bias.to(self.device)

            with torch.no_grad():
                aux = self.nets.encoder(images, penultimate=True)
                features = aux['penultimate']
                logit = self.nets.classifier(features)
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

        self.nets.encoder.train()
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

    def save_wrong_idx(self, loader, token='wrong_index.pth'):
        self.nets.encoder.eval()
        self.nets.classifier.eval()

        iterator = enumerate(loader)
        total_wrong, total_num = 0, 0
        wrong_idx = torch.empty(0).to(self.device)
        debias_idx = torch.empty(0).to(self.device)

        for _, (images, labels, bias_labels, idx) in iterator:
            idx = idx.to(self.device)
            labels = labels.to(self.device)
            bias_labels = bias_labels.to(self.device)
            images= images.to(self.device)

            with torch.no_grad():
                aux = self.nets.encoder(images, freeze=True, penultimate=True)
                features_penul = aux['penultimate']
                logits = self.nets.classifier(features_penul)

                pred = logits.data.max(1, keepdim=True)[1].squeeze(1)
                wrong = (pred != labels).long()
                debiased = (labels != bias_labels).long()

                total_wrong += wrong.sum()
                total_num += wrong.shape[0]
                wrong_idx = torch.cat((wrong_idx, idx[wrong == 1])).long()
                debias_idx = torch.cat((debias_idx, idx[debiased == 1])).long()

        assert total_wrong == len(wrong_idx)
        print(f'Number of wrong/total samples: {total_wrong}/{total_num}')

        self.nets.encoder.train()
        self.nets.classifier.train()
        self.confirm_pseudo_label(wrong_idx, debias_idx, total_num, token)

    def confirm_pseudo_label(self, wrong_idx, debias_idx, total_num, token):
        wrong_label = torch.zeros(total_num).to(self.device)
        debias_label = torch.zeros(total_num).to(self.device)

        for idx in wrong_idx:
            wrong_label[idx] = 1
        for idx in debias_idx:
            debias_label[idx] = 1

        spur_precision = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(wrong_label)
        print("Spurious precision", spur_precision)
        spur_recall = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(debias_label)
        print("Spurious recall", spur_recall)

        wrong_idx_path = ospj(self.args.checkpoint_dir, token)

        torch.save(wrong_label, wrong_idx_path)
        print(f'Saved wrong index label in {wrong_idx_path}')

    def linear_evaluation(self, loader, token='biased_linear'):
        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.args.log_dir, self.args)

        logging.info(f"Start Linear evaluation for {self.args.linear_iters} iterations.")

        for iter_counter in range(self.args.linear_iters):
            inputs = next(loader)
            images, labels = inputs.images, inputs.labels

            with autocast(enabled=self.args.fp16_precision):
                aux = self.nets.encoder(images, freeze=True, penultimate=True)
                features_penul = aux['penultimate']
                logits = self.nets.classifier(features_penul)
                loss = self.criterion(logits, labels)

            self.optims.classifier.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(self.optims.classifier)
            scaler.update()

            if (iter_counter + 1) % self.args.log_every_n_steps == 0:
                top1 = accuracy(logits, labels, topk=(1, ))
                self.writer.add_scalar('loss', loss, global_step=iter_counter+1)
                self.writer.add_scalar('acc/top1', top1[0], global_step=iter_counter+1)
                self.writer.add_scalar('learning_rate', self.scheduler.classifier.get_lr()[0], global_step=iter_counter+1)

            # warmup for the first 10 epochs
            if (iter_counter+1) / len(loader) >= self.args.lr_decay_offset and (iter_counter+1) % len(loader) == 0:
                self.scheduler.classifier.step()

            if (iter_counter+1) % self.args.eval_every == 0:
                total_acc, valid_attrwise_acc = self.validation(self.loaders.val)
                self.report_validation(valid_attrwise_acc, total_acc, iter_counter+1)
                msg = f"Iter: {iter_counter+1}\tLoss: {loss}\tAccuracy: {total_acc}"
                logging.info(msg)
                print(msg)

        logging.info("Training has finished.")
        # save model checkpoints
        self._save_checkpoint(step=iter_counter+1, token=token)

        logging.info(f"Model checkpoint and metadata has been saved at {self.args.log_dir}.")

    def train(self):
        try:
            self._load_checkpoint(self.args.simclr_epochs, 'biased_simclr')
            print('Pretrained SimCLR ckpt exists. Move onto linear evaluation')
        except:
            print('Start SimCLR pretraining...')
            self.contrastive_train()
            print('Finished pretraining. Move onto linear evaluation')

        if os.path.exists(ospj(self.args.checkpoint_dir, 'wrong_index.pth')):
            print('Ours: Linear evaluation with subsampled dataset.')
            wrong_label = torch.load(ospj(self.args.checkpoint_dir, 'wrong_index.pth'))
            sampling_weight = 1. - wrong_label
            balanced_loader = get_original_loader(self.args, sampling_weight=sampling_weight, simclr_aug=False)
            balanced_fetcher = InputFetcher(balanced_loader)
            self.linear_evaluation(balanced_fetcher)
            self.save_wrong_idx(self.loaders.train_linear, token='wrong_index2.pth')
        else:
            fetcher = InputFetcher(self.loaders.train_linear)
            self.linear_evaluation(fetcher)
            self.save_wrong_idx(self.loaders.train_linear)

    def evaluate(self):
        fetcher_val = self.loaders.test
        self._load_checkpoint(self.args.linear_iters, 'biased_linear')
        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0)
