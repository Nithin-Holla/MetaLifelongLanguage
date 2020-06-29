import logging
import math
from collections import defaultdict

import higher
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets
import models.utils
from models.base_models import TransformerRLN, ReplayMemory, TransformerClsModel

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ANML-Log')


class ANML:

    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.device = device

        self.nm = TransformerRLN(model_name=kwargs.get('model'),
                                 max_length=kwargs.get('max_length'),
                                 device=device)
        self.pn = TransformerClsModel(model_name=kwargs.get('model'),
                                      n_classes=n_classes,
                                      max_length=kwargs.get('max_length'),
                                      device=device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info('Loaded {} as NM'.format(self.nm.__class__.__name__))
        logger.info('Loaded {} as PN'.format(self.pn.__class__.__name__))

        meta_params = [p for p in self.nm.parameters() if p.requires_grad] + \
                      [p for p in self.pn.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def group_by_class(self, data_set, mini_batch_size):
        grouped_text = defaultdict(list)
        grouped_data_set = []
        for txt, lbl in zip(data_set['text'], data_set['label']):
            grouped_text[lbl].append(txt)
        for lbl in grouped_text.keys():
            for i in range(0, len(grouped_text[lbl]), mini_batch_size):
                subset = grouped_text[lbl][i: i + mini_batch_size]
                grouped_data_set.append((subset, [lbl] * len(subset)))
        return grouped_data_set

    def scaled_gating(self, tanh_value):
        return 0.5 * (tanh_value + 1)

    def save_model(self, model_path):
        checkpoint = {'nm': self.nm.state_dict(),
                      'pn': self.pn.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.nm.load_state_dict(checkpoint['nm'])
        self.pn.load_state_dict(checkpoint['pn'])

    def evaluate(self, dataloader, updates, mini_batch_size):

        support_set = defaultdict(list)
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=mini_batch_size)
            support_set['text'].extend(text)
            support_set['label'].extend(labels)

        support_set = self.group_by_class(support_set, mini_batch_size)

        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpn, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.pn.encode_text(text)
                repr = fpn(input_dict, out_from='transformers')
                modulation_tanh = self.nm(input_dict)
                modulation = self.scaled_gating(modulation_tanh)
                output = fpn(repr * modulation, out_from='linear')
                loss = self.loss_fn(output, labels)
                diffopt.step(loss)
                pred = models.utils.make_prediction(output.detach())
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(labels.tolist())

            acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                        'recall = {:.4f}, F1 score = {:.4f}'.format(np.mean(support_loss), acc, prec, rec, f1))

            all_losses, all_predictions, all_labels = [], [], []

            for text, labels in dataloader:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.pn.encode_text(text)
                with torch.no_grad():
                    repr = fpn(input_dict, out_from='transformers')
                    modulation_tanh = self.nm(input_dict)
                    modulation = self.scaled_gating(modulation_tanh)
                    output = fpn(repr * modulation, out_from='linear')
                    loss = self.loss_fn(output, labels)
                loss = loss.item()
                pred = models.utils.make_prediction(output.detach())
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(labels.tolist())

        acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))

        return acc, prec, rec, f1

    def training(self, train_datasets, **kwargs):
        n_episodes = kwargs.get('n_episodes')
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')

        replay_freq = int(max(1, math.ceil(1 / ((updates + 1) * self.replay_rate))))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=mini_batch_size, shuffle=False,
                                                collate_fn=datasets.utils.batch_encode))

        for episode_id in range(n_episodes):

            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

            with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpn, diffopt):

                # Inner loop
                support_set = defaultdict(list)
                task_predictions, task_labels = [], []
                for _ in range(updates):
                    try:
                        text, labels = next(train_dataloader)
                        support_set['text'].extend(text)
                        support_set['label'].extend(labels)
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                support_set = self.group_by_class(support_set, mini_batch_size)

                for text, labels in support_set:
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.pn.encode_text(text)
                    repr = fpn(input_dict, out_from='transformers')
                    modulation_tanh = self.nm(input_dict)
                    modulation = self.scaled_gating(modulation_tanh)
                    output = fpn(repr * modulation, out_from='linear')
                    loss = self.loss_fn(output, labels)
                    diffopt.step(loss)
                    pred = models.utils.make_prediction(output.detach())
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                    self.memory.write_batch(text, labels)

                acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

                logger.info('Episode {}/{} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1, n_episodes,
                                                                        np.mean(support_loss), acc, prec, rec, f1))

                # Outer loop
                query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                query_set = []
                try:
                    text, labels = next(train_dataloader)
                    query_set.append((text, labels))
                except StopIteration:
                    logger.info('Terminating training as all the data is seen')
                    return

                if (episode_id + 1) % replay_freq == 0:
                    text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                    query_set.append((text, labels))

                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.pn.encode_text(text)
                    repr = fpn(input_dict, out_from='transformers')
                    modulation_tanh = self.nm(input_dict)
                    modulation = self.scaled_gating(modulation_tanh)
                    output = fpn(repr * modulation, out_from='linear')
                    loss = self.loss_fn(output, labels)
                    query_loss.append(loss.item())
                    pred = models.utils.make_prediction(output.detach())

                    acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
                    query_acc.append(acc)
                    query_prec.append(prec)
                    query_rec.append(rec)
                    query_f1.append(f1)

                    # NM meta gradients
                    nm_params = [p for p in self.nm.parameters() if p.requires_grad]
                    meta_nm_grads = torch.autograd.grad(loss, nm_params, retain_graph=True)
                    for param, meta_grad in zip(nm_params, meta_nm_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PN meta gradients
                    pn_params = [p for p in fpn.parameters(time=0) if p.requires_grad]
                    meta_pn_grads = torch.autograd.grad(loss, pn_params)
                    pn_params = [p for p in self.pn.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pn_params, meta_pn_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                # Meta optimizer step
                nm_params = [p for p in self.nm.parameters() if p.requires_grad]
                pn_params = [p for p in self.pn.parameters() if p.requires_grad]
                for param in nm_params + pn_params:
                    param.grad /= len(query_set)
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

                logger.info('Episode {}/{} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1, n_episodes,
                                                                        np.mean(query_loss), np.mean(query_acc),
                                                                        np.mean(query_prec), np.mean(query_rec),
                                                                        np.mean(query_f1)))

    def testing(self, test_datasets, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        accuracies, precisions, recalls, f1s = [], [], [], []
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
