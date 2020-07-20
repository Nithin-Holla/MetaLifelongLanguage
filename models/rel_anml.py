import logging
import math
from collections import defaultdict

import higher
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets.utils
import models.utils
from models.base_models import ReplayMemory, TransformerNeuromodulator, TransformerClsModel

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ANML-Log')


class ANML:

    def __init__(self, device, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.device = device

        self.nm = TransformerNeuromodulator(model_name=kwargs.get('model'),
                                            device=device)
        self.pn = TransformerClsModel(model_name=kwargs.get('model'),
                                      n_classes=1,
                                      max_length=kwargs.get('max_length'),
                                      device=device)

        logger.info('Loaded {} as NM'.format(self.nm.__class__.__name__))
        logger.info('Loaded {} as PN'.format(self.pn.__class__.__name__))

        meta_params = [p for p in self.nm.parameters() if p.requires_grad] + \
                      [p for p in self.pn.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=3)
        self.loss_fn = nn.BCEWithLogitsLoss()

        inner_params = [p for p in self.pn.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def group_by_relation(self, data_set, mini_batch_size):
        grouped_text = defaultdict(list)
        grouped_data_set = []
        for txt, lbl, cand in zip(data_set['text'], data_set['label'], data_set['candidates']):
            key = ' '.join(lbl)
            grouped_text[key].append((txt, lbl, cand))
        for key in grouped_text.keys():
            for i in range(0, len(grouped_text[key]), mini_batch_size):
                subset = grouped_text[key][i: i + mini_batch_size]
                grouped_data_set.append(list(zip(*subset)))
        return grouped_data_set

    def save_model(self, model_path):
        checkpoint = {'nm': self.nm.state_dict(),
                      'pn': self.pn.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.nm.load_state_dict(checkpoint['nm'])
        self.pn.load_state_dict(checkpoint['pn'])

    def evaluate(self, dataloader, updates, mini_batch_size):

        self.nm.eval()
        self.pn.train()

        support_set = defaultdict(list)
        for _ in range(updates):
            text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
            support_set['text'].extend(text)
            support_set['label'].extend(label)
            support_set['candidates'].extend(candidates)

        support_set = self.group_by_relation(support_set, mini_batch_size)

        with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpn, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, label, candidates in support_set:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                         label,
                                                                                                         candidates)

                input_dict = self.pn.encode_text(list(zip(replicated_text, replicated_relations)))
                repr = fpn(input_dict, out_from='transformers')
                modulation = self.nm(input_dict)
                output = fpn(repr * modulation, out_from='linear')
                targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                loss = self.loss_fn(output, targets)

                diffopt.step(loss)
                pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(true_labels.tolist())

            acc = models.utils.calculate_accuracy(task_predictions, task_labels)

            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(support_loss), acc))

            all_losses, all_predictions, all_labels = [], [], []

            for text, label, candidates in dataloader:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                         label,
                                                                                                         candidates)
                with torch.no_grad():

                    input_dict = self.pn.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = fpn(input_dict, out_from='transformers')
                    modulation = self.nm(input_dict)
                    output = fpn(repr * modulation, out_from='linear')
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                loss = loss.item()
                pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(true_labels.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(all_losses), acc))

        return acc

    def training(self, train_datasets, **kwargs):
        n_episodes = kwargs.get('n_episodes')
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')

        if self.replay_rate != 0:
            replay_freq = int(math.ceil((1 - self.replay_rate) / (self.replay_rate * (updates + 1))))
        else:
            replay_freq = 0
        logger.info('Replay frequency: {}'.format(replay_freq))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=mini_batch_size, shuffle=False,
                                                collate_fn=datasets.utils.rel_encode))

        for episode_id in range(n_episodes):

            self.inner_optimizer.zero_grad()
            support_loss, support_acc = [], []

            with higher.innerloop_ctx(self.pn, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpn, diffopt):

                # Inner loop
                support_set = defaultdict(list)
                task_predictions, task_labels = [], []
                for _ in range(updates):
                    try:
                        text, label, candidates = next(train_dataloader)
                        support_set['text'].extend(text)
                        support_set['label'].extend(label)
                        support_set['candidates'].extend(candidates)
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                support_set = self.group_by_relation(support_set, mini_batch_size)

                for text, label, candidates in support_set:
                    replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                             label,
                                                                                                             candidates)

                    input_dict = self.pn.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = fpn(input_dict, out_from='transformers')
                    modulation = self.nm(input_dict)
                    output = fpn(repr * modulation, out_from='linear')
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                    diffopt.step(loss)
                    pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(true_labels.tolist())
                    self.memory.write_batch(text, label, candidates)

                acc = models.utils.calculate_accuracy(task_predictions, task_labels)

                logger.info('Episode {}/{} support set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
                                                                                                 n_episodes,
                                                                                                 np.mean(support_loss),
                                                                                                 acc))

                # Outer loop
                query_loss, query_acc = [], []
                query_set = []

                if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
                    text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
                    query_set.append((text, label, candidates))
                else:
                    try:
                        text, label, candidates = next(train_dataloader)
                        query_set.append((text, label, candidates))
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                for text, label, candidates in query_set:
                    replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                             label,
                                                                                                             candidates)

                    input_dict = self.pn.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = fpn(input_dict, out_from='transformers')
                    modulation = self.nm(input_dict)
                    output = fpn(repr * modulation, out_from='linear')
                    targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                    loss = self.loss_fn(output, targets)

                    query_loss.append(loss.item())
                    pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)

                    acc = models.utils.calculate_accuracy(pred.tolist(), true_labels.tolist())
                    query_acc.append(acc)

                    # NM meta gradients
                    nm_params = [p for p in self.nm.parameters() if p.requires_grad]
                    meta_nm_grads = torch.autograd.grad(loss, nm_params, retain_graph=True)
                    for param, meta_grad in zip(nm_params, meta_nm_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PN meta gradients
                    pn_params = [p for p in fpn.parameters() if p.requires_grad]
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

                logger.info('Episode {}/{} query set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
                                                                                               n_episodes,
                                                                                               np.mean(query_loss),
                                                                                               np.mean(query_acc)))

    def testing(self, test_dataset, **kwargs):
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')
        test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
        acc = self.evaluate(dataloader=test_dataloader, updates=updates, mini_batch_size=mini_batch_size)
        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
        return acc
