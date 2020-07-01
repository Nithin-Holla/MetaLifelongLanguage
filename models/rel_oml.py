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
from models.base_models import ReplayMemory, RelationLSTMRLN, RelationLinearPLN, TransformerRLN, LinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


class OML:

    def __init__(self, device, glove, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.device = device
        self.glove = glove
        self.model_type = kwargs.get('model')

        if self.model_type == 'lstm':
            self.rln = RelationLSTMRLN(input_size=300,
                                       hidden_size=kwargs.get('hidden_size'),
                                       device=device)
            self.pln = RelationLinearPLN(in_dim=2 * kwargs.get('hidden_size'),
                                         out_dim=kwargs.get('hidden_size') // 4,
                                         device=device)
            meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                          [p for p in self.pln.parameters() if p.requires_grad]
            self.meta_optimizer = optim.Adam(meta_params, lr=self.meta_lr)
        elif self.model_type == 'bert' or self.model_type == 'albert':
            self.rln = TransformerRLN(model_name=self.model_type,
                                      max_length=kwargs.get('max_length'),
                                      device=device)
            self.pln = LinearPLN(in_dim=768, out_dim=1, device=device)
            meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                          [p for p in self.pln.parameters() if p.requires_grad]
            self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)
        else:
            raise NotImplementedError

        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=3)
        self.loss_fn = nn.MarginRankingLoss(margin=kwargs.get('loss_margin'))
        self.cos = nn.CosineSimilarity(dim=1)

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
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
        checkpoint = {'rln': self.rln.state_dict(),
                      'pln': self.pln.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])

    def evaluate(self, dataloader, updates, mini_batch_size):

        self.rln.eval()
        self.pln.train()

        support_set = defaultdict(list)
        for _ in range(updates):
            text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
            support_set['text'].extend(text)
            support_set['label'].extend(label)
            support_set['candidates'].extend(candidates)

        support_set = self.group_by_relation(support_set, mini_batch_size)

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, label, candidates in support_set:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                         label,
                                                                                                         candidates)

                if self.model_type == 'lstm':
                    batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                    batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                    batch_x = batch_x.to(self.device)
                    batch_x_len = batch_x_len.to(self.device)
                    batch_rel = batch_rel.to(self.device)
                    batch_rel_len = batch_rel_len.to(self.device)
                    x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                    x_embed, rel_embed = fpln(x_embed, rel_embed)
                    cosine_sim = self.cos(x_embed, rel_embed)

                else:
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = self.rln(input_dict)
                    cosine_sim = torch.tanh(fpln(repr))

                pos_scores, neg_scores = models.utils.split_rel_scores(cosine_sim, ranking_label)

                loss = self.loss_fn(pos_scores, neg_scores, torch.ones(len(pos_scores), device=self.device))
                diffopt.step(loss)
                pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
                support_loss.append(loss.item())
                task_predictions.extend(pred.tolist())
                task_labels.extend(targets.tolist())

            acc = models.utils.calculate_accuracy(task_predictions, task_labels)

            logger.info('Support set metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(support_loss), acc))

            all_losses, all_predictions, all_labels = [], [], []

            for text, label, candidates in dataloader:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                         label,
                                                                                                         candidates)
                with torch.no_grad():

                    if self.model_type == 'lstm':

                        batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                        batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                        batch_x = batch_x.to(self.device)
                        batch_x_len = batch_x_len.to(self.device)
                        batch_rel = batch_rel.to(self.device)
                        batch_rel_len = batch_rel_len.to(self.device)
                        x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                        x_embed, rel_embed = fpln(x_embed, rel_embed)
                        cosine_sim = self.cos(x_embed, rel_embed)

                    else:
                        input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                        repr = self.rln(input_dict)
                        cosine_sim = torch.tanh(fpln(repr))

                    pos_scores, neg_scores = models.utils.split_rel_scores(cosine_sim, ranking_label)

                    loss = self.loss_fn(pos_scores, neg_scores, torch.ones(len(pos_scores), device=self.device))

                loss = loss.item()
                pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(targets.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)
        logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(np.mean(all_losses), acc))

        return acc

    def training(self, train_datasets, **kwargs):
        n_episodes = kwargs.get('n_episodes')
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')

        replay_freq = int(max(1, math.ceil(1 / ((updates + 1) * self.replay_rate))))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=mini_batch_size, shuffle=False,
                                                collate_fn=datasets.utils.rel_encode))

        for episode_id in range(n_episodes):

            self.inner_optimizer.zero_grad()
            support_loss, support_acc = [], []

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

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
                    if self.model_type == 'lstm':
                        batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                        batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                        batch_x = batch_x.to(self.device)
                        batch_x_len = batch_x_len.to(self.device)
                        batch_rel = batch_rel.to(self.device)
                        batch_rel_len = batch_rel_len.to(self.device)
                        x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                        x_embed, rel_embed = fpln(x_embed, rel_embed)
                        cosine_sim = self.cos(x_embed, rel_embed)

                    else:
                        input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                        repr = self.rln(input_dict)
                        cosine_sim = torch.tanh(fpln(repr))

                    pos_scores, neg_scores = models.utils.split_rel_scores(cosine_sim, ranking_label)

                    loss = self.loss_fn(pos_scores, neg_scores, torch.ones(len(pos_scores), device=self.device))
                    diffopt.step(loss)
                    pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(targets.tolist())
                    self.memory.write_batch(text, label, candidates)

                acc = models.utils.calculate_accuracy(task_predictions, task_labels)

                logger.info('Episode {}/{} support set: Loss = {:.4f}, accuracy = {:.4f}'.format(episode_id + 1,
                                                                                                 n_episodes,
                                                                                                 np.mean(support_loss),
                                                                                                 acc))

                # Outer loop
                query_loss, query_acc = [], []
                query_set = []
                try:
                    text, label, candidates = next(train_dataloader)
                    query_set.append((text, label, candidates))
                except StopIteration:
                    logger.info('Terminating training as all the data is seen')
                    return

                if (episode_id + 1) % replay_freq == 0:
                    text, label, candidates = self.memory.read_batch(batch_size=mini_batch_size)
                    query_set.append((text, label, candidates))

                for text, label, candidates in query_set:
                    replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text,
                                                                                                             label,
                                                                                                             candidates)

                    if self.model_type == 'lstm':
                        batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                        batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                        batch_x = batch_x.to(self.device)
                        batch_x_len = batch_x_len.to(self.device)
                        batch_rel = batch_rel.to(self.device)
                        batch_rel_len = batch_rel_len.to(self.device)
                        x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                        x_embed, rel_embed = fpln(x_embed, rel_embed)
                        cosine_sim = self.cos(x_embed, rel_embed)

                    else:
                        input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                        repr = self.rln(input_dict)
                        cosine_sim = torch.tanh(fpln(repr))

                    pos_scores, neg_scores = models.utils.split_rel_scores(cosine_sim, ranking_label)

                    loss = self.loss_fn(pos_scores, neg_scores, torch.ones(len(pos_scores), device=self.device))
                    query_loss.append(loss.item())
                    pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)

                    acc = models.utils.calculate_accuracy(pred.tolist(), targets.tolist())
                    query_acc.append(acc)

                    # RLN meta gradients
                    rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                    meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True)
                    for param, meta_grad in zip(rln_params, meta_rln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PLN meta gradients
                    pln_params = [p for p in fpln.parameters() if p.requires_grad]
                    meta_pln_grads = torch.autograd.grad(loss, pln_params)
                    pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pln_params, meta_pln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                # Meta optimizer step
                rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                for param in rln_params + pln_params:
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
