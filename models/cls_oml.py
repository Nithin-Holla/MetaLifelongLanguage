import logging
import math

import higher
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets
import models.utils
from models.base_models import TransformerRLN, LinearPLN, ReplayMemory

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


class OML:

    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
        self.device = device

        self.rln = TransformerRLN(model_name=kwargs.get('model'),
                                  max_length=kwargs.get('max_length'),
                                  device=device)
        self.pln = LinearPLN(in_dim=768, out_dim=n_classes, device=device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=2)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

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

        support_set = []
        for _ in range(updates):
            text, labels = self.memory.read_batch(batch_size=mini_batch_size)
            support_set.append((text, labels))

        with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                  copy_initial_weights=False,
                                  track_higher_grads=False) as (fpln, diffopt):

            # Inner loop
            task_predictions, task_labels = [], []
            support_loss = []
            for text, labels in support_set:
                labels = torch.tensor(labels).to(self.device)
                input_dict = self.rln.encode_text(text)
                repr = self.rln(input_dict)
                output = fpln(repr)
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
                input_dict = self.rln.encode_text(text)
                with torch.no_grad():
                    repr = self.rln(input_dict)
                    output = fpln(repr)
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
        updates = kwargs.get('updates')
        mini_batch_size = kwargs.get('mini_batch_size')

        if self.replay_rate != 0:
            replay_batch_freq = self.replay_every // mini_batch_size
            replay_freq = int(math.ceil((replay_batch_freq + 1) / (updates + 1)))
            replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)
        else:
            replay_freq = 0
            replay_steps = 0
        logger.info('Replay frequency: {}'.format(replay_freq))
        logger.info('Replay steps: {}'.format(replay_steps))

        concat_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = iter(data.DataLoader(concat_dataset, batch_size=mini_batch_size, shuffle=False,
                                                collate_fn=datasets.utils.batch_encode))

        episode_id = 0
        while True:

            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []

            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=False) as (fpln, diffopt):

                # Inner loop
                support_set = []
                task_predictions, task_labels = [], []
                for _ in range(updates):
                    try:
                        text, labels = next(train_dataloader)
                        support_set.append((text, labels))
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                for text, labels in support_set:
                    # zero trace
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.rln.encode_text(text)
                    repr = self.rln(input_dict)
                    output = fpln(repr)

                    for numstep in range(params['nbsteps']):
                        y, hebb = net(Variable(inputs[numstep], requires_grad=False),
                                      Variable(labels[numstep], requires_grad=False), hebb)

                    loss = self.loss_fn(output, labels)
                    diffopt.step(loss)
                    pred = models.utils.make_prediction(output.detach())
                    support_loss.append(loss.item())
                    task_predictions.extend(pred.tolist())
                    task_labels.extend(labels.tolist())
                    self.memory.write_batch(text, labels)

                acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)

                logger.info('Episode {} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(support_loss), acc, prec, rec, f1))

                # Outer loop
                query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                query_set = []

                if self.replay_rate != 0 and (episode_id + 1) % replay_freq == 0:
                    for _ in range(replay_steps):
                        text, labels = self.memory.read_batch(batch_size=mini_batch_size)
                        query_set.append((text, labels))
                else:
                    try:
                        text, labels = next(train_dataloader)
                        query_set.append((text, labels))
                        self.memory.write_batch(text, labels)
                    except StopIteration:
                        logger.info('Terminating training as all the data is seen')
                        return

                for text, labels in query_set:
                    labels = torch.tensor(labels).to(self.device)
                    input_dict = self.rln.encode_text(text)
                    repr = self.rln(input_dict)
                    output = fpln(repr)
                    loss = self.loss_fn(output, labels)
                    query_loss.append(loss.item())
                    pred = models.utils.make_prediction(output.detach())

                    acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
                    query_acc.append(acc)
                    query_prec.append(prec)
                    query_rec.append(rec)
                    query_f1.append(f1)

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
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

                logger.info('Episode {} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1,
                                                                        np.mean(query_loss), np.mean(query_acc),
                                                                        np.mean(query_prec), np.mean(query_rec),
                                                                        np.mean(query_f1)))

                episode_id += 1

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

        return accuracies
