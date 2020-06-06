import logging

import higher
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets
import models.utils
from models.base_models import TransformerRLN, LinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('OML-Log')


class OML:

    def __init__(self, device, n_classes, **kwargs):
        self.inner_lr = kwargs.get('inner_lr')
        self.meta_lr = kwargs.get('meta_lr')
        self.device = device

        self.rln = TransformerRLN(model_name=kwargs.get('model'),
                                  max_length=128,
                                  device=device)
        self.pln = LinearPLN(in_dim=768, out_dim=n_classes, device=device)
        self.loss_fn = nn.CrossEntropyLoss()

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

        meta_params = [p for p in self.rln.parameters() if p.requires_grad] + \
                      [p for p in self.pln.parameters() if p.requires_grad]
        self.meta_optimizer = AdamW(meta_params, lr=self.meta_lr)

        inner_params = [p for p in self.pln.parameters() if p.requires_grad]
        self.inner_optimizer = optim.SGD(inner_params, lr=self.inner_lr)

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.rln.eval()
        self.pln.eval()

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.rln.encode_text(text)
            with torch.no_grad():
                repr = self.rln(input_dict)
                output = self.pln(repr)
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
        batch_size = kwargs.get('batch_size')
        updates_per_task = kwargs.get('updates_per_task')
        train_dataloaders = [iter(data.DataLoader(dt, batch_size=32, shuffle=True,
                                                  collate_fn=datasets.utils.batch_encode)) for dt in train_datasets]

        for episode_id in range(n_episodes):

            self.inner_optimizer.zero_grad()
            support_loss, support_acc, support_prec, support_rec, support_f1 = [], [], [], [], []
            with higher.innerloop_ctx(self.pln, self.inner_optimizer,
                                      copy_initial_weights=False,
                                      track_higher_grads=True) as (fpln, diffopt):

                # Inner loop
                for task_dataloader in train_dataloaders:
                    task_losses, task_predictions, task_labels = [], [], []
                    for _ in range(updates_per_task):
                        text, labels = next(task_dataloader)
                        labels = torch.tensor(labels).to(self.device)
                        input_dict = self.rln.encode_text(text)
                        repr = self.rln(input_dict)
                        output = fpln(repr)
                        loss = self.loss_fn(output, labels)
                        diffopt.step(loss)
                        pred = models.utils.make_prediction(output.detach())
                        task_losses.append(loss.item())
                        task_predictions.extend(pred.tolist())
                        task_labels.extend(labels.tolist())

                    acc, prec, rec, f1 = models.utils.calculate_metrics(task_predictions, task_labels)
                    support_loss.append(np.mean(task_losses))
                    support_acc.append(acc)
                    support_prec.append(prec)
                    support_rec.append(rec)
                    support_f1.append(f1)

                logger.info('Episode {}/{} support set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                            'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1, n_episodes,
                                                                        np.mean(support_loss), np.mean(support_acc),
                                                                        np.mean(support_prec), np.mean(support_rec),
                                                                        np.mean(support_f1)))

                # Outer loop
                query_loss, query_acc, query_prec, query_rec, query_f1 = [], [], [], [], []
                for task_dataloader in train_dataloaders:
                    text, labels = next(task_dataloader)
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

                    logger.info('Episode {}/{} query set: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, '
                                'recall = {:.4f}, F1 score = {:.4f}'.format(episode_id + 1, n_episodes,
                                                                            np.mean(query_loss), np.mean(query_acc),
                                                                            np.mean(query_prec), np.mean(query_rec),
                                                                            np.mean(query_f1)))

                    # RLN meta gradients
                    rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                    meta_rln_grads = torch.autograd.grad(loss, rln_params, retain_graph=True)
                    for param, meta_grad in zip(rln_params, meta_rln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

                    # PLN meta gradients
                    pln_params = [p for p in fpln.parameters(time=0) if p.requires_grad]
                    meta_pln_grads = torch.autograd.grad(loss, pln_params)
                    pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                    for param, meta_grad in zip(pln_params, meta_pln_grads):
                        if param.grad is not None:
                            param.grad += meta_grad.detach()
                        else:
                            param.grad = meta_grad.detach()

            # Meta optimizer step
            if (episode_id + 1) % batch_size == 0:
                rln_params = [p for p in self.rln.parameters() if p.requires_grad]
                pln_params = [p for p in self.pln.parameters() if p.requires_grad]
                for param in rln_params + pln_params:
                    param.grad /= batch_size * len(train_dataloaders)
                self.meta_optimizer.step()
                self.meta_optimizer.zero_grad()

    def testing(self, test_datasets):
        accuracies, precisions, recalls, f1s = [], [], [], []
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
