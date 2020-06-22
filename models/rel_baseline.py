import logging
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data

import datasets
import models.utils
from models.base_models import RelationLSTMRLN, LinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Baseline-Log')


class Baseline:

    def __init__(self, device, glove, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.device = device
        self.glove = glove
        if kwargs.get('model') == 'lstm':
            self.rln = RelationLSTMRLN(input_size=300,
                                       hidden_size=kwargs.get('hidden_size'),
                                       device=device)
            self.pln = LinearPLN(in_dim=2*kwargs.get('hidden_size'),
                                 out_dim=kwargs.get('hidden_size'),
                                 device=device)
            params = [p for p in self.rln.parameters() if p.requires_grad] + \
                     [p for p in self.pln.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(params, lr=self.lr)
        else:
            raise NotImplementedError
        logger.info('Loaded {} as model'.format(self.rln.__class__.__name__))
        self.loss_fn = nn.MarginRankingLoss(margin=kwargs.get('loss_margin'))
        self.cos = nn.CosineSimilarity(dim=1)

    def save_model(self, model_path):
        checkpoint = {'rln': self.rln.state_dict(),
                      'pln': self.pln.state_dict()}
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.rln.load_state_dict(checkpoint['rln'])
        self.pln.load_state_dict(checkpoint['pln'])

    def train(self, dataloader, n_epochs, log_freq):

        self.rln.train()
        self.pln.train()

        for epoch in range(n_epochs):
            all_losses, all_predictions, all_labels = [], [], []
            iter = 0

            for text, label, candidates in dataloader:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text, label,
                                                                                                         candidates)
                batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)

                batch_x = batch_x.to(self.device)
                batch_x_len = batch_x_len.to(self.device)
                batch_rel = batch_rel.to(self.device)
                batch_rel_len = batch_rel_len.to(self.device)

                x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                x_embed = self.pln(x_embed)
                rel_embed = self.pln(rel_embed)

                cosine_sim = self.cos(x_embed, rel_embed)
                pos_scores, neg_scores = models.utils.split_rel_scores(cosine_sim, ranking_label)

                loss = self.loss_fn(pos_scores, neg_scores, torch.ones(len(pos_scores), device=self.device))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss = loss.item()
                pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(targets.tolist())
                iter += 1

                if iter % log_freq == 0:
                    acc = models.utils.calculate_accuracy(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc))
                    all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.rln.eval()
        self.pln.eval()

        for text, label, candidates in dataloader:
            replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text, label,
                                                                                                     candidates)
            batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
            batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)

            batch_x = batch_x.to(self.device)
            batch_x_len = batch_x_len.to(self.device)
            batch_rel = batch_rel.to(self.device)
            batch_rel_len = batch_rel_len.to(self.device)

            with torch.no_grad():
                x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                x_embed = self.pln(x_embed)
                rel_embed = self.pln(rel_embed)
                cosine_sim = self.cos(x_embed, rel_embed)

            pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
            all_predictions.extend(pred.tolist())
            all_labels.extend(targets.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)

        return acc

    def training(self, train_datasets, **kwargs):
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 20)
        mini_batch_size = kwargs.get('mini_batch_size')
        for cluster_idx, train_dataset in enumerate(train_datasets):
            logger.info('Training on cluster {}'.format(cluster_idx + 1))
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=datasets.utils.rel_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)

    def testing(self, test_dataset, **kwargs):
        mini_batch_size = kwargs.get('mini_batch_size')
        test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
        acc = self.evaluate(dataloader=test_dataloader)

        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
