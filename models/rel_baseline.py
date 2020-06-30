import logging
import torch
from torch import nn, optim

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets.utils
import models.utils
from models.base_models import RelationLSTMRLN, RelationLinearPLN, TransformerRLN, LinearPLN

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Baseline-Log')


class Baseline:

    def __init__(self, device, training_mode, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.device = device
        self.training_mode = training_mode
        self.model_type = kwargs.get('model')

        if self.model_type == 'lstm':
            self.glove = kwargs.get('glove')
            self.rln = RelationLSTMRLN(input_size=300,
                                       hidden_size=kwargs.get('hidden_size'),
                                       device=device)
            self.pln = RelationLinearPLN(in_dim=2 * kwargs.get('hidden_size'),
                                         out_dim=kwargs.get('hidden_size') // 4,
                                         device=device)
            params = [p for p in self.rln.parameters() if p.requires_grad] + \
                     [p for p in self.pln.parameters() if p.requires_grad]
            self.optimizer = optim.Adam(params, lr=self.lr)
        elif self.model_type == 'bert' or self.model_type == 'albert':
            self.rln = TransformerRLN(model_name=self.model_type,
                                      max_length=kwargs.get('max_length'),
                                      device=device)
            self.pln = LinearPLN(in_dim=768, out_dim=1, device=device)
            params = [p for p in self.rln.parameters() if p.requires_grad] + \
                     [p for p in self.pln.parameters() if p.requires_grad]
            self.optimizer = AdamW(params, lr=self.lr)
        else:
            raise NotImplementedError

        logger.info('Loaded {} as RLN'.format(self.rln.__class__.__name__))
        logger.info('Loaded {} as PLN'.format(self.pln.__class__.__name__))

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

                if self.model_type == 'lstm':
                    batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                    batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                    batch_x = batch_x.to(self.device)
                    batch_x_len = batch_x_len.to(self.device)
                    batch_rel = batch_rel.to(self.device)
                    batch_rel_len = batch_rel_len.to(self.device)
                    x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                    x_embed, rel_embed = self.pln(x_embed, rel_embed)
                    cosine_sim = self.cos(x_embed, rel_embed)

                else:
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = self.rln(input_dict)
                    cosine_sim = torch.sigmoid(self.pln(repr))

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

            with torch.no_grad():

                if self.model_type == 'lstm':

                    batch_x, batch_x_len = datasets.utils.glove_vectorize(replicated_text, self.glove)
                    batch_rel, batch_rel_len = datasets.utils.glove_vectorize(replicated_relations, self.glove)
                    batch_x = batch_x.to(self.device)
                    batch_x_len = batch_x_len.to(self.device)
                    batch_rel = batch_rel.to(self.device)
                    batch_rel_len = batch_rel_len.to(self.device)
                    x_embed, rel_embed = self.rln(batch_x, batch_x_len, batch_rel, batch_rel_len)
                    x_embed, rel_embed = self.pln(x_embed, rel_embed)
                    cosine_sim = self.cos(x_embed, rel_embed)

                else:
                    input_dict = self.rln.encode_text(list(zip(replicated_text, replicated_relations)))
                    repr = self.rln(input_dict)
                    cosine_sim = torch.sigmoid(self.pln(repr))

            pred, targets = models.utils.make_rel_prediction(cosine_sim, ranking_label)
            all_predictions.extend(pred.tolist())
            all_labels.extend(targets.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)

        return acc

    def training(self, train_datasets, **kwargs):
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 20)
        mini_batch_size = kwargs.get('mini_batch_size')
        if self.training_mode == 'sequential':
            for cluster_idx, train_dataset in enumerate(train_datasets):
                logger.info('Training on cluster {}'.format(cluster_idx + 1))
                train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                                   collate_fn=datasets.utils.rel_encode)
                self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
        elif self.training_mode == 'multi_task':
            train_dataset = data.ConcatDataset(train_datasets)
            logger.info('Training multi-task model on all datasets')
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=datasets.utils.rel_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)

    def testing(self, test_dataset, **kwargs):
        mini_batch_size = kwargs.get('mini_batch_size')
        test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
        acc = self.evaluate(dataloader=test_dataloader)

        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
