import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets
import models.utils
from models.base_models import PlasticTransformerClsModel

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PlasticNetwork-Log')


class PlasticNetwork:

    def __init__(self, device, n_classes, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.device = device
        self.model = PlasticTransformerClsModel(model_name=kwargs.get('model'),
                                                n_classes=n_classes,
                                                max_length=kwargs.get('max_length'),
                                                device=device)
        logger.info('Loaded {} as model'.format(self.model.__class__.__name__))
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = AdamW([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, dataloader, log_freq):

        self.model.train()

        all_losses, all_predictions, all_labels = [], [], []
        iter = 0

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            repr = self.model(input_dict, out_from='transformers')
            self.update_hebbian(repr, labels)
            output = self.model(repr, out_from='linear')
            loss = self.loss_fn(output, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss = loss.item()
            pred = models.utils.make_prediction(output.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            iter += 1

            if iter % log_freq == 0:
                acc, prec, rec, f1 = models.utils.calculate_metrics(all_predictions, all_labels)
                logger.info(
                    'Metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))
                all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(self.device)
            input_dict = self.model.encode_text(text)
            with torch.no_grad():
                output = self.model(input_dict, out_from='full')
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
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 500)
        mini_batch_size = kwargs.get('mini_batch_size')
        if self.training_mode == 'sequential':
            for train_dataset in train_datasets:
                logger.info('Training on {}'.format(train_dataset.__class__.__name__))
                train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                                   collate_fn=datasets.utils.batch_encode)
                self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
        elif self.training_mode == 'multi_task':
            train_dataset = data.ConcatDataset(train_datasets)
            logger.info('Training multi-task model on all datasets')
            train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True,
                                               collate_fn=datasets.utils.batch_encode)
            self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)
        else:
            raise ValueError('Invalid training mode')

    def testing(self, test_datasets, **kwargs):
        mini_batch_size = kwargs.get('mini_batch_size')
        accuracies, precisions, recalls, f1s = [], [], [], []
        for test_dataset in test_datasets:
            logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
            test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                              collate_fn=datasets.utils.batch_encode)
            acc, prec, rec, f1 = self.evaluate(dataloader=test_dataloader)
            accuracies.append(acc)
            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                    'F1 score = {:.4f}'.format(np.mean(accuracies), np.mean(precisions), np.mean(recalls),
                                               np.mean(f1s)))
