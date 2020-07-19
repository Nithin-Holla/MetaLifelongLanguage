import logging
import torch
from torch import nn

import numpy as np

from torch.utils import data
from transformers import AdamW

import datasets.utils
import models.utils
from models.base_models import TransformerClsModel, ReplayMemory

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Replay-Log')


class Replay:

    def __init__(self, device, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.device = device

        self.model = TransformerClsModel(model_name=kwargs.get('model'),
                                         n_classes=1,
                                         max_length=kwargs.get('max_length'),
                                         device=device)
        self.memory = ReplayMemory(write_prob=self.write_prob, tuple_size=3)
        logger.info('Loaded {} as the model'.format(self.model.__class__.__name__))

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(params, lr=self.lr)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def save_model(self, model_path):
        checkpoint = self.model.state_dict()
        torch.save(checkpoint, model_path)

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint)

    def train(self, dataloader, n_epochs, log_freq):

        self.model.train()

        for epoch in range(n_epochs):
            all_losses, all_predictions, all_labels = [], [], []
            iter = 0

            for text, label, candidates in dataloader:
                replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text, label,
                                                                                                         candidates)

                input_dict = self.model.encode_text(list(zip(replicated_text, replicated_relations)))
                output = self.model(input_dict)
                targets = torch.tensor(ranking_label).float().unsqueeze(1).to(self.device)
                loss = self.loss_fn(output, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                mini_batch_size = len(label)
                if self.replay_rate != 0 and (iter + 1) % int(1 / self.replay_rate) == 0:
                    ref_text, ref_label, ref_candidates = self.memory.read_batch(batch_size=mini_batch_size)
                    replicated_ref_text, replicated_ref_relations, ref_ranking_label = datasets.utils.replicate_rel_data(ref_text, ref_label, ref_candidates)
                    ref_input_dict = self.model.encode_text(list(zip(replicated_ref_text, replicated_ref_relations)))
                    ref_output = self.model(ref_input_dict)
                    ref_targets = torch.tensor(ref_ranking_label).float().unsqueeze(1).to(self.device)
                    ref_loss = self.loss_fn(ref_output, ref_targets)
                    self.optimizer.zero_grad()
                    ref_loss.backward()
                    self.optimizer.step()

                pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
                all_losses.append(loss)
                all_predictions.extend(pred.tolist())
                all_labels.extend(true_labels.tolist())
                iter += 1
                self.memory.write_batch(text, label, candidates)

                if iter % log_freq == 0:
                    acc = models.utils.calculate_accuracy(all_predictions, all_labels)
                    logger.info(
                        'Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc))
                    all_losses, all_predictions, all_labels = [], [], []

    def evaluate(self, dataloader):
        all_losses, all_predictions, all_labels = [], [], []

        self.model.eval()

        for text, label, candidates in dataloader:
            replicated_text, replicated_relations, ranking_label = datasets.utils.replicate_rel_data(text, label,
                                                                                                     candidates)

            with torch.no_grad():
                input_dict = self.model.encode_text(list(zip(replicated_text, replicated_relations)))
                output = self.model(input_dict)

            pred, true_labels = models.utils.make_rel_prediction(output, ranking_label)
            all_predictions.extend(pred.tolist())
            all_labels.extend(true_labels.tolist())

        acc = models.utils.calculate_accuracy(all_predictions, all_labels)

        return acc

    def training(self, train_datasets, **kwargs):
        n_epochs = kwargs.get('n_epochs', 1)
        log_freq = kwargs.get('log_freq', 20)
        mini_batch_size = kwargs.get('mini_batch_size')
        train_dataset = data.ConcatDataset(train_datasets)
        train_dataloader = data.DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=False,
                                           collate_fn=datasets.utils.rel_encode)
        self.train(dataloader=train_dataloader, n_epochs=n_epochs, log_freq=log_freq)

    def testing(self, test_dataset, **kwargs):
        mini_batch_size = kwargs.get('mini_batch_size')
        test_dataloader = data.DataLoader(test_dataset, batch_size=mini_batch_size, shuffle=False,
                                          collate_fn=datasets.utils.rel_encode)
        acc = self.evaluate(dataloader=test_dataloader)
        logger.info('Overall test metrics: Accuracy = {:.4f}'.format(acc))
        return acc
