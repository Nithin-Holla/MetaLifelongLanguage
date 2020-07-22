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
logger = logging.getLogger('AGEM-Log')


class AGEM:

    def __init__(self, device, **kwargs):
        self.lr = kwargs.get('lr', 3e-5)
        self.write_prob = kwargs.get('write_prob')
        self.replay_rate = kwargs.get('replay_rate')
        self.replay_every = kwargs.get('replay_every')
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

    def compute_grad(self, orig_grad, ref_grad):
        with torch.no_grad():
            flat_orig_grad = torch.cat([torch.flatten(x) for x in orig_grad])
            flat_ref_grad = torch.cat([torch.flatten(x) for x in ref_grad])
            dot_product = torch.dot(flat_orig_grad, flat_ref_grad)
            if dot_product >= 0:
                return orig_grad
            proj_component = dot_product / torch.dot(flat_ref_grad, flat_ref_grad)
            modified_grad = [o - proj_component * r for (o, r) in zip(orig_grad, ref_grad)]
            return modified_grad

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

                params = [p for p in self.model.parameters() if p.requires_grad]
                orig_grad = torch.autograd.grad(loss, params)

                mini_batch_size = len(label)
                replay_freq = self.replay_every // mini_batch_size
                replay_steps = int(self.replay_every * self.replay_rate / mini_batch_size)

                if self.replay_rate != 0 and (iter + 1) % replay_freq == 0:
                    ref_grad_sum = None
                    for _ in range(replay_steps):
                        ref_text, ref_label, ref_candidates = self.memory.read_batch(batch_size=mini_batch_size)
                        replicated_ref_text, replicated_ref_relations, ref_ranking_label = datasets.utils.replicate_rel_data(ref_text, ref_label, ref_candidates)
                        ref_input_dict = self.model.encode_text(list(zip(replicated_ref_text, replicated_ref_relations)))
                        ref_output = self.model(ref_input_dict)
                        ref_targets = torch.tensor(ref_ranking_label).float().unsqueeze(1).to(self.device)
                        ref_loss = self.loss_fn(ref_output, ref_targets)
                        ref_grad = torch.autograd.grad(ref_loss, params)
                        if ref_grad_sum is None:
                            ref_grad_sum = ref_grad
                        else:
                            ref_grad_sum = [x + y for (x, y) in zip(ref_grad, ref_grad_sum)]
                    final_grad = self.compute_grad(orig_grad, ref_grad_sum)
                else:
                    final_grad = orig_grad

                for param, grad in zip(params, final_grad):
                    param.grad = grad.data

                self.optimizer.step()

                loss = loss.item()
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
