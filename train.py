import logging
import os
import numpy as np

import torch
from torch import nn, optim
import torch.utils.data as data

import models.utils
import datasets.utils
from datasets.text_classification_dataset import AGNewsDataset
from models.base_models import AlbertClsModel


logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')


def train(dataloader, model, optimizer, loss_fn, device, n_epochs=1):
    log_freq = 1000

    model.train()

    for epoch in range(n_epochs):
        all_losses, all_predictions, all_labels = [], [], []
        iter = 0

        for text, labels in dataloader:
            labels = torch.tensor(labels).to(device)
            input_dict = model.encode_text(text)
            output = model(input_dict)
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            pred = models.utils.make_prediction(output.detach())
            all_losses.append(loss)
            all_predictions.extend(pred.tolist())
            all_labels.extend(labels.tolist())
            iter += 1

            if iter % log_freq == 0:
                acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
                logger.info('Epoch {}: Loss = {}, accuracy = {}, precision = {}, recall = {}, F1 score = {}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
                all_losses, all_predictions, all_labels = [], [], []


def evaluate(dataloader, model, loss_fn, device):
    all_losses, all_predictions, all_labels = [], [], []

    model.eval()

    for text, labels in dataloader:
        labels = torch.tensor(labels).to(device)
        input_dict = model.encode_text(text)
        with torch.no_grad():
            output = model(input_dict)
            loss = loss_fn(output, labels)
        loss = loss.item()
        pred = models.utils.make_prediction(output.detach())
        all_losses.append(loss)
        all_predictions.extend(pred.tolist())
        all_labels.extend(labels.tolist())

    acc, prec, rec, f1 = models.utils.calculate_metrics(all_labels, all_predictions)
    logger.info('Test metrics: Loss = {}, accuracy = {}, precision = {}, recall = {}, F1 score = {}'.format(np.mean(all_losses), acc, prec, rec, f1))


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '../data/ag_news_csv/train.csv')
    test_path = os.path.join(base_path, '../data/ag_news_csv/test.csv')

    torch.manual_seed(42)

    n_epochs = 1

    train_dataset = AGNewsDataset(train_path)
    test_dataset = AGNewsDataset(test_path)
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=datasets.utils.batch_encode)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=datasets.utils.batch_encode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlbertClsModel(n_classes=4, max_length=128, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)

    train(dataloader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device, n_epochs=n_epochs)
    evaluate(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
