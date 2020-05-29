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


if __name__ == '__main__':

    base_path = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_path, '../data/ag_news_csv/train.csv')
    test_path = os.path.join(base_path, '../data/ag_news_csv/test.csv')

    torch.manual_seed(42)

    train_dataset = AGNewsDataset(train_path)
    test_dataset = AGNewsDataset(test_path)
    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=datasets.utils.batch_encode)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=datasets.utils.batch_encode)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlbertClsModel(n_classes=4, max_length=128, device=device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)

    n_iter = 100
    iter = 0

    for text, labels in train_dataloader:
        labels = torch.tensor(labels).to(device)
        input_dict = model.encode_text(text)
        output = model(input_dict)
        loss = loss_fn(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()
        pred = models.utils.make_prediction(output.detach())
        acc, prec, rec, f1 = models.utils.calculate_metrics(pred.tolist(), labels.tolist())
        logger.info('Iter {}: Loss = {}, acc = {}, prec = {}, rec = {}, f1 = {}'.format(iter + 1, loss, acc, prec, rec, f1))
        iter += 1
        if iter == n_iter:
            break

    all_losses, all_predictions, all_labels = [], [], []
    for text, labels in test_dataloader:
        labels = torch.tensor(labels).to(device)
        input_dict = model.encode_text(text)
        with torch.no_grad():
            output = model(input_dict)
            loss = loss_fn(output, labels)
        loss = loss.item()
        all_losses.append(loss)
        pred = models.utils.make_prediction(output.detach())
        all_predictions.extend(pred.tolist())
        all_labels.extend(labels.tolist())

    acc, prec, rec, f1 = models.utils.calculate_metrics(all_labels, all_predictions)
    logger.info('Val: Loss = {}, acc = {}, prec = {}, rec = {}, f1 = {}'.format(np.mean(all_losses), acc, prec, rec, f1))
