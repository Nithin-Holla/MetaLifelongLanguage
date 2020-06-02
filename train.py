import logging
import os
from argparse import ArgumentParser

import numpy as np

import torch
from torch import nn, optim
import torch.utils.data as data

import models.utils
import datasets.utils
from datasets.text_classification_dataset import AGNewsDataset, DBPediaDataset, AmazonDataset, YelpDataset, \
    YahooAnswersDataset
from models.base_models import AlbertClsModel


logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')


def train(dataloader, model, optimizer, loss_fn, device, n_epochs=1):
    log_freq = 500

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
                logger.info('Epoch {} metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                            'F1 score = {:.4f}'.format(epoch + 1, np.mean(all_losses), acc, prec, rec, f1))
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
    logger.info('Test metrics: Loss = {:.4f}, accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                'F1 score = {:.4f}'.format(np.mean(all_losses), acc, prec, rec, f1))


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=int, help='Dataset to choose', required=True)
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))

    torch.manual_seed(42)

    n_epochs = 1

    logger.info('Loading the dataset')

    if args.dataset == 1:
        train_path = os.path.join(base_path, '../data/ag_news_csv/train.csv')
        test_path = os.path.join(base_path, '../data/ag_news_csv/test.csv')
        train_dataset = AGNewsDataset(train_path, 'train', reduce=True)
        test_dataset = AGNewsDataset(test_path, 'test', reduce=True)
    elif args.dataset == 2:
        train_path = os.path.join(base_path, '../data/amazon_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/amazon_review_full_csv/test.csv')
        train_dataset = AmazonDataset(train_path, 'train', reduce=True)
        test_dataset = AmazonDataset(test_path, 'test', reduce=True)
    elif args.dataset == 3:
        train_path = os.path.join(base_path, '../data/yelp_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yelp_review_full_csv/test.csv')
        train_dataset = YelpDataset(train_path, 'train', reduce=True)
        test_dataset = YelpDataset(test_path, 'test', reduce=True)
    elif args.dataset == 4:
        train_path = os.path.join(base_path, '../data/dbpedia_csv/train.csv')
        test_path = os.path.join(base_path, '../data/dbpedia_csv/test.csv')
        train_dataset = DBPediaDataset(train_path, 'train', reduce=True)
        test_dataset = DBPediaDataset(test_path, 'test', reduce=True)
    elif args.dataset == 5:
        train_path = os.path.join(base_path, '../data/yahoo_answers_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yahoo_answers_csv/test.csv')
        train_dataset = YahooAnswersDataset(train_path, 'train', reduce=True)
        test_dataset = YahooAnswersDataset(test_path, 'test', reduce=True)

    train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=datasets.utils.batch_encode)
    test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=datasets.utils.batch_encode)
    logger.info('Finished loading the dataset')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlbertClsModel(n_classes=train_dataset.n_classes, max_length=128, device=device)
    logger.info('Initialized the model')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)

    logger.info('-------------Training starts here-------------')
    train(dataloader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device, n_epochs=n_epochs)

    logger.info('-------------Testing starts here-------------')
    evaluate(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
