import logging
import os
import random
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


def get_dataset(base_path, dataset_id):
    if dataset_id == 0:
        train_path = os.path.join(base_path, '../data/ag_news_csv/train.csv')
        test_path = os.path.join(base_path, '../data/ag_news_csv/test.csv')
        train_dataset = AGNewsDataset(train_path, 'train', reduce=True)
        test_dataset = AGNewsDataset(test_path, 'test', reduce=True)
    elif dataset_id == 1:
        train_path = os.path.join(base_path, '../data/amazon_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/amazon_review_full_csv/test.csv')
        train_dataset = AmazonDataset(train_path, 'train', reduce=True)
        test_dataset = AmazonDataset(test_path, 'test', reduce=True)
    elif dataset_id == 2:
        train_path = os.path.join(base_path, '../data/yelp_review_full_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yelp_review_full_csv/test.csv')
        train_dataset = YelpDataset(train_path, 'train', reduce=True)
        test_dataset = YelpDataset(test_path, 'test', reduce=True)
    elif dataset_id == 3:
        train_path = os.path.join(base_path, '../data/dbpedia_csv/train.csv')
        test_path = os.path.join(base_path, '../data/dbpedia_csv/test.csv')
        train_dataset = DBPediaDataset(train_path, 'train', reduce=True)
        test_dataset = DBPediaDataset(test_path, 'test', reduce=True)
    elif dataset_id == 4:
        train_path = os.path.join(base_path, '../data/yahoo_answers_csv/train.csv')
        test_path = os.path.join(base_path, '../data/yahoo_answers_csv/test.csv')
        train_dataset = YahooAnswersDataset(train_path, 'train', reduce=True)
        test_dataset = YahooAnswersDataset(test_path, 'test', reduce=True)
    else:
        raise Exception('Invalid dataset ID')
    return train_dataset, test_dataset


def offset_labels(dataset):
    if isinstance(dataset, AmazonDataset) or isinstance(dataset, YelpDataset):
        offset_by = 0
    elif isinstance(dataset, AGNewsDataset):
        offset_by = 5
    elif isinstance(dataset, DBPediaDataset):
        offset_by = 5 + 4
    elif isinstance(dataset, YahooAnswersDataset):
        offset_by = 5 + 4 + 14
    dataset.data['labels'] = dataset.data['labels'] + offset_by
    return dataset


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

    return all_labels, all_predictions


if __name__ == '__main__':

    dataset_order_mapping = {
        1: [2, 0, 3, 1, 4],
        2: [3, 4, 0, 1, 2],
        3: [2, 4, 1, 3, 0],
        4: [0, 2, 1, 4, 3]
    }
    n_classes = 33

    parser = ArgumentParser()
    parser.add_argument('--order', type=int, help='Order of datasets', required=True)
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    n_epochs = 1

    logger.info('Loading the dataset')

    train_datasets, test_datasets = [], []
    for dataset_id in dataset_order_mapping[args.order]:
        train_dataset, test_dataset = get_dataset(base_path, dataset_id)
        logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        train_dataset = offset_labels(train_dataset)
        test_dataset = offset_labels(test_dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = AlbertClsModel(n_classes=n_classes, max_length=128, device=device)
    logger.info('Initialized the model')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=3e-5)

    logger.info('-------------Training starts here-------------')
    for train_dataset in train_datasets:
        logger.info('Training on {}'.format(train_dataset.__class__.__name__))
        train_dataloader = data.DataLoader(train_dataset, batch_size=32, shuffle=True,
                                           collate_fn=datasets.utils.batch_encode)
        train(dataloader=train_dataloader, model=model, optimizer=optimizer, loss_fn=loss_fn, device=device,
              n_epochs=n_epochs)

    logger.info('-------------Testing starts here-------------')
    labels_pool, predictions_pool = [], []
    for test_dataset in test_datasets:
        logger.info('Testing on {}'.format(test_dataset.__class__.__name__))
        test_dataloader = data.DataLoader(test_dataset, batch_size=32, shuffle=False,
                                          collate_fn=datasets.utils.batch_encode)
        labels, predictions = evaluate(dataloader=test_dataloader, model=model, loss_fn=loss_fn, device=device)
        labels_pool.extend(labels)
        predictions_pool.extend(predictions)

    acc, prec, rec, f1 = models.utils.calculate_metrics(labels_pool, predictions_pool)
    logger.info('Overall test metrics: Accuracy = {:.4f}, precision = {:.4f}, recall = {:.4f}, '
                'F1 score = {:.4f}'.format(acc, prec, rec, f1))
