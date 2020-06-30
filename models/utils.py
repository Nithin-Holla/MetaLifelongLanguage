import torch
import numpy as np
from sklearn import metrics


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels)
    return accuracy, precision, recall, f1_score


def calculate_accuracy(predictions, labels):
    predictions = np.array(predictions)
    labels = np.array(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    return accuracy


def make_prediction(output):
    with torch.no_grad():
        if output.size(1) == 1:
            pred = (output > 0).int()
        else:
            pred = output.max(-1)[1]
    return pred


def make_rel_prediction(cosine_sim, ranking_label, relation_lengths):
    pred = []
    targets = []
    start_idx = 0
    with torch.no_grad():
        for rel_len in relation_lengths:
            subset = cosine_sim[start_idx: start_idx + rel_len]
            pos_idx = ranking_label[start_idx: start_idx + rel_len].index(1)
            targets.append(pos_idx)
            pred.append(torch.argmax(subset))
            start_idx += rel_len

    pred = torch.tensor(pred)
    targets = torch.tensor(targets)
    return pred, targets


def split_rel_scores(cosine_sim, ranking_label, relation_lengths):
    pos_scores, neg_scores = [], []
    start_idx = 0

    for rel_len in relation_lengths:
        pos_idx = ranking_label[start_idx: start_idx + rel_len].index(1) + start_idx
        for i in range(start_idx, start_idx + rel_len):
            if ranking_label[i] == -1:
                pos_scores.append(cosine_sim[pos_idx])
                neg_scores.append(cosine_sim[i])
        start_idx += rel_len

    pos_scores = torch.stack(pos_scores)
    neg_scores = torch.stack(neg_scores)
    return pos_scores, neg_scores
