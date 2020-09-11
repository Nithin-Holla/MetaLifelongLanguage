import torch
import numpy as np
from sklearn import metrics


def calculate_metrics(predictions, labels, binary=False):
    averaging = 'binary' if binary else 'macro'
    predictions = np.array(predictions)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    accuracy = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    recall = metrics.recall_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
    f1_score = metrics.f1_score(labels, predictions, average=averaging, labels=unique_labels, zero_division=0)
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


def make_rel_prediction(cosine_sim, ranking_label):
    pred = []
    with torch.no_grad():
        pos_idx = [i for i, lbl in enumerate(ranking_label) if lbl == 1]
        if len(pos_idx) == 1:
            pred.append(torch.argmax(cosine_sim))
        else:
            for i in range(len(pos_idx) - 1):
                start_idx = pos_idx[i]
                end_idx = pos_idx[i+1]
                subset = cosine_sim[start_idx: end_idx]
                pred.append(torch.argmax(subset))
    pred = torch.tensor(pred)
    true_labels = torch.zeros_like(pred)
    return pred, true_labels


def split_rel_scores(cosine_sim, ranking_label):
    pos_scores, neg_scores = [], []
    pos_index = 0
    for i in range(len(ranking_label)):
        if ranking_label[i] == 1:
            pos_index = i
        elif ranking_label[i] == -1:
            pos_scores.append(cosine_sim[pos_index])
            neg_scores.append(cosine_sim[i])
    pos_scores = torch.stack(pos_scores)
    neg_scores = torch.stack(neg_scores)
    return pos_scores, neg_scores
