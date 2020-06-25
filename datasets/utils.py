import os
import random
import re

import torch
from sklearn.cluster import KMeans

from datasets.lifelong_fewrel_dataset import LifelongFewRelDataset
from datasets.text_classification_dataset import AGNewsDataset, AmazonDataset, YelpDataset, DBPediaDataset, \
    YahooAnswersDataset


def batch_encode(batch):
    text, labels = [], []
    for txt, lbl in batch:
        text.append(txt)
        labels.append(lbl)
    return text, labels


def rel_encode(batch):
    text, label, candidate_relations = [], [], []
    for txt, lbl, cand in batch:
        text.append(txt)
        label.append(lbl)
        candidate_relations.append(cand)
    return text, label, candidate_relations


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


def remove_return_sym(str):
    return str.split('\n')[0]


def read_relations(relation_file):
    relation_list = ['fill']
    with open(relation_file, encoding='utf8') as file_in:
        for line in file_in:
            line = remove_return_sym(line)
            line = re.sub(r'/', '', line)
            line = line.split()
            relation_list.append(line)
    return relation_list


def read_rel_data(sample_file):
    sample_data = []
    with open(sample_file, encoding='utf8') as file_in:
        for line in file_in:
            items = line.split('\t')
            if len(items[0]) > 0:
                relation_ix = int(items[0])
                if items[1] != 'noNegativeAnswer':
                    candidate_ixs = [int(ix) for ix in items[1].split() if int(ix) != relation_ix]
                    sentence = remove_return_sym(items[2]).split()
                    sample_data.append([relation_ix, candidate_ixs, sentence])
    return sample_data


def get_relation_embedding(relations, glove):
    rel_embed = []
    for rel in relations:
        word_embed = glove.get_vecs_by_tokens(rel, lower_case_backup=True)
        if len(word_embed.shape) == 2:
            rel_embed.append(torch.mean(word_embed, dim=0))
        else:
            rel_embed.append(word_embed)
    rel_embed = torch.stack(rel_embed)
    return rel_embed


def create_relation_clusters(num_clusters, relation_embedding):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(relation_embedding[1:].numpy())
    labels = kmeans.labels_
    rel_embed = {}
    cluster_index = {}
    for i in range(len(labels)):
        cluster_index[i + 1] = labels[i]
        rel_embed[i + 1] = relation_embedding[i]
    return cluster_index, rel_embed


def split_rel_data_by_clusters(data_set, cluster_labels, num_clusters, shuffle_index):
    splitted_data = [[] for i in range(num_clusters)]
    for data in data_set:
        cluster_number = cluster_labels[data[0]]
        index_number = shuffle_index[cluster_number]
        splitted_data[index_number].append(data)
    return splitted_data


def remove_unseen_relations(dataset, seen_relations):
    cleaned_data = []
    for data in dataset:
        neg_cands = [cand for cand in data[1] if cand in seen_relations]
        if len(neg_cands) > 0:
            cleaned_data.append([data[0], neg_cands, data[2]])
        else:
            cleaned_data.append([data[0], data[1][-2:], data[2]])
    return cleaned_data


def get_max_len(text_list):
    return max([len(x) for x in text_list])


def glove_vectorize(text, glove, dim=300):
    max_len = get_max_len(text)
    lengths = []
    vec = torch.ones((len(text), max_len, dim))
    for i, sent in enumerate(text):
        sent_emb = glove.get_vecs_by_tokens(sent, lower_case_backup=True)
        vec[i, :len(sent_emb)] = sent_emb
        lengths.append(len(sent))
    lengths = torch.tensor(lengths)
    return vec, lengths


def replicate_rel_data(text, label, candidates):
    replicated_text = []
    replicated_relations = []
    ranking_label = []
    for i in range(len(text)):
        replicated_text.append(text[i])
        replicated_relations.append(label[i])
        ranking_label.append(1)
        for j in range(len(candidates[i])):
            replicated_text.append(text[i])
            replicated_relations.append(candidates[i][j])
            ranking_label.append(-1)
    return replicated_text, replicated_relations, ranking_label


def prepare_rel_datasets(train_data, relation_names, relation_embeddings, num_clusters):
    train_datasets = []
    cluster_labels, relation_embeddings = create_relation_clusters(num_clusters, relation_embeddings)

    shuffle_index = list(range(num_clusters))
    random.shuffle(shuffle_index)

    splitted_train_data = split_rel_data_by_clusters(train_data, cluster_labels, num_clusters, shuffle_index)
    seen_relations = []

    for i in range(num_clusters):
        for data_entry in splitted_train_data[i]:
            if data_entry[0] not in seen_relations:
                seen_relations.append(data_entry[0])

        current_train_data = remove_unseen_relations(splitted_train_data[i], seen_relations)

        train_dataset = LifelongFewRelDataset(current_train_data, relation_names)
        train_datasets.append(train_dataset)
    return train_datasets
