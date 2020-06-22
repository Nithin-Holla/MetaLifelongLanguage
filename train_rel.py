import logging
import os
import random
from argparse import ArgumentParser

import torchtext
import torch
import numpy as np

import datasets.utils
import models.utils
from datasets.lifelong_fewrel_dataset import LifelongFewRelDataset
from models.rel_baseline import Baseline

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (only for baseline)', default=3)
    parser.add_argument('--lr', type=float, help='Learning rate (only for baseline)', default=0.001)
    parser.add_argument('-hidden_size', type=int, help='Hidden size (only for LSTM model)', default=200)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='lstm')
    parser.add_argument('--learner', type=str, help='Learner method', default='sequential')
    parser.add_argument('--n_episodes', type=int, help='Number of meta-training episodes', default=10000)
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=32)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=0.5)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=128)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--loss_margin', type=float, help='Loss margin for ranking loss', default=0.5)
    args = parser.parse_args()
    logger.info('Using configuration: {}'.format(vars(args)))

    # Set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Load training and validation data
    logger.info('Loading the dataset')
    data_dir = os.path.join(base_path, '../data/LifelongFewRel')
    relation_file = os.path.join(data_dir, 'relation_name.txt')
    training_file = os.path.join(data_dir, 'training_data.txt')
    validation_file = os.path.join(data_dir, 'val_data.txt')
    relation_names = datasets.utils.read_relations(relation_file)
    train_data = datasets.utils.read_rel_data(training_file)
    val_data = datasets.utils.read_rel_data(validation_file)
    logger.info('Finished loading the dataset')

    # Load GloVe vectors
    logger.info('Loading GloVe vectors')
    glove = torchtext.vocab.GloVe(name='840B', dim=300)
    logger.info('Finished loading GloVe vectors')

    # Get relation embeddings for clustering
    relation_embeddings = datasets.utils.get_relation_embedding(relation_names, glove)

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.learner == 'sequential':
        learner = Baseline(device=device, glove=glove, **vars(args))
    else:
        raise NotImplementedError
    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Generate continual learning training data
    logger.info('Generating continual learning data')
    num_clusters = 10
    train_datasets = datasets.utils.prepare_rel_datasets(train_data, relation_names, relation_embeddings, num_clusters)
    val_dataset = LifelongFewRelDataset(val_data, relation_names)
    logger.info('Finished generating continual learning data')

    # Training
    logger.info('----------Training starts here----------')
    print(sum([t.__len__() for t in train_datasets]))
    learner.training(train_datasets, **vars(args))

    # Testing
    logger.info('----------Testing starts here----------')
    learner.testing(val_dataset, **vars(args))



