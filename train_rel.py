import gc
import logging
import os
import random
from argparse import ArgumentParser
from datetime import datetime

import torchtext
import torch
import numpy as np

import datasets.utils
from datasets.lifelong_fewrel_dataset import LifelongFewRelDataset
from models.rel_agem import AGEM
from models.rel_anml import ANML
from models.rel_baseline import Baseline
from models.rel_maml import MAML
from models.rel_oml import OML
from models.rel_replay import Replay

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')


if __name__ == '__main__':

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--n_epochs', type=int, help='Number of epochs (only for MTL)', default=1)
    parser.add_argument('--lr', type=float, help='Learning rate (only for the baselines)', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='sequential')
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=4)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=1.0)
    parser.add_argument('--max_length', type=int, help='Maximum sequence length for the input', default=64)
    parser.add_argument('--seed', type=int, help='Random seed', default=42)
    parser.add_argument('--replay_rate', type=float, help='Replay rate from memory', default=0.01)
    parser.add_argument('--order', type=int, help='Number of task orders to run for', default=5)
    parser.add_argument('--num_clusters', type=int, help='Number of clusters to take', default=10)
    parser.add_argument('--replay_every', type=int, help='Number of data points between replay', default=1600)
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
    glove = torchtext.vocab.GloVe(name='6B', dim=300)
    logger.info('Finished loading GloVe vectors')

    # Get relation embeddings for clustering
    relation_embeddings = datasets.utils.get_relation_embedding(relation_names, glove)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate clusters
    relation_index = datasets.utils.get_relation_index(train_data)
    cluster_labels, relation_embeddings = datasets.utils.create_relation_clusters(args.num_clusters,
                                                                                  relation_embeddings, relation_index)

    # Validation dataset
    val_dataset = LifelongFewRelDataset(val_data, relation_names)

    # Run for different orders of the clusters
    accuracies = []
    for i in range(args.order):

        logger.info('Running order {}'.format(i + 1))

        # Initialize the model
        if args.learner == 'sequential':
            learner = Baseline(device=device, training_mode='sequential', **vars(args))
        elif args.learner == 'multi_task':
            learner = Baseline(device=device, training_mode='multi_task', **vars(args))
        elif args.learner == 'agem':
            learner = AGEM(device=device, **vars(args))
        elif args.learner == 'replay':
            learner = Replay(device=device, **vars(args))
        elif args.learner == 'maml':
            learner = MAML(device=device, **vars(args))
        elif args.learner == 'oml':
            learner = OML(device=device, **vars(args))
        elif args.learner == 'anml':
            learner = ANML(device=device, **vars(args))
        else:
            raise NotImplementedError
        logger.info('Using {} as learner'.format(learner.__class__.__name__))

        # Generate continual learning training data
        logger.info('Generating continual learning data')
        train_datasets = datasets.utils.prepare_rel_datasets(train_data, relation_names, cluster_labels, args.num_clusters)
        logger.info('Finished generating continual learning data')

        # Training
        logger.info('----------Training starts here----------')
        model_file_name = learner.__class__.__name__ + '-' + str(datetime.now()).replace(':', '-').replace(' ', '_') + '.pt'
        model_dir = os.path.join(base_path, 'saved_models')
        os.makedirs(model_dir, exist_ok=True)
        learner.training(train_datasets, **vars(args))
        learner.save_model(os.path.join(model_dir, model_file_name))
        logger.info('Saved the model with name {}'.format(model_file_name))

        # Testing
        logger.info('----------Testing starts here----------')
        acc = learner.testing(val_dataset, **vars(args))
        accuracies.append(acc)

        # Delete the model to free memory
        del learner
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info('Average accuracy across runs: {}'.format(np.mean(accuracies)))


