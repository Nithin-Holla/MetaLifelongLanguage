import logging
import os
import random
from argparse import ArgumentParser

import numpy as np

import torch

import datasets.utils
from models.baseline import Baseline
from models.oml import OML

logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ContinualLearningLog')


if __name__ == '__main__':

    # Define the ordering of the datasets
    dataset_order_mapping = {
        1: [2, 0, 3, 1, 4],
        2: [3, 4, 0, 1, 2],
        3: [2, 4, 1, 3, 0],
        4: [0, 2, 1, 4, 3]
    }
    n_classes = 33

    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--order', type=int, help='Order of datasets', required=True)
    parser.add_argument('--lr', type=float, help='Learning rate', default=3e-5)
    parser.add_argument('--inner_lr', type=float, help='Inner-loop learning rate', default=0.001)
    parser.add_argument('--meta_lr', type=float, help='Meta learning rate', default=3e-5)
    parser.add_argument('--model', type=str, help='Name of the model', default='bert')
    parser.add_argument('--learner', type=str, help='Learner method', default='oml')
    parser.add_argument('--n_episodes', type=int, help='Number of meta-training episodes', default=1000)
    parser.add_argument('--batch_size', type=int, help='Batch size of tasks', default=1)
    parser.add_argument('--mini_batch_size', type=int, help='Batch size of data points within an episode', default=32)
    parser.add_argument('--updates', type=int, help='Number of inner-loop updates', default=5)
    parser.add_argument('--write_prob', type=float, help='Write probability for buffer memory', default=0.5)
    args = parser.parse_args()
    logger.info('Using configuration: {}'.format(vars(args)))

    # Set base path
    base_path = os.path.dirname(os.path.abspath(__file__))

    # Set random seed
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Load the datasets
    logger.info('Loading the datasets')
    train_datasets, test_datasets = [], []
    for dataset_id in dataset_order_mapping[args.order]:
        train_dataset, test_dataset = datasets.utils.get_dataset(base_path, dataset_id)
        logger.info('Loaded {}'.format(train_dataset.__class__.__name__))
        train_dataset = datasets.utils.offset_labels(train_dataset)
        test_dataset = datasets.utils.offset_labels(test_dataset)
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    logger.info('Finished loading all the datasets')

    # Load the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.learner == 'baseline':
        learner = Baseline(device=device, n_classes=n_classes, **vars(args))
    elif args.learner == 'oml':
        learner = OML(device=device, n_classes=n_classes, **vars(args))
    else:
        raise NotImplementedError
    logger.info('Using {} as learner'.format(learner.__class__.__name__))

    # Training
    logger.info('----------Training starts here----------')
    learner.training(train_datasets, **vars(args))

    # Testing
    logger.info('----------Testing starts here----------')
    learner.testing(test_datasets, **vars(args))
