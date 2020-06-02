import pandas as pd

from torch.utils import data

MAX_TRAIN_SIZE = 115000
MAX_TEST_SIZE = 7600


class AGNewsDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 4
        if reduce:
            if split == 'train':
                self.data = self.data.sample(n=MAX_TRAIN_SIZE, random_state=42)
            else:
                self.data = self.data.sample(n=MAX_TEST_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label


class DBPediaDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 14
        if reduce:
            if split == 'train':
                self.data = self.data.sample(n=MAX_TRAIN_SIZE, random_state=42)
            else:
                self.data = self.data.sample(n=MAX_TEST_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label


class AmazonDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 5
        if reduce:
            if split == 'train':
                self.data = self.data.sample(n=MAX_TRAIN_SIZE, random_state=42)
            else:
                self.data = self.data.sample(n=MAX_TEST_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label


class YelpDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'text'],
                                index_col=False)
        self.data['labels'] = self.data['labels'] - 1
        self.n_classes = 5
        if reduce:
            if split == 'train':
                self.data = self.data.sample(n=MAX_TRAIN_SIZE, random_state=42)
            else:
                self.data = self.data.sample(n=MAX_TEST_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label


class YahooAnswersDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',',
                                names=['labels', 'question_title', 'question_content', 'best_answer'],
                                index_col=False)
        self.data['text'] = self.data['question_title'] + self.data['question_content'] + self.data['best_answer']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['question_title', 'question_content', 'best_answer'], inplace=True)
        self.n_classes = 10
        if reduce:
            if split == 'train':
                self.data = self.data.sample(n=MAX_TRAIN_SIZE, random_state=42)
            else:
                self.data = self.data.sample(n=MAX_TEST_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label
