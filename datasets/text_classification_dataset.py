import pandas as pd

from torch.utils import data


MAX_SIZE = 115000


class AGNewsDataset(data.Dataset):

    def __init__(self, file_path, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 4
        if reduce:
            self.data = self.data.sample(n=MAX_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label


class DBPediaDataset(data.Dataset):

    def __init__(self, file_path, reduce=False):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)
        self.n_classes = 14
        self.reduce = reduce
        if reduce:
            self.data = self.data.sample(n=MAX_SIZE, random_state=42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label
