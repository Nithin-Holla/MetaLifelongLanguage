import pandas as pd

from torch.utils import data


class AGNewsDataset(data.Dataset):

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
                                index_col=False)
        self.data['text'] = self.data['title'] + self.data['description']
        self.data['labels'] = self.data['labels'] - 1
        self.data.drop(columns=['title', 'description'], inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data['text'].iloc[index]
        label = self.data['labels'].iloc[index]
        return text, label
