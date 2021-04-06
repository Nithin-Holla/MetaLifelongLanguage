import re

import pandas as pd

from torch.utils import data

# MAX_TRAIN_SIZE = 115000
# MAX_TEST_SIZE = 7600

MAX_TRAIN_SIZE = 11500
MAX_TEST_SIZE = 760

def preprocess(text):
    """
    Preprocesses the text
    """
    text = text.lower()
    # removes '\n' present explicitly
    text = re.sub(r"(\\n)+", " ", text)
    # removes '\\'
    text = re.sub(r"(\\\\)+", "", text)
    # removes unnecessary space
    text = re.sub(r"(\s){2,}", u" ", text)
    # replaces repeated punctuation marks with single punctuation followed by a space
    # e.g, what???? -> what?
    text = re.sub(r"([.?!]){2,}", r"\1", text)
    # appends space to $ which will help during tokenization
    text = text.replace(u"$", u"$ ")
    # # replace decimal of the type x.y with x since decimal digits after '.' do not affect, e.g, 1.25 -> 1
    # text = re.sub(r"(\d+)\.(\d+)", r"\1", text)
    # removes hyperlinks
    text = re.sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text)
    return str(text)


class AGNewsDataset(data.Dataset):

    def __init__(self, file_path, split, reduce=False):

        self.data = pd.read_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/agn_{}.csv'.format(split))

        # self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
        #                         index_col=False)
        # self.data.dropna(inplace=True)
        # self.data['text'] = self.data['title'] + '. ' + self.data['description']
        # self.data['labels'] = self.data['labels'] - 1
        # self.data.drop(columns=['title', 'description'], inplace=True)
        # self.data['text'] = self.data['text'].apply(preprocess)
        #
        # self.data.to_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/agn_{}.csv'.format(split), index=False)



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
        self.data = pd.read_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/dbp_{}.csv'.format(split))
        # self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
        #                         index_col=False)
        #
        # self.data.dropna(inplace=True)
        # self.data['text'] = self.data['title'] + '. ' + self.data['description']
        # self.data['labels'] = self.data['labels'] - 1
        # self.data.drop(columns=['title', 'description'], inplace=True)
        # self.data['text'] = self.data['text'].apply(preprocess)
        # self.data.to_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/dbp_{}.csv'.format(split), index=False)

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
        self.data = pd.read_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/amz_{}.csv'.format(split))
        # self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'title', 'description'],
        #                         index_col=False)
        # self.data.dropna(inplace=True)
        # self.data['text'] = self.data['title'] + '. ' + self.data['description']
        # self.data['labels'] = self.data['labels'] - 1
        # self.data.drop(columns=['title', 'description'], inplace=True)
        # self.data['text'] = self.data['text'].apply(preprocess)
        # self.data.to_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/amz_{}.csv'.format(split), index=False)

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
        self.data = pd.read_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/yelp_{}.csv'.format(split))
        # self.data = pd.read_csv(file_path, header=None, sep=',', names=['labels', 'text'],
        #                         index_col=False)
        # self.data.dropna(inplace=True)
        # self.data['labels'] = self.data['labels'] - 1
        # self.data['text'] = self.data['text'].apply(preprocess)
        # self.data.to_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/yelp_{}.csv'.format(split), index=False)
        # print(len(self.data))
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
        self.data = pd.read_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/yahoo_ans_{}.csv'.format(split))
        #
        # self.data = pd.read_csv(file_path, header=None, sep=',',
        #                         names=['labels', 'question_title', 'question_content', 'best_answer'],
        #                         index_col=False)
        # self.data.dropna(inplace=True)
        # self.data['text'] = self.data['question_title'] + self.data['question_content'] + self.data['best_answer']
        # self.data['labels'] = self.data['labels'] - 1
        # self.data.drop(columns=['question_title', 'question_content', 'best_answer'], inplace=True)
        # self.data['text'] = self.data['text'].apply(preprocess)
        # self.data.to_csv('/Users/michael/UCL/Courses/NLP/MetaLifelongLanguage/clean/yahoo_ans_{}.csv'.format(split), index=False)

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
