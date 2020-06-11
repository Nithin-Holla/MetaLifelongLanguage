import random
import torch

from torch import nn
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer, BertModel


class TransformerClsModel(nn.Module):

    def __init__(self, model_name, n_classes, max_length, device):
        super(TransformerClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.linear = nn.Linear(768, n_classes)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         pad_to_max_length=True, return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out

    def forward_transformer(self, inputs):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return out

    def forward_linear(self, input):
        out = self.linear(input)
        return out


class TransformerRLN(nn.Module):

    def __init__(self, model_name, max_length, device):
        super(TransformerRLN, self).__init__()
        self.max_length = max_length
        self.device = device
        if model_name == 'albert':
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         pad_to_max_length=True, return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        return out


class LinearPLN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(LinearPLN, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.to(device)

    def forward(self, input):
        out = self.linear(input)
        return out


class ReplayMemory:

    def __init__(self, write_prob):
        self.buffer = []
        self.write_prob = write_prob

    def write(self, text, label):
        input_tuple = (text, label)
        if random.random() < self.write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        text, label = random.choice(self.buffer)
        return text, label

    def write_batch(self, text, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        for txt, lbl in zip(text, labels):
            self.write(txt, lbl)

    def read_batch(self, batch_size):
        text, label = [], []
        for _ in range(batch_size):
            txt, lbl = self.read()
            text.append(txt)
            label.append(lbl)
        return text, label

    def len(self):
        return len(self.buffer)
