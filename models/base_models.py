import math
import random
from collections import defaultdict

import torch

from torch import nn
from torch.autograd import Variable
from torch.nn import init, Parameter
from torch.nn import functional as F
from transformers import AlbertModel, AlbertTokenizer, BertTokenizer, BertModel


class Plastic(nn.Module):
    '''Hebbian layer with weight decay (eta) as input - nerunomodulation of plasticity'''

    def __init__(self, in_features, out_features, activation=F.softmax):
        super().__init__()

        self.activation = activation

        # Regular weights
        self.w = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Just a bias term
        self.b = Parameter(.01 * torch.randn(1), requires_grad=True)
        # Plasticity coefficients
        self.alpha = Parameter(.01 * torch.randn(in_features, out_features), requires_grad=True)
        # Initialize hebbian trace
        self.trace = Variable(torch.zeros(in_features, out_features))

    def forward(self, x, eta, is_training=True):
        if is_training:
            self.reset_trace()
        output = torch.zeros(x.shape[0], self.w.shape[-1])
        for i, (x_in, eta_in) in enumerate(zip(x, eta)):
            eta_in = eta_in.reshape(self.w.shape)
            x_in = x_in.reshape(1, -1)
            x_out = self.activation(x_in.mm(self.w + torch.mul(self.alpha, self.trace)) + self.b)
            self.trace = (1 - eta_in) * self.trace + eta_in * torch.bmm(x_in.unsqueeze(2), x_out.unsqueeze(1))[0]
            self.trace = torch.clamp(self.trace, -1, 1)
            output[i] = x_out
        return output

    def reset_trace(self):
        self.trace = Variable(torch.zeros(self.w.shape))


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
        # self.linear = nn.Linear(768, n_classes)
        self.hebbian = Plastic(768, n_classes)

        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         truncation=True, padding='max_length', return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs, modulation, out_from='full', is_training=True):
        if out_from == 'full':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            out = self.hebbian(out, modulation, is_training)
        elif out_from == 'transformers':
            _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        elif out_from == 'linear':
            out = self.hebbian(inputs, modulation, is_training)
        else:
            raise ValueError('Invalid value of argument')
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
                                                         truncation=True, padding='max_length', return_tensors='pt')
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


class TransformerNeuromodulator(nn.Module):

    def __init__(self, n_classes, model_name, device):
        super(TransformerNeuromodulator, self).__init__()
        self.device = device
        self.n_classes = n_classes
        if model_name == 'albert':
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')
        elif model_name == 'bert':
            self.encoder = BertModel.from_pretrained('bert-base-uncased')
        else:
            raise NotImplementedError
        self.encoder.requires_grad = False
        self.linear = nn.Sequential(nn.Linear(768, 768),
                                    nn.ReLU(),
                                    nn.Linear(768, 768 * n_classes),
                                    nn.Sigmoid())
        self.to(self.device)

    def forward(self, inputs, out_from='full'):
        _, out = self.encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out


class ReplayMemory:

    def __init__(self, write_prob, tuple_size):
        self.buffer = []
        self.write_prob = write_prob
        self.tuple_size = tuple_size

    def write(self, input_tuple):
        if random.random() < self.write_prob:
            self.buffer.append(input_tuple)

    def read(self):
        return random.choice(self.buffer)

    def write_batch(self, *elements):
        element_list = []
        for e in elements:
            if isinstance(e, torch.Tensor):
                element_list.append(e.tolist())
            else:
                element_list.append(e)
        for write_tuple in zip(*element_list):
            self.write(write_tuple)

    def read_batch(self, batch_size):
        contents = [[] for _ in range(self.tuple_size)]
        for _ in range(batch_size):
            read_tuple = self.read()
            for i in range(len(read_tuple)):
                contents[i].append(read_tuple[i])
        return tuple(contents)

    def len(self):
        return len(self.buffer)

    def reset_memory(self):
        self.buffer = []
