from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch import nn
from transformers import AlbertModel, AlbertTokenizer


class AlbertClsModel(nn.Module):

    def __init__(self, n_classes, max_length, device):
        super(AlbertClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        self.albert = AlbertModel.from_pretrained('albert-base-v2')
        self.linear = nn.Linear(768, n_classes)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         pad_to_max_length=True, return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.albert(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out


class BertClsModel(nn.Module):

    def __init__(self, n_classes, max_length, device):
        super(BertClsModel, self).__init__()
        self.n_classes = n_classes
        self.max_length = max_length
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.albert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, n_classes)
        self.to(self.device)

    def encode_text(self, text):
        encode_result = self.tokenizer.batch_encode_plus(text, return_token_type_ids=False, max_length=self.max_length,
                                                         pad_to_max_length=True, return_tensors='pt')
        for key in encode_result:
            encode_result[key] = encode_result[key].to(self.device)
        return encode_result

    def forward(self, inputs):
        _, out = self.albert(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        out = self.linear(out)
        return out
