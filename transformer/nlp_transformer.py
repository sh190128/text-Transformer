# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 3:56
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : nlp_transformer.py
import torch
from torch import nn
from .ts_transformer import PositionalEncoding
from torch.nn import functional as F
from transformers import BertModel


class NLPClassifier(nn.Module):
    def __init__(self, max_len, d_model, n_heads, d_ff, num_layers, num_classes, dropout=0.1, activation='gelu'):
        super(NLPClassifier, self).__init__()
        self.embedding = nn.Embedding(32000, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_ff,
                                                dropout=dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu

        self.dropout = nn.Dropout(dropout)

        self.class_head = nn.Linear(max_len * d_model, num_classes)

    def forward(self, x, mask=None):
        output = self.embedding(x).permute(1, 0, 2)
        output = self.pos_enc(output)
        output = self.transformer_encoder(output, src_key_padding_mask=~mask)
        output = self.activation(output).permute(1, 0, 2)
        output = self.dropout(output)

        # Class Head
        output = output.reshape(output.shape[0], -1)
        output = self.class_head(output)

        return output


class BertClassifier(nn.Module):
    def __init__(self):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.class_head = nn.Linear(768, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask):
        output = self.bert(input_ids=x, attention_mask=mask)
        cls_hidden_state = output[0][:, 0, :]
        output = self.class_head(cls_hidden_state)

        return self.sigmoid(output)


if __name__ == '__main__':
    input = torch.tensor([[101, 345, 256, 67], [101, 23, 99, 897]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]])
    model = BertClassifier()
    output = model(input, mask)
    print(output)
