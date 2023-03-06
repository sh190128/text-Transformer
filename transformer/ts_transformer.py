# -*- coding: utf-8 -*-
# @Time    : 2023/2/13 19:37
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : ts_transformer.py
from typing import Callable

import torch
from torch import nn, Tensor
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.modules import TransformerEncoderLayer
from .layers import TransformerBatchNormEncoderLayer
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1, scale_factor=1.0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self, x):
        """
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        # x = x + Variable(self.pe[x.size(0):, :], requires_grad=False)
        return self.dropout(x)


class TSTClassifier(nn.Module):
    activation: Callable[[Tensor, bool], Tensor]

    def __init__(self, max_len, feat_dim, d_model, n_heads, n_ff, num_layers, num_classes,
                 dropout=0.1, activation='gelu', norm='BatchNorm'):
        super(TSTClassifier, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project = nn.Linear(feat_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        if norm == 'BatchNorm':
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, n_heads, dim_feedforward=n_ff,
                                                             dropout=dropout, activation=activation)
        else:
            encoder_layer = TransformerEncoderLayer(d_model, n_heads, dim_feedforward=n_ff,
                                                    dropout=dropout, activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if activation == 'gelu':
            self.activation = F.gelu
        else:
            self.activation = F.relu

        self.dropout = nn.Dropout(dropout)

        self.class_head = nn.Linear(max_len * d_model, num_classes)

    def forward(self, x, mask=None):
        """
       Args:
           x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
           mask: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
       Returns:
           output: (batch_size, num_classes)
        """
        # Encoder
        # [batch_size, seq_length, feat_dim] -> [seq_length, batch_size, feat_dim]
        x = x.permute(1, 0, 2)
        output = self.project(x) * math.sqrt(self.d_model)  # project input vectors to d_model dimensional space
        output = self.pos_enc(output)
        output = self.transformer_encoder(output, src_key_padding_mask=~mask)
        output = self.activation(output)
        output = output.permute(1, 0, 2)
        output = self.dropout(output)

        # Class Head
        # flatten [batch_size, seq_length * feat_dim]
        output = output.reshape(output.shape[0], -1)
        output = self.class_head(output)

        return output
