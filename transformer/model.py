# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 22:04
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : model.py

from einops import repeat
from torch import nn
import torch
from torch.nn import functional as F


class EHRTransformer(nn.Module):
    def __init__(self, max_len, feat_dim, d_model, n_heads, d_ff, num_layers, num_classes, dropout=0.1, activation='gelu'):
        super(EHRTransformer, self).__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.linear = nn.Linear(feat_dim, d_model)

        # Position Encoding
        W_pos = torch.empty((max_len + 1, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        self.W_pos = nn.Parameter(W_pos, requires_grad=True)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu

        self.dropout = nn.Dropout(dropout)

        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        batch_size = x.shape[0]
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=batch_size)
        output = torch.cat((cls_tokens, self.linear(x)), dim=1)
        output = self.dropout(output)

        output = self.encoder(output, src_key_padding_mask=~mask)[:, 0, :]
        output = self.activation(output)
        output = self.dropout(output)
        output = self.mlp_head(output)

        return output


if __name__ == '__main__':
    # print('a' == str)  # ???
    input = torch.tensor([[[0.1, 2, 25],
                           [0.2, 3, 29]],
                          [[0.8, 2, 27],
                           [0.5, 8, 30]]
                          ])
    mask = torch.zeros((2, 2))
    model = EHRTransformer(feat_dim=3, max_len=2, d_model=32, n_heads=8,
                           d_ff=256, num_layers=6, num_classes=2)
    output = model(input, mask)
    print(output)
