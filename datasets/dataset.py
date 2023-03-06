# -*- coding: utf-8 -*-
# @Time    : 2023/2/12 19:18
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os


def normalization(df):
    mean = df.mean()
    std = df.std()
    return (df - mean) / (std + np.finfo(float).eps)


class MIMICiiiDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        super(MIMICiiiDataset, self).__init__()
        self.root = os.path.join(root_dir, split)
        self.listfile = pd.read_csv(os.path.join(self.root, 'listfile.csv'))
        self.paths = [os.path.join(self.root, p) for p in self.listfile.loc[:, 'stay']]
        self.labels = self.listfile.loc[:, 'y_true']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feat = pd.read_csv(self.paths[index])
        feat = normalization(feat).values
        return torch.tensor(feat.astype(float)), torch.tensor(self.labels[index])


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def collate_fn(data, max_len=None):
    """
    data: len(batch_size) list of tuples (X, y).
        - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
        - y: torch tensor of shape (num_labels,) : class indices or numerical targets
    """
    batch_size = len(data)
    features, labels = zip(*data)

    lengths = [(X.shape[0]) for X in features]
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i]-1, max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                 max_len=max_len)  # (batch_size, padded_length)

    return X, targets, padding_masks


if __name__ == '__main__':
    data_root = "../data"
    train_dataset = MIMICiiiDataset(data_root, split='train')
    dataloader = DataLoader(train_dataset, batch_size=2, num_workers=4, shuffle=True,
                            collate_fn=lambda x: collate_fn(x, max_len=200))
    # print(train_dataset[35])
    # print(train_dataset[35][0].shape)
    for i, item in enumerate(dataloader):
        feat, label, mask = item
        print(feat)
        print(label)
        print(mask)
