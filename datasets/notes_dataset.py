# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 1:08
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : notes_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from transformers import AutoTokenizer


class NotesDataset(Dataset):
    def __init__(self, root_dir, split='train', max_length=2048):
        super(NotesDataset, self).__init__()
        self.root = root_dir
        self.max_length = max_length
        self.listfile = pd.read_csv(os.path.join(self.root, 'listfile', 'listfile_{}.csv'.format(split)))
        self.samples = [p for p in self.listfile.loc[:, 'stay']]
        self.labels = self.listfile.loc[:, 'y_true']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        id = self.samples[index].split('_')[0]
        notes = pd.read_csv(os.path.join(self.root, 'MP_IN_adm.csv'))
        note = notes.loc[notes['id'] == int(id), 'text'].iloc[0]
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.model_max_length = 2048
        tokens = tokenizer(text=note,
                           add_special_tokens=True,  # 给序列加上特殊符号，如[CLS],[SEP]
                           padding='max_length',  # 给序列补全到一定长度
                           max_length=self.max_length,
                           truncation=True,  # 截断操作
                           return_tensors='pt',  # 返回tensor
                           return_token_type_ids=False  # 不返回属于哪个句子
                           )
        input = tokens['input_ids'].squeeze(0)
        mask = tokens['attention_mask'].squeeze(0)

        return input, torch.tensor(self.labels[index]), mask.bool()


if __name__ == '__main__':
    data_root = "../data"
    dataset = NotesDataset(data_root, split='test')
    print(dataset[0][2].dtype)
    # dataloader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False)
    # max_len = 0
    # for i, item in enumerate(dataloader):
    #     input, label, mask = item
    #     if max_len < input.shape[0]:
    #         max_len = input.shape[0]
    # print(max_len)
