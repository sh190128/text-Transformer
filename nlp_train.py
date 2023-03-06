# -*- coding: utf-8 -*-
# @Time    : 2023/2/27 4:10
# @Author  : xsy
# @Project ：TST-mimic 
# @File    : nlp_train.py

import torch
from torch import nn
from tqdm import tqdm
import os
from torch.utils.data import DataLoader, random_split

from datasets.notes_dataset import NotesDataset
from transformer.nlp_transformer import NLPClassifier, BertClassifier

from utils import config

# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = [0, 1]


def train(opt):
    # dataset split
    dataset = NotesDataset(opt.data_path, split='train', max_length=opt.max_len)
    n_val = int(len(dataset) * opt.val_ratio)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # dataloader
    train_loader = DataLoader(train_set,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              num_workers=opt.num_workers,
                              )

    val_loader = DataLoader(val_set,
                            batch_size=opt.batch_size,
                            shuffle=False,
                            num_workers=opt.num_workers,
                            )

    # model
    # model = NLPClassifier(opt.max_len, opt.d_model, opt.n_heads, opt.d_ff, opt.num_layers,
    #                       opt.num_classes, opt.dropout, opt.activation)
    model = BertClassifier()
    model = nn.DataParallel(model, device_ids=device_ids).to(device)

    # loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-12)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1 / (epoch + 1), verbose=True)

    best_acc = 0
    for epoch in range(opt.epoch):
        # 训练
        model.train()
        epoch_loss = 0
        total_samples = 0
        pbar = tqdm((train_loader), total=len(train_loader))
        for item in pbar:
            feat, label, mask = item
            feat = feat.to(device)
            label = label.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            output = model(feat, mask)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_samples += 1
                epoch_loss += loss.item()

            pbar.set_description(f'Epoch [{epoch+1}/{opt.epoch}]')
            pbar.set_postfix(loss=loss.item())
        scheduler.step()
        epoch_loss = epoch_loss / total_samples
        print("Train:loss={}".format(epoch_loss))
        # 验证
        model.eval()
        epoch_loss = 0
        total_samples = 0
        total = 0
        correct = 0
        for i, item in enumerate(val_loader):
            feat, label, mask = item
            feat = feat.to(device)
            label = label.to(device)
            mask = mask.to(device)
            output = model(feat, mask)
            # loss
            loss = criterion(output, label)
            total_samples += 1
            epoch_loss += loss.item()
            # acc
            _, pred = torch.max(output, dim=1)
            total += label.size(0)
            correct += (pred == label).sum()
        epoch_loss = epoch_loss / total_samples
        val_acc = correct / total
        print("Val:loss={} | acc={}".format(epoch_loss, val_acc))
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'weights/nlp-{}.pth'.format(epoch+1))


if __name__ == '__main__':
    args = config.config()
    train(args)
