import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import get_scheduler
import random
import pandas as pd
random.seed(42)
from datasets import load_dataset
dataset = load_dataset("MP_IN_adm.csv")
print(dataset[0])
# batch_size = 1024

# t = pd.read_csv("MP_IN_adm.csv")
# id_list = t['id'].astype(int)  # id, int
# text_list = t['text'].astype(str).tolist()   # text, str
# label_list = t['hospital_expire_flag'].astype(int)  # hospital_expipre_flag, int

# # 直接将text进行tokenize，构造数据集
# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# dataset = [{"labels":label_list[i], "text":text_list[i]} for i in range(len(label_list))]
# def tokenize_function(examples):

#     return tokenizer(examples["text"], padding="max_length", truncation=True)

# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# print(tokenized_datasets[0])
# inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
# tokenized_datasets = [{"labels":label_list[i], "text":inputs[i]} for i in range(len(label_list))]

# split_index = int(len(label_list) * 0.6)

# # Split the list into two parts
# train_dataset = tokenized_datasets[:split_index]
# eval_dataset = tokenized_datasets[split_index:]

# # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)


# train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(eval_dataset, batch_size=8)
# print(next(iter(eval_dataloader)))
# class TextDataset(Dataset):
#     def __init__(self, id, text, flag):
#         self.id = id
#         self.text = text
#         self.flag = flag

#     def __len__(self):
#         if len(self.id) == len(self.text) == len(self.flag):
#             return len(self.text)

#     def __getitem__(self, index):
#         return self.id[index], self.text[index], self.flag[index]

# def onehot(x, num_classes=2):
#     one_hot_labels = np.zeros((len(x), num_classes))
#     one_hot_labels[np.arange(len(x)), x] = 1
#     return one_hot_labels



# # Mean Pooling - Take attention mask into account for correct averaging
# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # # Define a tokenizer to split the text into individual tokens
# # tokenizer = get_tokenizer('basic_english')
# # # tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# # # A function to yield individual tokens from the list of strings
# # def yield_tokens(data_list):
# #     for text in data_list:
# #         # yield tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# #         yield tokenizer(text)

# # data_list = text_list[:100]
# # # print(tokenizer(data_list, padding=True, truncation=True, return_tensors="pt")['input_ids'].shape)

# # # Build a vocabulary from the tokens in the list of strings
# # vocab = build_vocab_from_iterator(yield_tokens(data_list))

# # # A function to convert a string to a tensor of integers
# # def text_transform(text):
# #     # tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
# #     tokens = tokenizer(text)
# #     return torch.tensor([vocab[token] for token in tokens], dtype=torch.long)

# # # Convert each string in the list to a tensor of integers
# # data_tensors = [text_transform(text) for text in data_list]

# # # Combine the tensors into a single tensor
# # data_tensor = torch.cat(data_tensors)
# # print(data_tensor.shape)

# # # Create a dataset from the tensor
# # dataset = TensorDataset(data_tensor)

# # # Create a dataloader to iterate over the dataset
# # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # print(next(iter(dataloader)))

# label_list = onehot(label_list)

# tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# # model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

# sentences = text_list
# texts = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
# # texts = texts['input_ids']
# # print(texts.shape)
# dataset = TextDataset(id_list, texts, label_list)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# # print(next(iter(dataloader)))

# # id_list: 48684, texts: 48684x512, labels: 48684x2(one-hot)

# # Train
# num_labels = 2  # 类型数量
# LR = 1e-5
# num_epochs = 3

# num_training_steps = num_epochs * len(dataloader)

# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# criterion = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print("using device: ", device)
# model.to(device)
# model.train()

# print_interval = 1000
# for epoch in range(num_epochs):
#     print('---------------epoch {}---------------'.format(epoch))
#     total_loss = 0
#     i = 0
#     for batch in dataloader:
#         i += 1
#         _, x, y = batch
#         x = x.to(device)
#         y = y.to(device)

#         optimizer.zero_grad()
#         outputs = model(**x).logits
#         loss = criterion(outputs, y)
#         loss.backward()
#         total_loss+=loss.item()

#         optimizer.step()

#         lr_scheduler.step()
        
#         if i % print_interval == 0:
#             print("Batch {:3d} of total {:3d} || loss: {}".format(i, len(dataloader), loss.item()))   

#     print("Epoch {:3d} total loss : {}".format(epoch, total_loss))   

        
