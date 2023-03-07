import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import get_scheduler
from eval_metrics import print_metrics_binary
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

t = pd.read_csv("MP_IN_adm.csv")
id_list = np.array(t['id'].astype(int))  # id, int
text_list = t['text'].astype(str).tolist()   # text, str
label_list = torch.tensor(t['hospital_expire_flag'].astype(int))   # hospital_expipre_flag, int


datasize = 2000
id_list = id_list[:datasize]
text_list = text_list[:datasize]
label_list = label_list[:datasize]

class TextDataset(Dataset):
    def __init__(self, id, inputs_ids, attention_mask, flag):
        self.id = id
        self.ids = inputs_ids
        self.mask = attention_mask
        self.flag = flag

    def __len__(self):
        return len(self.flag)

    def __getitem__(self, index):
        # inputs for model from "transformers" should be model(input_ids, attention_mask)

        # return "id" of patient, "input_ids" of tokenized texts, "attention_mask" of tokenized texts, "y_true" of patient
        return self.id[index], self.ids[index], self.mask[index], self.flag[index]


def onehot(x, num_classes=2):
    one_hot_labels = np.zeros((len(x), num_classes))
    one_hot_labels[np.arange(len(x)), x] = 1
    return one_hot_labels


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def split_len(list, ratio):
    length = len(list)
    split = int(length * ratio)
    return split

# 将y_true转化为one-hot编码
label_list = torch.FloatTensor(onehot(label_list))

# Train
batch_size = 64
num_labels = 2  # 类型数量
LR = 1e-3
num_epochs = 5

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
sentences = text_list
texts = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
"""
    texts = {"input_ids":[[][]...[]], "attention_mask":[[][]...[]]}
    data shape:
        id_list: 48684, 
        texts["input_ids"]: 48684x512, texts["attention_mask"]: 48684x512, 
        labels: 48684x2(one-hot)
"""
tr_len = split_len(id_list, 0.6)
va_len = tr_len + split_len(id_list, 0.2)

train_dataset = TextDataset(id_list[:tr_len], texts["input_ids"][:tr_len], texts["attention_mask"][:tr_len], label_list[:tr_len])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


val_dataset = TextDataset(id_list[tr_len:va_len], texts["input_ids"][tr_len:va_len], texts["attention_mask"][tr_len:va_len], label_list[tr_len:va_len])
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TextDataset(id_list[va_len:], texts["input_ids"][va_len:], texts["attention_mask"][va_len:], label_list[va_len:])
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



num_training_steps = num_epochs * len(train_loader)

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=2, bias=True),
    nn.Softmax(dim=1)
)
model.classifier = classifier

criterion = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("using device: ", device)
model.to(device)


print_interval = 1000
for epoch in range(num_epochs):
    model.train()
    print('---------------epoch {}---------------'.format(epoch+1))
    total_loss = 0
    i = 0
    for batch in train_loader:
        i += 1
        _, x_inputids, x_attentionmask, y = batch
        x_inputids = torch.Tensor(x_inputids).to(device)
        x_attentionmask = torch.Tensor(x_attentionmask).to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # 预训练模型需要输入tokenized后的input_ids和attention_mask
        outputs = model(x_inputids, attention_mask=x_attentionmask).logits

        loss = criterion(outputs, y)
        loss.backward()
        total_loss+=loss.item()

        optimizer.step()

        lr_scheduler.step()
        
        if i % print_interval == 0:
            print("Batch {:3d} of total {:3d} || loss: {}".format(i, len(dataloader), loss.item()))   

    print("Epoch: {} || train_loss: {:.4f}".format(epoch+1, total_loss))   

    # Validation
    epoch_losses = []
    y_pred = []
    y_true = []
    model.eval()

    with torch.no_grad():
        for batch in val_loader:

            _, x_inputids, x_attentionmask, y = batch
            x_inputids = torch.Tensor(x_inputids).to(device)
            x_attentionmask = torch.Tensor(x_attentionmask).to(device)
            y = y.to(device)

            outputs = model(x_inputids, attention_mask=x_attentionmask).logits
            loss = criterion(outputs, y)
            epoch_losses.append(loss.item())

            y_pred.extend(outputs.tolist())
            y = torch.argmax(y, dim=1)  # one-hot -> list
            y_true.extend(y.tolist())

            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eval_metrics = print_metrics_binary(y_true, y_pred, verbose=0)
    epoch_losses = np.array(epoch_losses)
    epoch_loss = epoch_losses.sum() / len(epoch_losses)
    print("Epoch: {} || val_loss: {:.4f} || val_acc: {:.4f}".format(epoch+1, epoch_loss, eval_metrics["acc"]))


# Test
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        _, x_inputids, x_attentionmask, y = batch
        x_inputids = torch.Tensor(x_inputids).to(device)
        x_attentionmask = torch.Tensor(x_attentionmask).to(device)
        y = y.to(device)

        outputs = model(x_inputids, attention_mask=x_attentionmask).logits

        pred = torch.argmax(outputs, dim=1)
        y_pred.extend(outputs.tolist())

        y = torch.argmax(y, dim=1)
        y_true.extend(y.tolist())

print("\nTEST METRICS:")
test_metric = print_metrics_binary(y_true, y_pred, verbose=1)
# print("Test Acc: {:.4f}".format(test_acc))