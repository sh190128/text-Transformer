import numpy as np
from transformers import AutoTokenizer, AutoModel, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


t = pd.read_csv("MP_IN_adm.csv")
id_list = np.array(t['id'].astype(int))  # id, int
text_list = t['text'].astype(str).tolist()   # text, str
label_list = torch.tensor(t['hospital_expire_flag'].astype(int))   # hospital_expipre_flag, int

def onehot(x, num_classes=2):
    one_hot_labels = np.zeros((len(x), num_classes))
    one_hot_labels[np.arange(len(x)), x] = 1
    return one_hot_labels



# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# # Sentences we want sentence embeddings for
# sentences = text_list[:100]

# # Load model from HuggingFace Hub
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# # Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# # Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# # Perform pooling
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# # Normalize embeddings
# sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# print(sentence_embeddings.shape)



tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


sentences = text_list[:5]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
print(inputs)
print(inputs['input_ids'].shape)
print(inputs["attention_mask"].shape)

# with torch.no_grad():

#     logits = model(**inputs).logits

# print(logits)
# predicted_class_id = logits.argmax(dim=1)
# print(predicted_class_id)

# Train
num_labels = 2  # 类型数量

model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
in_features = model.classifier.in_features
classifier = nn.Sequential(
    nn.Linear(in_features=768, out_features=2, bias=True),
    nn.Softmax(dim=1)
)
model.classifier = classifier
# print(model)

out = model(**inputs).logits
print(out)
print(out.argmax(dim=1))
# labels转化为one-hot编码
labels = torch.FloatTensor(onehot(label_list[:5]))
print(labels)

criterion = torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
loss = criterion(out, labels)
loss.backward()
print(loss.item())