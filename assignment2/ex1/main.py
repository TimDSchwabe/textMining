import json
import os
import torch
import torch.nn as nn
from model import ClassifierWithAttention
from utils import WinogradDataset
import utils
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


train_data = WinogradDataset("WNLI/train.tsv")
val_data = WinogradDataset("WNLI/dev.tsv")
train_loader = DataLoader(train_data, batch_size=8,
                          shuffle=True, collate_fn=utils.collate)
val_loader = DataLoader(val_data, batch_size=256,
                        shuffle=True, collate_fn=utils.collate)

# model
largest_index = max(list(utils.word2index.values()))
# two different models are available, one simple RNN and one RNN with attention
model_settings = {
    "vocab_size": largest_index+1,
    "embedding_size": 16,
    "hidden_size": 32,
    "attention_size": 64,
    "num_layers": 2,
}
with open("model_settings.json", "w") as f:
    json.dump(model_settings, f)
# model = GRUClassifier(largest_index, 16, 16, 32, num_layers=2)
model = ClassifierWithAttention(**model_settings)
if os.path.exists("model.pt"):
    try:
        model.load_state_dict(torch.load("model.pt"))
    except:
        print("model loading failed")
model = model.to(utils.get_device())
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("parameter count: ", sum(p.numel()
      for p in model.parameters() if p.requires_grad))
minimal_loss = np.inf
for epoch in range(100):
    # validation
    with torch.no_grad():
        total_loss = 0
        total_acc = 0
        for batch in val_loader:
            sent1, sent2, label = batch
            output = model(sent1, sent2).squeeze()
            loss = criterion(output, label.float())
            total_loss += loss.item()
            total_acc += utils.accuracy(output, label.to(utils.get_device()))
        print(
            f"Epoch {epoch} \tLoss: {total_loss/len(val_loader):.4f} \tAcc: {(total_acc / len(val_loader))*100:.2f}%")
        if total_loss < minimal_loss:
            minimal_loss = total_loss
            torch.save(model.state_dict(), "model.pt")
    # training
    training_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        sent1, sent2, label = batch
        output = model(sent1, sent2).squeeze()
        loss = criterion(output, label.float())
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Train \t\tLoss: {training_loss/len(train_loader):.4f}")
