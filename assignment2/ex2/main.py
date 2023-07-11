from tqdm import tqdm
from torch.optim import AdamW
import torch
import torch.nn as nn
from utils import *
from model import RNN
import copy


def train(model, train_loader, val_loader):
    epochs = 10
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.0001)
    for e in range(epochs):
        acc, loss = validate(model, val_loader, criterion)
        print(f"Epoch {e}: \tAcc\t{acc*100:.2f}%\tLoss \t{loss:.6f}")
        model = train_step(model, train_loader, criterion, optimizer)
        if acc > best_acc:
            torch.save(model.state_dict(), 'best_val_model.pt')
            best_acc = acc


def train_step(model, train_loader, criterion, optimizer):
    model.train()
    for x, y in train_loader:
        output = model(x)  # [<# words in sent>, 9]
        loss = criterion(output, y.squeeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def validate(model, val_loader, criterion):
    model.eval()
    accs = []
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            output = model(x)
            losses.append(criterion(output, y.squeeze(0)))
            pred = torch.argmax(output, dim=1)
            y = y.float()
            y = y.squeeze(0)
            accs.append(torch.sum(pred == y) / len(y))
    return torch.tensor(accs).mean(), torch.tensor(losses).mean()


if __name__ == "__main__":
    train_sents, train_tags = get_data('conll2003_train.pkl')
    val_sents, val_tags = get_data('conll2003_val.pkl')
    test_sents, test_tags = get_data('conll2003_test.pkl')
    corpus = copy.deepcopy(train_sents)
    corpus.extend(val_sents)
    corpus.extend(test_sents)

    train_data = DataNER(train_sents, train_tags, corpus)
    val_data = DataNER(val_sents, val_tags, corpus)

    train_loader = DataLoader(train_data, batch_size=1,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=1, collate_fn=collate_fn)

    model = RNN(vocab_size=len(train_data.word2index))
    model = model.to(get_device())
    train(model, train_loader, val_loader)
