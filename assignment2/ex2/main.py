from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from utils import *
from model import RNN
import statistics

def train(model, train_loader, val_loader):
    epochs = 10
    val_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    for e in range(epochs):
        for X, y in tqdm(train_loader):
            output = model(X) # [<# words in sent>, 9]
            pred = []
            for i in output.detach().numpy():
                pred.append(np.argmax(i))
            pred = torch.Tensor(pred)
            pred.requires_grad = True
            y = y.float()

            loss = criterion(pred, y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        acc = validate(model, val_loader)
        print(f"Accuracy at the end of epoch {e}: {acc}")
        if acc > val_accuracy:
            torch.save(model.state_dict(), 'best_val_model.pt')
            val_accuracy = acc
        model.train()

def validate(model, val_loader):
    accs = []
    with torch.no_grad():
        for X, y in tqdm(val_loader):
            output = model(X)
            pred = []
            for i in output.detach().numpy():
                pred.append(np.argmax(i))
            pred = torch.Tensor(pred)
            y = y.float()
            y = y.reshape(-1)
            same = 0
            for i in range(pred.size(dim=0)):
                if pred[i].item() == y[i].item(): same += 1
            accs.append(same/pred.size(dim=0))

    return statistics.mean(accs)

if __name__ == "__main__":
    train_sents, train_tags = get_data('conll2003_train.pkl')
    val_sents, val_tags = get_data('conll2003_val.pkl')
    test_sents, test_tags = get_data('conll2003_test.pkl')
    sents_cpy = copy.deepcopy(train_sents)
    sents_cpy.extend(val_sents)
    sents_cpy.extend(test_sents)

    train_data = DataNER(train_sents, train_tags, sents_cpy)
    val_data = DataNER(val_sents, val_tags, sents_cpy)

    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)

    model = RNN(vocab_size=len(train_data.word2index))
    train(model, train_data, val_data)