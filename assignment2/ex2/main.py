from torch.optim import Adam
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from utils import *
from model import RNN

def train(model, train_loader, val_loader):
    epochs = 10
    val_accuracy = 0
    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    for e in range(epochs):
        for X, y in tqdm(train_loader):
            #seq_ind = [word2index[w] for w in X if w in word2index]
            #seq_ind = torch.LongTensor(seq_ind)
            output = model(X)
            #output_lm = output[:-1,:]
            #labels = seq_ind[1:]
            #output_lm = torch.log_softmax(output_lm, dim=1)
            #loss = criterion(output_lm.squeeze(0), labels.squeeze(0))
            loss = criterion(output, y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        acc = validate(model, val_loader)
        print(f"Accuracy at the end of epoch - {e}: {acc}")
        if acc > val_accuracy:
            torch.save(model.state_dict(), 'best_val_model.pt')
            val_accuracy = acc
        model.train()

def validate(model, val_loader):
    pred = []
    truth = []
    with torch.no_grad():
        for X, y in val_loader:
            out = model(X)
            cls = torch.argmax(out, dim=1) # the class predicted by the model for the current val example
            pred.extend(cls.reshape(-1).numpy().tolist())
            truth.extend(y.reshape(-1).numpy().tolist())

    return accuracy_score(truth, pred)

if __name__ == "__main__":
    train_sents, train_tags = get_data('conll2003_train.pkl')
    val_sents, val_tags = get_data('conll2003_val.pkl')
    test_sents, test_tags = get_data('conll2003_test.pkl')

    sents_cpy = copy.deepcopy(train_sents)
    sents_cpy.extend(val_sents)
    sents_cpy.extend(test_sents)

    train_data = DataNER(train_sents, train_tags, sents_cpy)
    val_data = DataNER(val_sents, val_tags, sents_cpy)
    test_data = DataNER(test_sents, test_tags, sents_cpy)

    train_loader = DataLoader(train_data, batch_size=4)
    val_loader = DataLoader(val_data, batch_size=4)
    test_loader = DataLoader(test_data, batch_size=4)

    model = RNN(vocab_size=len(train_data.word2index))
    #print(train_data[3])
    #print(train_sents[3])
    train(model, train_loader, val_loader)