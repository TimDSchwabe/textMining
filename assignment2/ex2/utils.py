import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import copy

class DataNER(Dataset):
    def __init__(self, sents, labels, corpus):
        self.sents = sents
        self.labels = labels
        self.word2index = word_2_index(corpus)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        sent = self.sents[idx]
        sent_idcs = [self.word2index[w] for w in sent if w in self.word2index]
        label = torch.LongTensor([self.labels[idx]])
        return torch.LongTensor(sent_idcs), label   

def word_2_index(sents):
    vocab = Counter()
    for sent in sents:
        for w in sent:
            vocab[w] += 1
    i = 1
    word2index = {}
    for w in vocab:
        if vocab[w] > 5:
            word2index[w] = i
            i += 1
    return word2index

def get_data(file):
    tags = []
    sentences = []
    with open(file, 'rb') as f:
        data = pickle.load(f)
        for line in data:
            sentences.append(line[0])
            tags.append(line[1])
    return sentences, tags

