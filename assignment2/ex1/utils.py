from torch.utils.data import Dataset
import string
import nltk
import torch
import os
import pandas
import torch.nn as nn


def get_device():
    return "cpu"
    # return "cuda" if torch.cuda.is_available() else "cpu"


def accuracy(output, label):
    output = torch.sigmoid(output)
    output = torch.round(output)
    return (output == label).sum().item() / len(label)


def collate(batch):
    sent1 = [torch.tensor(item[0]) for item in batch]
    sent2 = [torch.tensor(item[1]) for item in batch]
    labels = [item[2] for item in batch]
    labels = torch.tensor(labels)
    sent1_packed = nn.utils.rnn.pack_sequence(sent1, enforce_sorted=False)
    sent2_packed = nn.utils.rnn.pack_sequence(sent2, enforce_sorted=False)
    return sent1_packed.to(get_device()), sent2_packed.to(get_device()), labels.to(get_device())


def text_preprocessing(text: str):
    # tokenize
    text = nltk.word_tokenize(text)
    # lower case
    text = [word.lower() for word in text]
    puncts = string.punctuation
    # remove punctuation
    text = ["".join([letter for letter in word if letter not in puncts])
            for word in text]
    # stemming
    stemmer = nltk.stem.WordNetLemmatizer()
    text = [stemmer.lemmatize(word) for word in text]
    return text


# read the entire corpus to build the vocabulary
text_data = []
for file in ['train', 'test', 'dev']:
    path = os.path.join('WNLI', f"{file}.tsv")
    table = pandas.read_csv(path, sep='\t')
    for column in ['sentence1', 'sentence2']:
        text_data.extend(table[column].tolist())
text_data = " ".join(text_data)
text_data = text_preprocessing(text_data)
word2index = {}
index2word = {}
# add start and end of sentence tokens
word2index["<SOS>"] = 0
index2word[0] = "<SOS>"
word2index["<EOS>"] = 1
index2word[1] = "<EOS>"
# add the rest of the words with index starting from 2 because 0 and 1 are taken
cur_idx = 2
for word in text_data:
    if word == "":
        continue
    if word not in word2index:
        word2index[word] = cur_idx
        index2word[cur_idx] = word
        cur_idx += 1
print(f"Vocab size: {len(word2index)}")


class WinogradDataset(Dataset):
    def __init__(self, path):
        self.path = path
        table = pandas.read_csv(path, sep='\t')
        self.sent1_data = []
        self.sent2_data = []
        for idx, row in table.iterrows():
            sent1, sent2 = row["sentence1"], row["sentence2"]
            # tokenize the sentences which transforms them from two strings to two lists of strings
            sent1, sent2 = text_preprocessing(sent1), text_preprocessing(sent2)
            # convert each token to a number via the vocabulary
            sent1 = [word2index[word] for word in sent1 if word != ""]
            sent2 = [word2index[word] for word in sent2 if word != ""]
            # add start and end of sentence tokens
            self.sent1_data.append([0]+sent1+[1])
            self.sent2_data.append([0]+sent2+[1])
        self.labels = table["label"].tolist()

    def __getitem__(self, index):
        # return the two sentences and the label
        return self.sent1_data[index], self.sent2_data[index], self.labels[index]

    def __len__(self):
        # ensure that the lengths of the lists are the same
        assert len(self.sent1_data) == len(self.sent2_data) and len(
            self.sent1_data) == len(self.labels)
        return len(self.sent1_data)
