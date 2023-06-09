import torch
import torch.nn as nn
from utils import *

class RNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.hidden_size = 32
        self.vocab_size = vocab_size
        self.emb_dim = 32
        self.emb = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)
        self.rnn = nn.RNN(self.emb_dim, self.hidden_size)
        self.lin = nn.Linear(self.hidden_size, 9)
        self.relu = nn.ReLU()
    
    def forward(self, inp_seq):
        inp = self.emb(inp_seq)
        h_0 = torch.rand(1, self.hidden_size).to(get_device())
        all_hidden_states, last_hidden_state = self.rnn(inp, h_0)
        out = self.lin(all_hidden_states)
        out = self.relu(out)
        return out
