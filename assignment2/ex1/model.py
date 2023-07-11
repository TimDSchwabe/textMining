import torch
import utils
from torch import nn

# repacking a sequence without indices sorts the sequence by length leading to losing the pairing of the sentences between the sent1 and sent2 tensors


def repack_sequence(data: torch.Tensor, sequence: nn.utils.rnn.PackedSequence) -> nn.utils.rnn.PackedSequence:
    return nn.utils.rnn.PackedSequence(data, sequence.batch_sizes, sequence.sorted_indices, sequence.unsorted_indices)


# helper class for GRUClassifier
class GRUForClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size,
                          batch_first=True, num_layers=num_layers)

    def forward(self, x_seq, hidden=None):
        x = self.embedding(x_seq.data)  # sequence to tensor
        x = repack_sequence(x, x_seq)  # tensor to sequence
        outputs, hidden = self.gru(x, hidden)
        return outputs, hidden


# main class for the simple RNN solution
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, linear_hidden_size, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru_sent1 = GRUForClassifier(
            vocab_size, embedding_size, hidden_size, num_layers=num_layers)
        self.gru_sent2 = GRUForClassifier(
            vocab_size, embedding_size, hidden_size, num_layers=num_layers)
        self.linear = nn.Sequential(nn.Linear(hidden_size*2, linear_hidden_size),
                                    nn.Linear(linear_hidden_size, 1)
                                    )

    def forward(self, sent1: nn.utils.rnn.PackedSequence, sent2: nn.utils.rnn.PackedSequence):
        sent1_outputs, sent1_hidden = self.gru_sent1(sent1)
        # use the hidden state of the first sentence as the initial hidden state of the second sentence
        sent2_outputs, _ = self.gru_sent2(sent2, sent1_hidden)
        sent1_outputs = nn.utils.rnn.unpack_sequence(sent1_outputs)
        sent2_outputs = nn.utils.rnn.unpack_sequence(sent2_outputs)
        # mean pooling over the sequence dimension
        sent1_outputs = torch.stack(
            [out.mean(dim=0, keepdim=False) for out in sent1_outputs])
        sent2_outputs = torch.stack(
            [out.mean(dim=0, keepdim=False) for out in sent2_outputs])
        # concatenate the outputs of the two sentences for the FFNN classifier
        outputs = torch.cat([sent1_outputs, sent2_outputs], dim=1)
        outputs = self.linear(outputs)
        return outputs


# attention mechanism for the GRUWithAttention class
class AdditiveAttention(nn.Module):
    def __init__(self, embedding_size, hidden_size, attention_size=32):
        super().__init__()
        self.hidden_size = hidden_size
        self.query_linear = nn.Linear(hidden_size, attention_size)
        self.tokens_linear = nn.Linear(embedding_size, attention_size)
        self.attention_linear = nn.Linear(attention_size, 1)

    def forward(self, tokens, query=None):
        # hidden N L H (embedding)
        # query N H (hidden)
        if query is None:
            query = torch.zeros(tokens.size(0), self.hidden_size).to(tokens.device)
        tokens_processed = self.tokens_linear(tokens)
        query = self.query_linear(query).unsqueeze(1)
        query = query.repeat(1, tokens.shape[1], 1)
        attention = self.attention_linear(torch.tanh(tokens_processed+query))
        attention = torch.softmax(attention, dim=1)
        result = torch.bmm(attention.transpose(1, 2), tokens_processed).squeeze()
        return result


# helper class for GRUWithAttention
class GRUWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, attention_size=16, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.attention = AdditiveAttention(
            embedding_size=embedding_size, hidden_size=hidden_size, attention_size=attention_size)
        self.gru = nn.GRU(attention_size, hidden_size,
                          num_layers=num_layers, batch_first=True)

    def forward(self, x_sequence, hidden_state=None):
        x = nn.utils.rnn.unpack_sequence(x_sequence)
        # remember how long each sequence is
        sequence_lengths = torch.LongTensor([len(sent)-1 for sent in x]).to(utils.get_device())
        # x.shape N L
        x = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=1)
        x = self.embedding(x)
        # x.shape N L H (embedding)
        gru_outputs = []
        # iterate over the sequence
        for i in range(x.shape[1]):
            attention_input = x[:, :i+1]
            if hidden_state is None:
                # first iteration use attention with a zero vector as the query
                input = self.attention(attention_input)
                hidden_state = torch.zeros(
                    self.gru.num_layers, x.shape[0], self.hidden_size).to(utils.get_device())
            else:
                # subsequent iterations use the top level hidden state of the GRU as the query
                input = self.attention(attention_input, hidden_state[-1])
            output, hidden_state = self.gru(input.unsqueeze(1), hidden_state)
            gru_outputs.append(output)
        gru_outputs = torch.stack([out.squeeze(1) for out in gru_outputs]) # L N H
        gru_outputs = gru_outputs.swapaxes(0, 1)  # N L H
        result = []
        # take only the GRU outputs at the index at which each sequence respectively ends
        for i, t, in enumerate(gru_outputs):
            result.append(t[sequence_lengths[i]-1])
        result = torch.stack(result)
        return result


# main class for the RNN with attention solution
class ClassifierWithAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, attention_size=16, num_layers=2):
        super().__init__()
        self.gru_sent1 = GRUWithAttention(
            vocab_size, embedding_size, hidden_size, attention_size, num_layers)
        self.gru_sent2 = GRUWithAttention(
            vocab_size, embedding_size, hidden_size, attention_size, num_layers)
        self.linear = nn.Linear(hidden_size*2, 1)

    def forward(self, sent1, sent2):
        sent1 = self.gru_sent1(sent1)
        sent2 = self.gru_sent2(sent2)
        embedding = torch.cat([sent1, sent2], dim=1)
        embedding = self.linear(embedding)
        return embedding
