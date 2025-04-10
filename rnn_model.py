import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.set_default_dtype(torch.float64)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_dim, output_size, num_layers, use_cuda, bidirectional,
                 dropout=(0.4, 0.0, 0.0)):
        super(LSTMClassifier, self).__init__()
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.dropout1 = nn.Dropout(p=dropout[0])
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=self.num_layers, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout[1])
        self.bn1 = nn.BatchNorm1d(hidden_dim * self.num_directions)
        self.dropout2 = nn.Dropout(p=dropout[2])
        self.hidden2out = nn.Linear(hidden_dim * self.num_directions, output_size)

    def disable_dropout(self):
        self.lstm.dropout = .0
        self.dropout1.p = .0
        self.dropout2.p = .0

    def init_hidden(self, batch_size):
        if self.use_cuda:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda(),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim,
                            dtype=torch.float64).cuda())
        else:
            return (
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim, dtype=torch.float64))

    def forward(self, batch, lengths=None):
        # Check if input is unbatched (2-D) or batched (3-D)
        if batch.dim() == 2:  # Unbatched: (sequence_length, input_size)
            batch = batch.unsqueeze(0)  # Add batch dim: (1, sequence_length, input_size)
            batch_size = 1
        elif batch.dim() == 3:  # Batched: (batch_size, sequence_length, input_size)
            batch_size = batch.size(0)
        else:
            raise ValueError("Input must be 2-D (unbatched) or 3-D (batched) tensor.")
        
        self.hidden = self.init_hidden(batch.size(0))
        batch = self.dropout1(batch)
        if lengths:
            # Ensure lengths is a list or tensor for pack_padded_sequence
            if batch_size == 1 and isinstance(lengths, int):
                lengths = [lengths]
            batch = pack_padded_sequence(batch, lengths, batch_first=True)
        output, self.hidden = self.lstm(batch, self.hidden)
        if lengths:
            output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.bn1(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = self.dropout2(output)
        output = self.hidden2out(output)
        return output