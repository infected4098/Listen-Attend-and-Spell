import torch
import torch.nn as nn

class Hearer(nn.Module):
    def __init__(self, input_feature_dim = 40, hidden_dim = 256, num_layers = 2, dropout = 0.0,
                 bidirectional = True, rnn_type = "LSTM"):
        super(Hearer, self).__init__()
        self.input_feature_dim = input_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional =  bidirectional
        self.rnn_type = rnn_type

        self.lstm = nn.LSTM(input_size = self.input_feature_dim, hidden_size = self.hidden_dim,
                            num_layers = self.num_layers, batch_first = True, dropout = self.dropout,
                            bidirectional = self.bidirectional)

    def forward(self, input):
        output, _ = self.lstm(input)

        return output



