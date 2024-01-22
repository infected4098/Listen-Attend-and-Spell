import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import torch.nn as nn

# BLSTM layer for pBLSTM
# Step 1. Reduce time resolution to half
# Step 2. Run through BLSTM
class pBLSTMLayer(nn.Module):

    def __init__(self, input_feature_dim, hidden_dim, rnn_unit='LSTM', dropout_rate=0.0):
        # input_feature_dim.shape = (batch_size, #timestep, #mel bin = input_feature_dim)
        super(pBLSTMLayer, self).__init__()
        self.rnn_unit = getattr(nn, rnn_unit.upper()) #nn.LSTM

        # feature dimension will be doubled since time resolution reduction
        self.BLSTM = self.rnn_unit(input_feature_dim, int(hidden_dim/4), 1, bidirectional = True,
                                   dropout = dropout_rate, batch_first = True) # output.shape = (batch_size, #timestep, #mel_bin)
        self.hidden_dim = hidden_dim

    def forward(self, input_x):
        batch_size = input_x.size(0)
        timestep = input_x.size(1)
        feature_dim = input_x.size(2)
        """ Reducing time resolution
        input_x = input_x.contiguous().view(batch_size, int(timestep / 2), feature_dim * 2) 
        """
        output, hidden = self.BLSTM(input_x) # output.shape = [batch_size, #timestep, hidden_dim/2]
        # hidden = (h_n, c_n). #h_n.shape = [batch_size, 2, hidden_dim]
        output = output.contiguous().view(batch_size, int(timestep/2), self.hidden_dim)
        # output.shape = [batch_size, int(timestep/2), hidden_dim]
        return output, hidden

# Listener is a pBLSTM stacking 3 layers to reduce time resolution 8 times
# Input shape should be [# of sample, timestep, features]
class Listener(nn.Module):
    def __init__(self, input_feature_dim, listener_hidden_dim, listener_layer, rnn_unit, dropout_rate=0.0,
                 **kwargs):
        super(Listener, self).__init__()
        # Listener RNN layer
        self.listener_layer = listener_layer
        assert self.listener_layer >= 1

        self.pLSTM_layer0 = pBLSTMLayer(input_feature_dim, listener_hidden_dim, rnn_unit=rnn_unit,
                                        dropout_rate=dropout_rate)

        for i in range(1, self.listener_layer):
            setattr(self, 'pLSTM_layer' + str(i),
                    pBLSTMLayer(listener_hidden_dim, listener_hidden_dim, rnn_unit=rnn_unit,
                                dropout_rate=dropout_rate))

    def forward(self, input_x):
        outputs,_ = self.pLSTM_layer0(input_x)
        for i in range(1, self.listener_layer):
            outputs, _ = getattr(self, "pLSTM_layer"+str(i))(outputs)
        return outputs
