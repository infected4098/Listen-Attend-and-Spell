import torch
import torch.nn as nn
from util import CreateOnehotVariable
import numpy as np; import pandas as pd
from listener import Listener
from attention import MultiHeadAttention
from hearer import Hearer
import torch.nn.functional as F
rev_char_map = (pd.read_csv("C:/Users/infected4098/Desktop/LYJ/librispeech/train/idx2char.csv")
                .set_index("idx", inplace = True))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(156)


class Speller2(nn.Module):
    def __init__(self, output_class_dim, rnn_unit, rnn_hidden_dim, listener_hidden_dim,
                 n_heads, dimension, max_label_len):
        super(Speller2, self).__init__()
        # Listening
        self.rnn_unit = getattr(nn, rnn_unit) #LSTM
        self.listener_hidden_dim = listener_hidden_dim #128
        self.Listener = Listener(40,
                                 self.listener_hidden_dim, 3, "LSTM", 0.0).to(device)
        self.hearer = Hearer(hidden_dim = self.listener_hidden_dim).to(device)
        # Attention
        self.n_heads = n_heads #6
        self.dimension = dimension #6*64
        self.rnn_hidden_dim = rnn_hidden_dim #128
        self.MHA = MultiHeadAttention(self.n_heads, self.dimension,
                                      self.listener_hidden_dim, self.rnn_hidden_dim).to(device)

        # Decoding
        self.output_class_dim = output_class_dim #30
        self.speller_hidden_dim = self.dimension + self.rnn_hidden_dim # 열 방향 concat 할 것이므로.
        self.character_distribution = nn.Linear(self.speller_hidden_dim, self.output_class_dim)
        self.max_label_len = max_label_len
        self.softmax = nn.Softmax(dim = -1)
        self.under_rnn_layer = self.rnn_unit(self.output_class_dim, self.rnn_hidden_dim, num_layers = 1, batch_first = True)
        self.float_type = torch.cuda.FloatTensor
        self.upper_rnn_layer = nn.LSTM(self.speller_hidden_dim, self.speller_hidden_dim, num_layers = 1, batch_first = True)
    def forward_step(self, input_word, decoder_state, upper_decoder_state, listener_feature):
        """
        :param input_word: [batch_size, 1, output_class_dim]
        :param decoder_state: [batch_size, 1, self.rnn_hidden_dim] or None
        :param listener_feature: [batch_size, T/8 = U, listener_hidden_dim = 128]
        """

        rnn_output, hidden_state = self.under_rnn_layer(input_word, decoder_state)
        # rnn_output.shape = [batch_size, 1, rnn_hidden_dim]
        # hidden_state.shape = ([batch_size, 1, rnn_hidden_dim], [batch_size, 1, rnn_hidden_dim])
        context, p_attn = self.MHA(rnn_output, listener_feature)
        # context.shape = [batch_size, 1, dimension] #p_attn.shape = [batch_size, n_heads, 1, U]
        concats = torch.cat([rnn_output.squeeze(dim=1), context.squeeze(dim = 1)], dim=-1)
        # concats.shape = [batch_size, speller_hidden_dim]

        after_rnn, upper_hidden_state = self.upper_rnn_layer(concats, upper_decoder_state)
        # after_rnn.shape = [batch_size, speller_hidden_dim]

        output = F.softmax(self.character_distribution(after_rnn), dim = -1) # output.shape = [batch_size, output_class_dim]

        return output, hidden_state, upper_hidden_state, context, p_attn

    def decode_onestep(self, word_tensor, rev_char_map):
        # word_tensor.shape = [batch_size, 1, label_dim]
        word_tensor = word_tensor.squeeze(dim = 1)
        values = torch.argmax(word_tensor, dim = -1) # [batch_size, label_dim]
        return rev_char_map[values] # [batch_size, ]


    def forward(self, listener_tensor, ground_truth = None, teacher_force_rate = 0.95):
        """
        Set teacher_force_rate = 0.0 in order to move on to inference phase!!
        listener_tensor.shape = [batch_size, #timestep = T, feature_size = 40]
        ground_truth = [batch_size, #max_label_len, #output_class_dim]
        """

        listener_feature = self.Listener(listener_tensor) #[batch_size, T/8 = U, listener_hidden_dim = 128]
        #hearer_feature = self.hearer(listener_tensor) #[batch_size, T, listener]
        teacher_force = True if np.random.random_sample() < teacher_force_rate else False
        batch_size = listener_tensor.shape[0]

        init_word = CreateOnehotVariable(self.float_type(np.zeros((batch_size, 1))), self.output_class_dim).to(device)
        # init_word.shape = [batch_size, 1, label_dim]
        init_word[:, :, 0] = 1 # <sos> token 만들기
        hidden_state = None
        upper_hidden_state = None
        pred_lst = []
        attention_score_lst = []

        if teacher_force:  # Training with teacher forcing

            for step in range(self.max_label_len - 1):
                raw_pred, hidden_state, upper_hidden_state, context, p_attn = self.forward_step(init_word, hidden_state, upper_hidden_state, listener_feature)
                pred_lst.append(raw_pred)
                attention_score_lst.append(p_attn)

                output_word = ground_truth[:, step+1:step+2, :].type(self.float_type)
                init_word = output_word

        # shape = ([batch_size, max_label_len, output_class_dim],
        # [max_label_len, batch_size, n_heads, 1, dimensions/n_heads])

        else:  # Training without teacher forcing
            for step in range(self.max_label_len - 1):
                raw_pred, hidden_state, upper_hidden_state, context, p_attn = self.forward_step(init_word, hidden_state, upper_hidden_state, listener_feature)
                pred_lst.append(raw_pred)
                attention_score_lst.append(p_attn)
                output_word = raw_pred.unsqueeze(dim = 1) # [batch_size, 1, output_class_dim]
                init_word = output_word

        return torch.stack(pred_lst).transpose(1, 0), torch.stack(attention_score_lst)


