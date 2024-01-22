import torch
import math
if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from util import TimeDistributed, CreateOnehotVariable
import numpy as np


class MultiHeadAttention(nn.Module):
    # Input[0] : decoder_state                     [batch size, 1, decoder hidden dimension]
    # Input[1]       listener_feature                  [batch size, timestep/8 = U, listener feature dimension]
    def __init__(self,  heads = 6, dimension = 6*64, listener_hidden_dim = 128, decoder_hidden_dim = 128):
        super(MultiHeadAttention, self).__init__()
        self.h = heads
        self.d = dimension
        assert self.d % self.h == 0 #마지막에 concat을 해야 하니까 나눠서 0이 될 수 있어야 하는 것.
        self.d_v = self.d // self.h #각 head의 attention output dimension -> 64
        self.listener_hidden_dim = listener_hidden_dim #128
        self.decoder_hidden_dim = decoder_hidden_dim
        self.W_q = nn.Linear(self.decoder_hidden_dim, self.d)
        self.W_k = nn.Linear(self.listener_hidden_dim, self.d)
        self.W_v = nn.Linear(self.listener_hidden_dim, self.d)
        self.W_o = nn.Linear(self.d, self.d)
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, query, key, value):
        d_k = query.size(-1) #decoder_hidden_dimension
            # query.shape = [batch_size, h, 1, d_v]
            # key, value.shape = [batch_size, h, U, d_v]
            # scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, h, 1, U]
        p_attn = scores.softmax(dim=-1)  # [batch_size, h, 1, U]

        return torch.matmul(p_attn, value), p_attn  # [batch_size, h, 1, d_v], #[batch_size, h, 1, U]

    def forward(self, decoder_state, listener_feature):

        comp_decoder_state = decoder_state #[batch_size, 1, decoder_hidden_dimension = 6*64)
        comp_listener_feature = listener_feature #[batch_size, timestep/8 = U, listener feature dimension = 128]
        num_batches = decoder_state.size(0)  # batch_size

        # query.shape = [batch_size, 1, d] -> [batch_size, 1, h, d_v] -> [batch_size, h, 1, d_v]
        # key.shape = [batch_size, U, d] -> [batch_size, U, h, d_v] -> [batch_size, h, U, d_v]
        # value.shape = [batch_size, U, d] -> [batch_size, U, h, d_v] -> [batch_size, h, U, d_v]

        query = self.W_q(comp_decoder_state).reshape(num_batches, -1, self.h, self.d_v).transpose(1, 2)  # sequence 길이 모르는걸 -1로 설정.
        key = self.W_k(comp_listener_feature).reshape(num_batches, -1, self.h, self.d_v).transpose(1, 2)
        value = self.W_v(comp_listener_feature).reshape(num_batches, -1, self.h, self.d_v).transpose(1, 2)

        x, p_attn = self.attention(query, key, value)
        x = x.transpose(1, 2).reshape(num_batches, -1, self.d)
        out = self.W_o(x) #[batch_size, 1, d]

        return out, p_attn #[batch_size, 1, d]


"""
decoder = torch.rand([32, 1, 128])
key = torch.rand([32, 300, 128])
mha = MultiHeadAttention(heads = 6, dimension = 6*64)
output, p_attn = mha(decoder, key)
print(output.shape, p_attn.shape) #[32, 1, 384] #[32, 6, 1, 300]
"""