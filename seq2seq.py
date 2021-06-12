import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from attention import Attention

class AttentionEncoderLayer(nn.Module):
    def __init__(self, depth, hidden_dim):
        super(AttentionEncoderLayer, self).__init__()
        self.gru = nn.GRU(depth, hidden_dim, bidirectional=True, batch_first=True)
        self.bi2uni = nn.Linear(hidden_dim*2, hidden_dim)
        self.attn = Attention(hidden_dim, depth)

    def forward(self, src, state):
        output, state = self.gru(src, state)
        output = self.bi2uni(output)
        output, _ = self.attn(output, output, torch.zeros((src.size(1), src.size(1))))
        output += src # skip connection
        return output, state

class AttentionDecoderLayer(nn.Module):
    def __init__(self, depth, hidden_dim):
        super(AttentionDecoderLayer, self).__init__()
        self.gru
        self.attn = Attention(depth, hidden_dim)
        self.self_attn_gru = AttentionEncoderLayer(depth, hidden_dim)


    def forward(self, x):

        return x


el = AttentionEncoderLayer(10, 20)
i = torch.randn((7, 5, 10)) # inputs
s = torch.zeros((2, 7, 20)) # states
o, s = el(i, s)
print(o.size(), s.size())