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

    def forward(self, src, state, src_mask):
        output, state = self.gru(src, state)
        output = self.bi2uni(output)
        output, _ = self.attn(output, output, src_mask)
        output += src # skip connection
        state = torch.sum(torch.stack(torch.split(state, 2, dim=0)), 1, keepdim=False) # bi-directional state -> uni-directional state
        return output, state

class AttentionDecoderLayer(nn.Module):
    def __init__(self, depth, hidden_dim):
        super(AttentionDecoderLayer, self).__init__()
        self.gru = nn.GRU(depth, hidden_dim, batch_first=True)
        self.t_attn = Attention(depth, hidden_dim)
        self.st_attn = Attention(depth, hidden_dim)

    def forward(self, src, tgt, state, tgt_mask, src_tgt_mask):
        tgt = self.gru(tgt, state)
        tgt = self.t_attn(tgt, tgt, tgt_mask)
        tgt = self.st_attn(src, tgt, src_tgt_mask)
        return tgt
        

el = AttentionEncoderLayer(10, 20)
dl = AttentionDecoderLayer(10, 20)

i = torch.randn((7, 5, 10)) # inputs
s = torch.zeros((2, 7, 20)) # states
ssm = torch.zeros((5, 5))
o, s = el(i, s, ssm)
print(o.size(), s.size())


t = torch.randn((7, 6, 10))
ttm = torch.zeros(6, 6)
stm = torch.zeros(5, 6)
o, s = dl (o, t, s, ttm, stm)
print(o.size(), s.size())
