import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# source target attention
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.linear = nn.Linear(input_dim*2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, src, tgt):
        # src    : [batch, src_len, input_dim]
        # tgt    : [batch, tgt_len, input_dim]
        # output : [batch, tgt_len, output_dim]
        t_src = torch.transpose(src, 1, 2) # [batch, input_dim, src_len]
        weight = torch.bmm(tgt, t_src) # [batch, tgt_len, src_len]
        print(weight.size(), weight[0])
        weight = self.softmax(weight)
        print(weight.size(), weight[0])
        
        # [batch, tgt_len, input_dim]
        weight_sum = torch.bmm(weight, src) # [input_dim, tgt_len]
        output = torch.cat([weight_sum, tgt], dim=2)
        output = self.linear(output)
        return output, weight

