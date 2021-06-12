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

    def forward(self, src, tgt, mask): #TODO: write mask
        # src    : [batch, src_len, input_dim]
        # tgt    : [batch, tgt_len, input_dim]
        # mask   : [batch, src_len, tgt_len] 
        # output : [batch, tgt_len, output_dim], [batch, tgt_len, src_len]

        # transpose
        t_src = torch.transpose(src, 1, 2) # [batch, input_dim, src_len]
        weight = torch.bmm(tgt, t_src) # [batch, tgt_len, src_len]
        
        # mask
        mask = torch.transpose(mask, 0, 1) # [batch, tgt_len, src_len]
        mask = mask.to(torch.float64) # convert bool -> float
        mask *= -1e16
        weight += mask

        # softmax
        weight = self.softmax(weight)
        
        weight_sum = torch.bmm(weight, src) # [batch, input_dim, tgt_len]
        output = torch.cat([weight_sum, tgt], dim=2)
        output = self.linear(output)
        return output, weight

## DEBUG CODES
#attn = Attention(20, 20)
#src, tgt = torch.randn((10, 5, 20)), torch.randn((10, 4, 20))
#mask = torch.ones((5, 4))
#result = attn(src, tgt, mask)
