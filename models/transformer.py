import torch
from normalization import AdaLN
from torch import nn
from xformers.components.multi_head_dispatch import MultiHeadDispatch
from xformers.components.attention.scaled_dot_product import ScaledDotProduct
from typing import Callable


class Block(nn.Module):
    """a transformer decoder block w/o cross attn"""
    def __init__(self, dim:int, nheads:int, dropout:float, norm_first:bool=True, 
                 adaLN_C:float=0.5, adaLN_eps:float=1e-6, activation:Callable[..., nn.Module]=nn.SiLU):
        super().__init__()
        self.dim = dim
        self.dropout = dropout
        self.nheads = nheads
        self.norm_first = norm_first


        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            activation(),
            nn.Dropout(self.dropout),
            nn.Linear(dim * 4, dim),
            activation(),
            nn.Dropout(self.dropout)
        )
        self.attn = MultiHeadDispatch(self.dim, self.nheads, ScaledDotProduct(dropout=self.dropout, causal=True))
        self.mlp_norm = AdaLN(adaLN_C, adaLN_eps, dim=-1)
        self.attn_norm = AdaLN(adaLN_C, adaLN_eps, dim=-1)
                
    def forward(self, x:torch.Tensor):
        if self.norm_first:
            x += self.attn_norm(x)
            x = self.attn(x, x, x)
        else:
            x = x + self.attn_norm(self.attn.forward(x, x, x))
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(self, n_blocks:int=6, n_heads:int=8, dim:int=512, dropout:float=.1, 
                 norm_first:bool=True, adaLN_C:float=.5, adaLN_eps:float=1e-6, activation:Callable[..., nn.Module]=nn.SiLU):
        super().__init__()
        self.decoder = nn.Sequential(*[Block(dim, n_heads, dropout, norm_first, adaLN_C, adaLN_eps, activation) for _ in range(n_blocks)])
    
    def forward(self, x:torch.Tensor):
        return self.decoder(x)
