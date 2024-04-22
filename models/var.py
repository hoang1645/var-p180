import torch
from torch import nn
from .autoencoder.autoencoders_cnn import MultiScaleAutoencoderVQ
from .transformer import TransformerDecoder
from typing import Tuple, Callable


from labml_nn.sampling.nucleus import NucleusSampler
from labml_nn.sampling.top_k import TopKSampler
from labml_nn.sampling.greedy import GreedySampler
from labml_nn.sampling.temperature import TemperatureSampler


class VAR(nn.Module):
    def __init__(self, vae_n_init_channels:int=32, interpolate_steps:Tuple[int]=(1,2,3,4,5,6,8,10,13,16),
                 latent_space_channels:int=32, vae_codebook_size:int=4096, transformer_n_blocks:int=4,
                 transformer_dim:int=384, transformer_n_heads:int=8, transformer_dropout:float=.1,
                 transformer_activation:Callable[..., nn.Module]=nn.SiLU, adaLN_C:float=.5, adaLN_eps:float=1e-6):
        self.vae = MultiScaleAutoencoderVQ(vae_n_init_channels, interpolate_steps,
                                           latent_space_channels, vae_codebook_size)
        self.transformer = TransformerDecoder(transformer_n_blocks, transformer_n_heads, transformer_dim,
                                              transformer_dropout, adaLN_C=adaLN_C, adaLN_eps=adaLN_eps, activation=transformer_activation)
        self.transformer_head = nn.Sequential(nn.Linear(transformer_dim, vae_codebook_size), nn.Softmax(dim=-1))

    def forward(self, x:torch.Tensor): # forwards in only the transformer part. the whole thing: see below
        """x: feature tensor (batch_size, seq_length, model_dim)"""
        return self.transformer_head(self.transformer(x))
    
    def decode(self, x:torch.Tensor):
        """x: reconstructed image feature map (batch_size, latent_dim ** 2, model_dim)"""
        return self.vae.decode()
    