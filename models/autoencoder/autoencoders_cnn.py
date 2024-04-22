import torch
from torch import nn 
import torch.nn.functional as F
from .cnn.encoder import LatentSpaceEncoder
from .cnn.decoder import LatentSpaceDecoder
from .interpolate import Interpolator
from typing import Tuple, List
from vector_quantize_pytorch import VectorQuantize
from abc import abstractmethod
from einops import rearrange
from torch.cuda.amp.autocast_mode import autocast

    
class Autoencoder(nn.Module):
    def __init__(self, n_channels_init:int, latent_space_channel_dim:int=32):
        super().__init__()
        self.encoder = LatentSpaceEncoder(n_channels_init, out_chan=latent_space_channel_dim, num_layers_main=4)
        self.decoder = LatentSpaceDecoder(n_channels_init, in_chan=latent_space_channel_dim, num_layers_main=4)
        self.device = 'cuda'
    def encode(self, x:torch.Tensor): return self.encoder(x)
    def decode(self, x:torch.Tensor): return self.decoder(x)
    def forward(self, x:torch.Tensor):
        return self.encoder(self.decoder(x))
    
    @abstractmethod
    def reg_loss(self, _input:torch.Tensor):
        raise NotImplementedError()

class AutoencoderKL(Autoencoder):
    def __init__(self,  n_channels_init:int, latent_space_channel_dim:int=32,
                 kl_penalty:float=1e-6):
        super().__init__(n_channels_init, latent_space_channel_dim)
        self.regularization_loss = nn.KLDivLoss(False)
        self.kl_penalty = kl_penalty
        self.device = 'cuda'
    
    @autocast(enabled=False)
    def reg_loss(self, _input:torch.Tensor):
        target = F.softmax(torch.randn_like(_input))
        return self.kl_penalty * self.regularization_loss.forward(F.log_softmax(_input), target)
    
class AutoencoderVQ(Autoencoder):
    def __init__(self, n_channels_init:int, latent_space_channel_dim:int=32,
                 codebook_size:int=512):
        super().__init__(n_channels_init, latent_space_channel_dim)
        quant_dim = latent_space_channel_dim
        self.codebook_size = codebook_size
        self.vquantizer = VectorQuantize(quant_dim, self.codebook_size, channel_last=False, ema_update=False, learnable_codebook=True)
        self.quant_conv = nn.Conv2d(latent_space_channel_dim, quant_dim, 1)
        self.post_quant_conv = nn.Conv2d(quant_dim, latent_space_channel_dim, 1)
        self.device = 'cuda'
        # print(self.vquantizer.codebook, self.vquantizer.codebook.shape)

    def encode(self, x:torch.Tensor):
        dtype = x.dtype
        x = self.encoder(x)
        h = x.shape[2]
        x = self.quant_conv(x)
        x = rearrange(x, "b c h w -> b c (h w)")
        x = x.float()
        x, idx, loss = self.vquantizer.forward(x)
        # self.vquantizer.get_codes_from_indices: the lookup function
        x = x.to(dtype)
        loss = loss.to(dtype)
        x = rearrange(x, "b c (h w) -> b c h w", h=h)
        return x, idx, loss
    
    def decode(self, x:torch.Tensor):
        x = self.post_quant_conv(x)
        x = self.decoder(x)
        return x
    
    def forward(self, x:torch.Tensor):
        x, idx, loss = self.encode(x)
        x = self.decode(x)
        return x, idx, loss
    
    def reg_loss(self, _input:torch.Tensor):
        _, _, loss = self.vquantizer(_input)
        return loss

# ae = AutoencoderVQ(4)
# imp = torch.randn((4, 3, 256, 256))
# x, idx, _ = ae.encode(imp)
# y = ae.vquantizer.get_output_from_indices(idx).reshape(x.shape)
# print(torch.isclose(x, y))


class MultiScaleAutoencoderVQ(Autoencoder):
    def __init__(self, n_channels_init:int, steps:Tuple[int]=(1,2,3,4,5,6,8,10,13,16), latent_space_channel_dim:int=32, codebook_size:int=4096):
        super().__init__(n_channels_init, latent_space_channel_dim)
        self.steps = steps
        self.codebook_size = codebook_size
        self.latent_space_channel_dim = latent_space_channel_dim

        quant_dim = latent_space_channel_dim

        self.vquantizer = VectorQuantize(quant_dim, self.codebook_size, channel_last=False, ema_update=False, learnable_codebook=True)
        self.quant_conv = nn.Conv2d(latent_space_channel_dim, quant_dim, 1)
        self.post_quant_conv = nn.Conv2d(quant_dim, latent_space_channel_dim, 1)

        self.resolution_conv = nn.ModuleList()                  # decoding the latents
        self.forward_interpolator = nn.ModuleList()             # feeding the latents

        for step in self.steps:
            self.resolution_conv.append(
                nn.Sequential(
                    Interpolator((steps[-1], steps[-1])),
                    nn.Conv2d(latent_space_channel_dim, latent_space_channel_dim, 3, padding=1)
                )                  # HxH => 16x16
            )
            self.forward_interpolator.append(
                Interpolator((step, step))
            )

    def encode(self, x:torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        dtype = x.dtype
        x = self.encoder(x)         #(B, 3, 256, 256) => (B, C, 16, 16)
        # h = x.shape[2]
        x = self.quant_conv(x)
        # x = rearrange(x, "b c h w -> b c (h w)")
        x = x.float()
        token_maps = [] # token maps
        vqlosses = 0

        for i, (interpolator, resolution_conv) in enumerate(zip(self.forward_interpolator, self.resolution_conv)):
            z, idx, loss = self.vquantizer.forward(rearrange(interpolator(x), "b c h w -> b c (h w)")) # z_k, r_k, vqloss
            token_maps.append(idx)
            z = rearrange(z, "b c (h w) -> b c h w", h=self.steps[i])
            z = resolution_conv(z)
            x = x - z
            vqlosses += loss
        
        return token_maps, vqlosses

    def decode(self, token_maps:List[torch.Tensor]) -> torch.Tensor:
        f = torch.zeros((token_maps[0].shape[0], self.latent_space_channel_dim, self.steps[-1], self.steps[-1]), device=token_maps[0].device)
        for k, r_k in enumerate(token_maps):
            z_k = self.vquantizer.get_output_from_indices(r_k)
            z_k = rearrange(z_k, "b c (h w) -> b c h w", h=self.steps[k])
            z_k = self.resolution_conv[k].forward(z_k)
            f += z_k        
        return self.decoder(f)

    def forward(self, x:torch.Tensor):
        R, vqloss = self.encode(x)
        x = self.decode(R)
        return x, R, vqloss
    
    def get_next_mapping(self, token_map:torch.Tensor|None, feature_map:torch.Tensor):
        """Get the next feature mapping from a token map
        @param token_map: a Tensor with integer dtype, containing the quantization token indices
        @param feature_map: a Tensor with float dtype, size equal to the latent dim, containing the feature map
        Returns:
        - the feature map with the quantized map added (shape retained)
        - the flattened quantized latent map interpolated to the next step size (shape (B, C_latent, H_step * W_step))
        ***Interpolated quantized map will be the Transformer input***
        ***Should only be used when training the whole model (with the Transformer)***
        """
        if token_map == None:
            return feature_map, feature_map
        z = self.vquantizer.get_output_from_indices(token_map)
        dim = int(round((token_map.shape[1]) ** .5))
        z = rearrange(z, "b c (h w) -> b c h w", h=dim)
        idx = self.steps.index(dim)
        z = self.resolution_conv(z)
        assert z.shape == feature_map.shape, "looks like the feature map shape does not match the latent dim. double check."
        feature_map += z
        if idx < self.steps.__len__() - 1:
            z = self.forward_interpolator[idx + 1](z)
        return feature_map, rearrange(z, "b c h w -> b c (h w)")
        