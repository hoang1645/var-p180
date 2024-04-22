import torch
from torch import nn
from torch.nn.functional import interpolate


class Interpolator(nn.Module):
    def __init__(self, size:torch.Size|tuple=(256, 256)):
        super().__init__()
        self.size = size

    def forward(self, x:torch.Tensor):
        return interpolate(x, self.size, mode="area")