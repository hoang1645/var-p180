import torch
from torch import nn 
import torch.nn.functional as F

class RootMeanSquaredNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.randn(dim,))
        self.dim = dim
    def forward(self, x:torch.Tensor):
        if x.dim() == 1:
            return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)
        return F.normalize(x, dim = 1) * self.g.reshape([1, self.dim] + [1] * (x.dim() - 2)) * (x.shape[1] ** 0.5)
    

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, eps, C:float, k=.1, dim:int=1):
        super().__init__()
        self.eps = eps
        self.C = C
        self.k = k
        self.dim = dim
        
    def __phi(self, y):
        return self.C * (1 - self.k * y)


    def forward(self, x:torch.Tensor):
        y = F.normalize(x, dim=self.dim, eps=self.eps)
        return self.__phi(y)
    
AdaLN = AdaptiveLayerNorm
