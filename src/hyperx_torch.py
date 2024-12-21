import torch
import torch.nn as nn

class HyperXActivation(nn.Module):
    def __init__(self, k=1.0):
        super(HyperXActivation, self).__init__()
        self.k = k

    def forward(self, x):
        return x * torch.tanh(self.k * x)
