import torch

class CustomLayer(torch.nn.Module):
    def __init__(self, k):
        super(CustomLayer, self).__init__()
        self.k = k

    def forward(self, inputs):
        inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        inputs = torch.where(torch.isinf(inputs), torch.sign(inputs) * 1e10, inputs)
        return inputs * torch.tanh(self.k * inputs)


class ClampedLayer(torch.nn.Module):
    def __init__(self, k):
        super(ClampedLayer, self).__init__()
        self.k = k

    def forward(self, inputs):
        result = inputs * torch.tanh(self.k * inputs)
        return torch.minimum(result, inputs)  # Clamp output
