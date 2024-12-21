import torch
import torch.nn as nn
from src.hyperx_torch import HyperXActivation

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 5)
        self.activation = HyperXActivation(k=1.5)

    def forward(self, x):
        x = self.fc(x)
        return self.activation(x)

# Example usage
model = SimpleModel()
x = torch.randn(2, 10)  # Random input tensor
output = model(x)
print("Output:", output)
