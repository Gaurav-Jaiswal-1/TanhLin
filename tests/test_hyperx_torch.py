import torch
from src.hyperx_torch import HyperXActivation

def test_hyperx_pytorch_forward():
    activation = HyperXActivation(k=1.0)
    x = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float32)
    result = activation(x)
    assert result.shape == x.shape
    assert torch.all(result <= x)  # Check output shape and value constraints

def test_hyperx_pytorch_gradients():
    activation = HyperXActivation(k=1.0)
    x = torch.tensor([[1.0, -1.0], [0.5, -0.5]], requires_grad=True)
    result = activation(x)
    result.mean().backward()
    assert x.grad is not None  # Ensure gradients are computed
