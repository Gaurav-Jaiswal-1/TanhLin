import pytest
import torch
import tensorflow as tf
from src.hyperx_torch import HyperXActivation as PyTorchHyperX
from src.hyperx_tf import HyperXActivation as TensorFlowHyperX

# PyTorch Tests
def test_pytorch_hyperx():
    activation = PyTorchHyperX(k=1.0)
    x = torch.tensor([[1.0, -1.0], [0.5, -0.5]])
    result = activation(x)
    assert result.shape == x.shape
    assert torch.all(result <= x)  # Ensure HyperX reduces extreme values

# TensorFlow Tests
def test_tensorflow_hyperx():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.constant([[1.0, -1.0], [0.5, -0.5]], dtype=tf.float32)
    result = activation(x)
    assert result.shape == x.shape
    assert tf.reduce_all(result <= x)  # Ensure HyperX reduces extreme values
