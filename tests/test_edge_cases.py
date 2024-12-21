import torch
import tensorflow as tf
from src.hyperx_torch import HyperXActivation as PyTorchHyperX
from src.hyperx_tf import HyperXActivation as TensorFlowHyperX

def test_edge_cases_pytorch():
    activation = PyTorchHyperX(k=1.0)
    x = torch.tensor([[float("inf"), -float("inf")], [0.0, float("nan")]])
    result = activation(x)
    assert not torch.any(torch.isnan(result)), "NaN values in PyTorch output!"

def test_edge_cases_tensorflow():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.constant([[float("inf"), -float("inf")], [0.0, float("nan")]])
    result = activation(x)
    assert not tf.reduce_any(tf.math.is_nan(result)), "NaN values in TensorFlow output!"



