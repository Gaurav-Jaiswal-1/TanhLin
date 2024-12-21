import time
import torch
import tensorflow as tf
from src.hyperx_torch import HyperXActivation as PyTorchHyperX
from src.hyperx_tf import HyperXActivation as TensorFlowHyperX

def test_pytorch_performance():
    activation = PyTorchHyperX(k=1.0)
    x = torch.randn(1000, 1000)
    start = time.time()
    for _ in range(100):
        activation(x)
    end = time.time()
    print(f"PyTorch HyperX: {end - start:.4f} seconds for 100 iterations.")

def test_tensorflow_performance():
    activation = TensorFlowHyperX(k=1.0)
    x = tf.random.normal((1000, 1000))
    start = time.time()
    for _ in range(100):
        activation(x)
    end = time.time()
    print(f"TensorFlow HyperX: {end - start:.4f} seconds for 100 iterations.")
