import time
import torch
from src.hyperx_tf import HyperXActivation

def benchmark_activation(activation, iterations=1000, size=(1000, 1000)):
    x = torch.randn(size)
    start = time.time()
    for _ in range(iterations):
        _ = activation(x)
    return time.time() - start

activation = HyperXActivation(k=1.0)
time_taken = benchmark_activation(activation)
print(f"HyperX Activation took {time_taken:.4f} seconds for 1000 iterations.")
