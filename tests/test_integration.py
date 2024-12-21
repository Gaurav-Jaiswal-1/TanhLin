import torch
from src.hyperx_torch import HyperXActivation




def test_pytorch_integration():
    # Define a simple PyTorch model using HyperXActivation
    class PyTorchModel(torch.nn.Module):
        def __init__(self):
            super(PyTorchModel, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
            self.hyperx = HyperXActivation(k=1.0)
            self.fc = torch.nn.Linear(32 * 26 * 26, 10)  # Flattened size after Conv2d

        def forward(self, x):
            x = self.conv1(x)
            x = self.hyperx(x)
            x = x.view(x.size(0), -1)  # Flatten for the fully connected layer
            x = self.fc(x)
            return x

    # Initialize the model
    model = PyTorchModel()
    x = torch.randn(1, 1, 28, 28)  # Dummy image
    output = model(x)
    assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"



import tensorflow as tf
from src.hyperx_tf import HyperXActivation

def test_tensorflow_integration():
    # Define a simple TensorFlow model using HyperXActivation
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), padding="valid"),
        HyperXActivation(k=1.0),  # Custom activation
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')  # 10-class output
    ])

    # Test the model with dummy input
    x = tf.random.normal((1, 28, 28, 1))  # Dummy image
    output = model(x)
    assert output.shape == (1, 10), f"Unexpected output shape: {output.shape}"
