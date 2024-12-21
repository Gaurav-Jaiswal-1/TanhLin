import numpy as np

def hyperx(x, k=1.0):
    """
    HyperX Activation Function: x * tanh(kx)
    
    Parameters:
        x (numpy.ndarray or float): Input value(s).
        k (float): Scaling factor for the input (default is 1.0).
    
    Returns:
        numpy.ndarray or float: Output of the activation function.
    """
    return x * np.tanh(k * x)

def hyperx_derivative(x, k=1.0):
    """
    Derivative of the HyperX Activation Function.
    
    Parameters:
        x (numpy.ndarray or float): Input value(s).
        k (float): Scaling factor for the input (default is 1.0).
    
    Returns:
        numpy.ndarray or float: Derivative of the activation function.
    """
    return np.tanh(k * x) + k * x * (1 - np.tanh(k * x) ** 2)
