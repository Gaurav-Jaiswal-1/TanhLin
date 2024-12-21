import tensorflow as tf
from src.hyperx_tf import HyperXActivation

def test_hyperx_tensorflow_forward():
    activation = HyperXActivation(k=1.0)
    x = tf.constant([[1.0, -1.0], [0.5, -0.5]], dtype=tf.float32)
    result = activation(x)
    assert result.shape == x.shape
    assert tf.reduce_all(result <= x)  # Check output shape and value constraints

def test_hyperx_tensorflow_gradients():
    with tf.GradientTape() as tape:
        x = tf.Variable([[1.0, -1.0], [0.5, -0.5]], dtype=tf.float32)
        tape.watch(x)
        activation = HyperXActivation(k=1.0)
        y = activation(x)
    grads = tape.gradient(y, x)
    assert grads is not None  # Ensure gradients exist
