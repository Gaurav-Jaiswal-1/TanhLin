import tensorflow as tf
class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, k):
        super(CustomLayer, self).__init__()
        self.k = k

    def call(self, inputs):
        inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs), inputs)
        inputs = tf.where(tf.math.is_inf(inputs), tf.sign(inputs) * 1e10, inputs)
        return inputs * tf.tanh(self.k * inputs)


class ClampedLayer(tf.keras.layers.Layer):
    def __init__(self, k):
        super(ClampedLayer, self).__init__()
        self.k = k

    def call(self, inputs):
        result = inputs * tf.tanh(self.k * inputs)
        return tf.minimum(result, inputs)  # Clamp output
