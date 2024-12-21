import tensorflow as tf

class HyperXActivation(tf.keras.layers.Layer):
    def __init__(self, k=1.0, **kwargs):
        super(HyperXActivation, self).__init__(**kwargs)
        self.k = k

    def call(self, inputs):
        return inputs * tf.tanh(self.k * inputs)
