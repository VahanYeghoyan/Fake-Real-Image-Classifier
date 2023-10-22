import tensorflow as tf
import numpy as np


class Dense_layer(tf.Module):
    def __init__(self, input_size, output_size, activation=tf.identity, bias=True):
        super(Dense_layer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.enable_bias = bias

        if self.enable_bias:
            self.b = tf.Variable(initial_value=np.zeros((1, self.output_size), dtype=np.float64), dtype=tf.float32)

        xavier_initialization_var = 2.0 / (self.input_size + self.output_size)
        xavier_initialization_sigma = np.sqrt(xavier_initialization_var)
        z = np.random.randn(self.input_size * self.output_size).astype(np.float32).reshape(
            self.input_size, self.output_size)
        weight_initial_value = z * xavier_initialization_sigma

        self.w = tf.Variable(initial_value=weight_initial_value, dtype=np.float32)
        self.activation = activation

    def __call__(self, x):
        if self.enable_bias:
            y = tf.matmul(x, self.w) + self.b
        else:
            y = tf.matmul(x, self.w)

        return self.activation(y)

    def get_dict(self):
        if self.enable_bias:
            return {self.b.name: self.b,
                    self.w.name: self.w, }
        else:
            return {self.w.name: self.w}
