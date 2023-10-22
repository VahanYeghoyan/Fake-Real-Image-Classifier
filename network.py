import numpy as np
import tensorflow as tf
from utlis import Dense_layer
from config import ModelConfig


class Network:
    def __init__(self, config: ModelConfig):
        self.name = config.name
        self.image_height = config.image_height
        self.image_width = config.image_width
        self.image_channels = config.image_channels
        self.create_net_variables()


    def get_opt_var_list(self):
        return list(self.opt_var_dict.values())

    def get_opt_var_dict(self):
        return self.opt_var_dict

    def create_net_variables(self):

        self.hidden_layer_1 = Dense_layer(self.image_width * self.image_height * self.image_channels,1000, tf.nn.relu)
        self.hidden_layer_2 = Dense_layer(1000, 100, tf.nn.relu)
        self.output_layer = Dense_layer(100, 2, tf.nn.softmax)
        self.opt_var_dict = {**self.hidden_layer_1.get_dict(), **self.hidden_layer_2.get_dict(), **self.output_layer.get_dict()}

    @tf.function
    def network(self, input):
        flat_input = tf.reshape(input,[tf.shape(input)[0], -1])
        h1 = self.hidden_layer_1(flat_input)
        h2 = self.hidden_layer_2(h1)
        o = self.output_layer(h2)
        return o

    @tf.function
    def get_loss(self, label, pred):
        return tf.reduce_mean(tf.reduce_sum(-label*tf.math.log(pred+1e-8), axis=-1))

    @tf.function
    def network_and_loss_call(self,input, label):
        pred = self.network(input)
        loss = self.get_loss(label, pred)
        return loss