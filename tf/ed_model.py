import math

import numpy as np
import tensorflow as tf
from tensorflow import keras


def sigmoid(x, u0=0.4):
    return 1 / (1 + tf.exp(-2 * x / u0))


def sigmoid_derivative(x, u0=0.4):
    return sigmoid(x) * (1 - sigmoid(x)) * 2 / u0


def ed_relu(x, move=0.1):
    return tf.cast(tf.where(x > move, x - move, 0), tf.float32)


def ed_relu_derivative(x, move=0.1):
    return tf.cast(tf.where(x > move, 1, 0), tf.float32)


class EDDense(keras.layers.Layer):
    def __init__(
        self,
        in_neuron_types: list[str],
        out_neuron_types: list[str],
        activate: str,
        out_size: int,
        bias: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bias = bias
        in_neuron_types = in_neuron_types[:] + ["+", "-"]

        self.in_features = len(in_neuron_types)
        self.out_features = len(out_neuron_types)
        self.out_size = out_size
        layer_shape = (self.out_size, self.in_features, self.out_features)

        if activate == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activate == "relu":
            self.activation = ed_relu
            self.activation_derivative = ed_relu_derivative
        else:
            self.activation = None
            self.activation_derivative = None

        # --- operator : pp+ pn- np- nn+
        self.forward_ope = np.zeros(layer_shape)
        self.update_p_ope = np.zeros(layer_shape)
        self.update_n_ope = np.zeros(layer_shape)
        for k in range(self.out_size):
            for i in range(self.in_features):
                for j in range(self.out_features):
                    if in_neuron_types[i] == "+":
                        if out_neuron_types[j] == "+":
                            self.forward_ope[k][i][j] = 1
                        elif out_neuron_types[j] == "-":
                            self.forward_ope[k][i][j] = -1
                        self.update_p_ope[k][i][j] = 1.0
                        self.update_n_ope[k][i][j] = 0.0
                    elif in_neuron_types[i] == "-":
                        if out_neuron_types[j] == "+":
                            self.forward_ope[k][i][j] = -1
                        elif out_neuron_types[j] == "-":
                            self.forward_ope[k][i][j] = 1
                        self.update_p_ope[k][i][j] = -0.0
                        self.update_n_ope[k][i][j] = -1.0
        self.forward_ope = tf.constant(self.forward_ope, self.dtype)
        self.update_p_ope = tf.constant(self.update_p_ope, self.dtype)
        self.update_n_ope = tf.constant(self.update_n_ope, self.dtype)

        # --- build
        self.weight = self.add_weight(
            name="weight",
            shape=layer_shape,
            initializer=keras.initializers.RandomUniform(0, 1 / math.sqrt(self.out_features)),
            trainable=True,
        )

    def call(self, x, training=False):
        # (batch, out_num, x) -> (batch, out_num, x + bias)
        bias = tf.fill((tf.shape(x)[0], self.out_size, 2), self.bias)
        x = tf.concat([x, bias], -1)
        w = self.weight * self.forward_ope

        # x: (batch, out_num,       1, in_size)
        # w: (    1, out_num, in_size, out_size)
        y = tf.matmul(tf.expand_dims(x, axis=2), tf.expand_dims(w, axis=0))

        # (batch, out_num, 1, out_size) -> (batch, out_num, out_size)
        y = tf.squeeze(y, axis=2)

        if training:
            self.recent_in = x
            self.recent_out = y

        if self.activation is not None:
            y = self.activation(y)
        return y

    def update_weights(self, loss, lr: float):
        # in_vals: (batch, out_num, unit) -> (batch, out_num, in_unit, out_size(tile))
        # grad   : (batch, out_num, unit) -> (batch, out_num, in_unit(tile), out_size)
        # loss   : (batch, out_num) -> (batch, out_num, 1, 1)
        in_vals = tf.tile(tf.expand_dims(self.recent_in, 3), (1, 1, 1, self.out_features))
        if self.activation_derivative is not None:
            grad = self.activation_derivative(self.recent_out)
            grad = tf.tile(tf.expand_dims(grad, 2), (1, 1, self.in_features, 1))
        else:
            grad = 1
        loss = tf.expand_dims(tf.expand_dims(loss, axis=-1), axis=-1)

        # 正負によって使うoperatorをチェンジ
        ope = tf.where(loss > 0, self.update_p_ope, self.update_n_ope)

        w = ope * loss * in_vals * grad

        # batch処理、差分を合計した値を更新幅とする
        w = tf.reduce_sum(w, axis=0)

        self.weight.assign_add(lr * w / self.out_features)


class EDModel(keras.Model):
    def __init__(
        self,
        input_num: int,
        output_num: int,
        layers: tuple | list = [
            (64, "sigmoid"),
            (1, "sigmoid"),
        ],
        out_type: str = "sigmoid",
        training_mode: str = "mse",
        lr: float = 0.1,
        bias: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.output_num = output_num
        self.training_mode = training_mode
        self.lr = lr
        self.bias = bias
        self.layer_num = len(layers)

        in_neurons = ["+"] * input_num + ["-"] * input_num
        self.h_layers = []
        for size, act_type in layers:
            out_neurons = [("+" if n % 2 == 0 else "-") for n in range(size)]
            self.h_layers.append(EDDense(in_neurons, out_neurons, act_type, output_num))
            in_neurons = out_neurons
        self.h_layers.append(EDDense(in_neurons, ["+"], out_type, output_num))

    def call(self, x, training=False):
        # (batch, x) -> (batch, out_num, x*2)
        tile_dims = (1, self.output_num, 2)
        x = tf.tile(tf.expand_dims(x, axis=1), tile_dims)
        for h in self.h_layers:
            x = h(x, training=training)
        return tf.squeeze(x, axis=-1)

    def train_step(self, data):
        x, y = data

        x = self(x, training=True)
        if self.training_mode == "mse":
            loss = y - x
        elif self.training_mode == "ce":
            x = tf.nn.softmax(x, -1)
            loss = y - x
        else:
            raise ValueError(self.training_mode)

        for h in self.h_layers:
            h.update_weights(loss, self.lr)

        return {"loss": tf.reduce_mean(loss)}
