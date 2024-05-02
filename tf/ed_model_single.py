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
        bias: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.bias = bias
        in_neuron_types = in_neuron_types[:] + ["+", "-"]

        self.in_features = len(in_neuron_types)
        self.out_features = len(out_neuron_types)
        layer_shape = (self.in_features, self.out_features)

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
        for i in range(self.in_features):
            for j in range(self.out_features):
                if in_neuron_types[i] == "+":
                    if out_neuron_types[j] == "+":
                        self.forward_ope[i][j] = 1
                    elif out_neuron_types[j] == "-":
                        self.forward_ope[i][j] = -1
                    self.update_p_ope[i][j] = 1.0
                    self.update_n_ope[i][j] = 0.0
                elif in_neuron_types[i] == "-":
                    if out_neuron_types[j] == "+":
                        self.forward_ope[i][j] = -1
                    elif out_neuron_types[j] == "-":
                        self.forward_ope[i][j] = 1
                    self.update_p_ope[i][j] = -0.0
                    self.update_n_ope[i][j] = -1.0
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
        # (batch, x) -> (batch, x + bias)
        bias = tf.fill((tf.shape(x)[0], 2), self.bias)
        x = tf.concat([x, bias], -1)
        w = self.weight * self.forward_ope

        # x: (batch  , in_size)
        # w: (in_size, out_size)
        # y: (batch  , out_size)
        y = tf.matmul(x, w)

        if training:
            self.recent_in = x
            self.recent_out = y

        if self.activation is not None:
            y = self.activation(y)
        return y

    def update_weights(self, loss, lr: float):
        # in_vals: (batch, unit) -> (batch, in_unit, out_size(tile))
        # grad   : (batch, unit) -> (batch, in_unit(tile), out_size)
        # loss   : (batch, 1) -> (batch, 1, 1)
        in_vals = tf.tile(tf.expand_dims(self.recent_in, 2), (1, 1, self.out_features))
        if self.activation_derivative is not None:
            grad = self.activation_derivative(self.recent_out)
            grad = tf.tile(tf.expand_dims(grad, 1), (1, self.in_features, 1))
        else:
            grad = 1
        loss = tf.expand_dims(loss, axis=-1)

        # 正負によって使うoperatorをチェンジ
        ope = tf.where(loss > 0, self.update_p_ope, self.update_n_ope)

        w = ope * loss * in_vals * grad

        # batch処理、差分を合計した値を更新幅とする
        w = tf.reduce_sum(w, axis=0)

        self.weight.assign_add(lr * w / self.out_features)


class EDModelSingle(keras.Model):
    def __init__(
        self,
        input_num: int,
        layers: tuple | list = [
            (64, "sigmoid"),
            (1, "sigmoid"),
        ],
        out_type: str = "sigmoid",
        lr: float = 0.1,
        bias: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = lr
        self.bias = bias
        self.layer_num = len(layers)

        in_neurons = ["+"] * input_num + ["-"] * input_num
        self.h_layers = []
        for size, act_type in layers:
            out_neurons = [("+" if n % 2 == 0 else "-") for n in range(size)]
            self.h_layers.append(EDDense(in_neurons, out_neurons, act_type))
            in_neurons = out_neurons
        self.h_layers.append(EDDense(in_neurons, ["+"], out_type))

        self.build((None, input_num))

    def call(self, x, training=False):
        # (batch, x) -> (batch, x*2)
        tile_dims = (1, 2)
        x = tf.tile(x, tile_dims)
        for h in self.h_layers:
            x = h(x, training=training)
        return x

    def train_step(self, data):
        x, y = data

        x = self(x, training=True)
        loss = y - x  # mse

        for h in self.h_layers:
            h.update_weights(loss, self.lr)

        return {"loss": tf.reduce_mean(loss)}


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

        self.models = [EDModelSingle(input_num, layers, out_type, lr, bias) for _ in range(output_num)]

        self.build((None, input_num))

    def call(self, x, training=False):
        outputs = []
        for n in range(self.output_num):
            outputs.append(self.models[n](x, training=training))
        return tf.concat(outputs, axis=-1)

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

        for n in range(self.output_num):
            for h in self.models[n].h_layers:
                h.update_weights(tf.expand_dims(loss[:, n], -1), self.lr)

        return {"loss": tf.reduce_mean(loss)}
