import math
from typing import Any

import torch
import torch.nn as nn


def sigmoid(x, u0=0.4):
    return 1 / (1 + torch.exp(-2 * x / u0))


def sigmoid_derivative(x, u0=0.4):
    return sigmoid(x) * (1 - sigmoid(x)) * 2 / u0


def ed_relu(x):
    return torch.where(x > 0.1, x - 0.1, 0)


def ed_relu_derivative(x):
    return torch.where(x > 0.1, 1, 0)


class EDDense(nn.Module):
    def __init__(
        self,
        in_neuron_types: list[str],
        out_neuron_types: list[str],
        activate: str,
        bias: float = 0.8,
        device: Any = "cpu",
    ):
        super().__init__()
        self.bias = bias
        self.device = device
        in_neuron_types = in_neuron_types[:] + ["+", "-"]

        self.in_features = len(in_neuron_types)
        self.out_features = len(out_neuron_types)
        layer_shape = (self.in_features, self.out_features)
        self.weight = nn.Parameter(torch.empty(layer_shape)).to(device)

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
        self.forward_ope = torch.zeros(layer_shape).to(device)
        self.update_p_ope = torch.zeros(layer_shape).to(device)
        self.update_n_ope = torch.zeros(layer_shape).to(device)
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

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.uniform_(0, 1 / math.sqrt(self.weight.size(1)))

    def forward(self, x):
        # (batch, x) -> (batch, x + bias)
        bias = torch.full((x.size(0), 2), self.bias).to(self.device)
        x = torch.cat([x, bias], dim=-1)
        w = self.weight * self.forward_ope

        # x: (batch  ,  in_size)
        # w: (in_size, out_size)
        # y: (batch  , out_size)
        y = torch.matmul(x, w)

        # 学習時のみ必要、forwardだけの場合不要
        self.recent_in = x
        self.recent_out = y

        if self.activation is not None:
            y = self.activation(y)
        return y

    def update_weights(self, loss, lr: float):
        # in_vals: (batch, unit) -> (batch, in_unit, out_size(tile))
        # grad   : (batch, unit) -> (batch, in_unit(tile), out_size)
        # loss   : (batch) -> (batch, 1, 1)
        in_vals = torch.tile(torch.unsqueeze(self.recent_in, 2), (1, 1, self.out_features))
        if self.activation_derivative is not None:
            grad = self.activation_derivative(self.recent_out)
            grad = torch.tile(torch.unsqueeze(grad, 1), (1, self.in_features, 1))
        else:
            grad = 1
        loss = torch.unsqueeze(loss, -1)

        # 正負によって使うoperatorをチェンジ
        ope = torch.where(loss > 0, self.update_p_ope, self.update_n_ope)

        w = ope * loss * in_vals * grad

        # batch処理、差分を合計した値を更新幅とする
        w = w.sum(dim=0)

        self.weight += lr * w / self.out_features


class EDModelSingle(nn.Module):
    def __init__(
        self,
        input_num: int,
        layers: tuple | list = [
            (64, "sigmoid"),
            (1, "sigmoid"),
        ],
        out_type: str = "sigmoid",
        bias: float = 0.8,
        lr: float = 0.1,
        device: Any = "cpu",
    ):
        super().__init__()
        self.bias = bias
        self.lr = lr
        self.device = device
        self.layer_num = len(layers)

        in_neurons = ["+"] * input_num + ["-"] * input_num
        self.layers = nn.ModuleList()
        for size, act_type in layers:
            out_neurons = [("+" if n % 2 == 0 else "-") for n in range(size)]
            self.layers.append(EDDense(in_neurons, out_neurons, act_type, device=device))
            in_neurons = out_neurons
        self.layers.append(EDDense(in_neurons, ["+"], out_type, device=device))

    def forward(self, x: torch.Tensor):
        # (batch, x) -> (batch, x*2)
        tile_dims = (1, 2)
        x = torch.tile(x, dims=tile_dims).to(self.device)
        for h in self.layers:
            x = h(x)
        return x

    def train(self, inputs, target):
        with torch.no_grad():
            x = self(inputs)
            loss = target - x  # mse
            for h in self.layers:
                h.update_weights(loss, self.lr)
            return loss


class EDModel(nn.Module):
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
        device: Any = "cpu",
    ):
        super().__init__()
        self.output_num = output_num
        self.training_mode = training_mode
        self.lr = lr

        self.models = [EDModelSingle(input_num, layers, out_type, bias, lr, device) for _ in range(output_num)]

    def forward(self, x: torch.Tensor):
        outputs = []
        for n in range(self.output_num):
            outputs.append(self.models[n].forward(x))
        return torch.cat(outputs, dim=-1)

    def train(self, inputs, target):
        with torch.no_grad():
            x = self(inputs)

            if self.training_mode == "mse":
                loss = target - x
            elif self.training_mode == "ce":
                x = torch.softmax(x, -1)
                loss = target - x
            else:
                raise ValueError(self.training_mode)

            for n in range(self.output_num):
                for h in self.models[n].layers:
                    h.update_weights(torch.unsqueeze(loss[:, n], -1), self.lr)

            return loss
