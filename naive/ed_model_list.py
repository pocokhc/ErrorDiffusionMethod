import math
import random


def sigmoid(x, u0=0.4):
    return 1 / (1 + math.exp(-2 * x / u0))


def sigmoid_derivative(x, u0=0.4):
    return sigmoid(x) * (1 - sigmoid(x)) * 2 / u0


def ed_relu(x, move=0.1):
    return x - move if x > move else 0


def ed_relu_derivative(x, move=0.1):
    return 1 if x > move else 0


class EDNeuron:
    def __init__(
        self,
        in_neuron_types: list[str],
        neuron_type: str,
        activate: str,
        bias: float = 0.8,
        lr: float = 0.8,
        quantization: bool = False,
        name: str = "",
    ) -> None:
        self.in_neuron_types = ["+", "-"] + in_neuron_types[:]
        self.neuron_type = neuron_type
        self.bias = bias
        self.lr = lr
        self.quantization = quantization
        self.name = name

        if activate == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activate == "relu":
            self.activation = ed_relu
            self.activation_derivative = ed_relu_derivative
        else:
            self.activation = None
            self.activation_derivative = None

        # --- init weights
        self.weights = []
        for i in range(len(self.in_neuron_types)):
            self.weights.append(random.random() / len(self.in_neuron_types))

        # --- operator: pp+ pn- np- nn+
        self.forward_ope_list = []
        for n in self.in_neuron_types:
            if neuron_type == "+":
                if n == "+":
                    self.forward_ope_list.append(1)
                else:
                    self.forward_ope_list.append(-1)
            else:
                if n == "+":
                    self.forward_ope_list.append(-1)
                else:
                    self.forward_ope_list.append(1)

        # --- update index
        # 入力元が+ならupper時に学習
        # 入力元が-ならlower時に学習
        self.upper_idx_list = []
        self.lower_idx_list = []
        for i, n in enumerate(self.in_neuron_types):
            if n == "+":
                self.upper_idx_list.append(i)
            else:
                self.lower_idx_list.append(i)

    def forward(self, x, training=False):
        x = [self.bias, self.bias] + x
        assert len(self.weights) == len(x)

        if self.quantization:
            a = sum(self.weights) / len(self.weights)
            if a < 0.0001:
                a = 0.0001

        y = 0
        for i in range(len(self.weights)):
            if self.quantization:
                w = 1 if self.weights[i] / a > 0.5 else 0
            else:
                w = self.weights[i]
            y += x[i] * w * self.forward_ope_list[i]

        if training:
            self.prev_in = x
            self.prev_out = y

        if self.activation is not None:
            y = self.activation(y)
        return y

    def update_weight(self, diff, upper_rate: float, lower_rate: float):
        if len(self.in_neuron_types) == 0:
            return

        if self.activation_derivative is not None:
            grad = self.activation_derivative(self.prev_out)
        else:
            grad = 1

        for idx in range(len(self.in_neuron_types)):
            if idx in self.upper_idx_list:
                _rate = upper_rate
                _diff = diff
            elif idx in self.lower_idx_list:
                _rate = lower_rate
                _diff = -diff

            _old_w = self.weights[idx]  # debug
            delta = _diff * grad * self.prev_in[idx]
            self.weights[idx] += self.lr * delta * _rate

            # --- debug
            s = f"{idx:2d}"
            s += f", {self.prev_in[idx]:5.2f}"
            s += f", f'({self.prev_out:5.2f})={grad:5.2f}"
            s += f", diff {diff:5.2f}"
            s += f", {delta:6.3f}"
            s += f", w {_old_w:5.2f} -> {self.weights[idx]:5.2f}"
            # print(s)

    def __str__(self):
        s = f"{self.name} {self.neuron_type}"
        arr = []
        for i in range(len(self.weights)):
            o = "+" if i in self.upper_idx_list else "-"
            arr.append(f"{self.weights[i]:6.3f}({o})")
        s += " [" + ", ".join(arr) + "]"
        return s


class EDModel:
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
        quantization: bool = False,
    ) -> None:
        self.lr = lr

        # --- hidden
        self.hidden_neurons_list: list[list[EDNeuron]] = []
        in_neurons = ["+", "-"] * input_num
        for j, (size, activate) in enumerate(layers):
            out_neurons = [("+" if n % 2 == 0 else "-") for n in range(size)]
            hidden_neurons = []
            for i in range(size):
                hidden_neurons.append(
                    EDNeuron(
                        in_neurons,
                        neuron_type=("+" if i % 2 == 0 else "-"),
                        activate=activate,
                        bias=bias,
                        lr=lr,
                        quantization=quantization,
                        name=f"h{j}{i}",
                    )
                )
            self.hidden_neurons_list.append(hidden_neurons)
            in_neurons = out_neurons

        # --- output
        self.out_neuron = EDNeuron(
            in_neurons,
            "+",
            activate=out_type,
            bias=bias,
            lr=lr,
            quantization=quantization,
            name="out",
        )

    def forward(self, inputs, training=False):
        x = []
        for n in inputs:
            x.append(n)  # +
            x.append(n)  # -

        for neurons in self.hidden_neurons_list:
            x = [h.forward(x, training=training) for h in neurons]

        x = self.out_neuron.forward(x, training=training)
        return x

    def update_weights(self, inputs, target):
        x = self.forward(inputs, training=True)

        # --- update(ED)
        diff = target - x
        if diff > 0:
            upper_rate = 1.0
            lower_rate = -0.0
        else:
            upper_rate = -0.0
            lower_rate = 1.0

        # update
        for neurons in self.hidden_neurons_list:
            for n in neurons:
                n.update_weight(diff, upper_rate, lower_rate)
        self.out_neuron.update_weight(diff, upper_rate, lower_rate)

        return diff
