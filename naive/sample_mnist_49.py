import random

import numpy as np
import pandas as pd
import tensorflow as tf
from ed_model_list import EDModel
from matplotlib import pyplot as plt
from tqdm import tqdm

random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def _create_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    train_indices = np.where((y_train == 0) | (y_train == 1))[0]
    test_indices = np.where((y_test == 0) | (y_test == 1))[0]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]
    # debug
    x_train = x_train[:100]
    y_train = y_train[:100]
    return (x_train, y_train), (x_test, y_test)


def main(layer_num, unit_num, activation, lr, quantization):
    (x_train, y_train), (x_test, y_test) = _create_dataset()

    model = EDModel(
        input_num=28 * 28,
        layers=[(unit_num, activation) for _ in range(layer_num)],
        out_type="sigmoid",
        lr=lr,
        quantization=quantization,
    )

    loss_list = []
    epochs = 1
    for i in range(epochs):
        for j in tqdm(range(len(x_train))):
            x = x_train[j]
            target = y_train[j]
            loss = model.update_weights(x, target)
            loss_list.append(loss)

    correct = 0
    total = 0
    for i in tqdm(range(len(x_test))):
        y = model.forward(x_test[i])
        y = 1 if y > 0.5 else 0
        if y_test[i] == y:
            correct += 1
        total += 1
    print(f"{100 * correct / total:.2f}%")

    plt.plot(pd.DataFrame(loss_list).rolling(20).mean())
    plt.plot(loss_list, alpha=0.2)
    plt.grid()
    plt.xlabel("step")
    plt.ylabel("diff")
    plt.show()


if __name__ == "__main__":
    main(layer_num=5, unit_num=16, activation="sigmoid", lr=0.1, quantization=False)
    # main(layer_num=5, unit_num=16, activation="sigmoid", lr=0.1, quantization=True)
