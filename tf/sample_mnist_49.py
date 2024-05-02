import os
import random
import time

import numpy as np
import tensorflow as tf
from ed_model import EDModel
from ed_model_single import EDModelSingle
from matplotlib import pyplot as plt
from tensorflow import keras

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def _create_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    train_indices = np.where((y_train == 4) | (y_train == 9))[0]
    test_indices = np.where((y_test == 4) | (y_test == 9))[0]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    y_train = np.where((y_train == 9), 1, 0)[..., np.newaxis].astype(np.float32)
    y_test = np.where((y_test == 9), 1, 0)[..., np.newaxis].astype(np.float32)

    # debug
    # x_train = x_train[:500]
    # y_train = y_train[:500]
    return (x_train, y_train), (x_test, y_test)


def create_tf_model(layer_num, unit_num, lr, activation):
    layers = [keras.layers.Input(shape=(28 * 28,))]
    for _ in range(layer_num):
        layers.append(keras.layers.Dense(unit_num, activation=activation))
    layers.append(keras.layers.Dense(1, activation="sigmoid"))
    model = keras.models.Sequential(layers)
    model.compile(optimizer=keras.optimizers.RMSprop(lr), loss="mse", metrics=["accuracy"])
    return model


def create_single_ed(layer_num, unit_num, lr, activation):
    layers = [(unit_num, activation) for _ in range(layer_num)]
    model = EDModelSingle(
        input_num=28 * 28,
        layers=layers,
        out_type="sigmoid",
        lr=lr,
    )
    model.compile(metrics=["accuracy"])
    return model


def create_ed_model(layer_num, unit_num, lr, activation):
    layers = [(unit_num, activation) for _ in range(layer_num)]
    model = EDModel(
        input_num=28 * 28,
        output_num=1,
        layers=layers,
        out_type="sigmoid",
        training_mode="mse",
        lr=lr,
    )
    model.compile(metrics=["accuracy"])
    return model


def main(layer_num, unit_num, activation, lr):
    (x_train, y_train), (x_test, y_test) = _create_dataset()
    name_list = []
    time_list = []
    for name, model in [
        ["TF", create_tf_model(layer_num, unit_num, lr, activation)],
        ["EDMethod(Single)", create_single_ed(layer_num, unit_num, lr, activation)],
        ["EDMethod", create_ed_model(layer_num, unit_num, lr, activation)],
    ]:
        t0 = time.time()
        history = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_data=(x_test, y_test))
        time_list.append(time.time() - t0)
        name_list.append(name)

        model.evaluate(x_test, y_test)
        plt.plot(history.history["val_accuracy"], label=name)

    plt.ylim(0.49, 1)
    plt.grid()
    plt.legend()
    plt.show()

    plt.bar(name_list, time_list)
    plt.show()


if __name__ == "__main__":
    main(layer_num=3, unit_num=32, activation="sigmoid", lr=0.001)
    # main(layer_num=3, unit_num=32, activation="relu", lr=0.001)
    # main(layer_num=100, unit_num=16, activation="sigmoid", lr=0.001)
