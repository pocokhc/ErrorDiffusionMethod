import os
import random

import numpy as np
from ed_model import EDModel
from ed_model_single import EDModelSingle

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def _simple_run(dataset, model, N):
    for i in range(N):
        # x, target = dataset[i % len(dataset)]  # debug
        x, target = dataset[random.randint(0, len(dataset)) - 1]

        np_x = np.array([x], np.float32)
        np_target = np.array([target], np.float32)

        y1 = model(np_x).numpy()[0][0]
        loss = model.train_step((np_x, np_target))
        loss = loss["loss"].numpy()
        y2 = model(np_x).numpy()[0][0]

        print(f"{i} {x} {y1:8.5f} -> {y2:8.5f}, target {target}, loss {loss}")

    print("--- weights ---")
    for p in model.weights:
        print(p)

    print("--- result ---")
    for x, target in dataset:
        y = model(np.array([x], np.float32)).numpy()[0][0]
        print(f"{x} -> {y:8.5f}, target {target}")


def main_not_single_model():
    model = EDModelSingle(
        input_num=1,
        layers=[
            (2, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.1,
    )
    dataset = [
        [[0], [1]],
        [[1], [0]],
    ]
    _simple_run(dataset, model, 100)


def main_not():
    model = EDModel(
        input_num=1,
        output_num=1,
        layers=[
            (2, "sigmoid"),
        ],
        out_type="sigmoid",
        training_mode="mse",
        lr=0.1,
        quantization=False,
    )
    dataset = [
        [[0], [1]],
        [[1], [0]],
    ]
    _simple_run(dataset, model, 100)


def main_not_quantization():
    model = EDModel(
        input_num=1,
        output_num=1,
        layers=[
            (2, "sigmoid"),
        ],
        out_type="sigmoid",
        training_mode="mse",
        lr=0.1,
        quantization=True,
    )
    dataset = [
        [[0], [1]],
        [[1], [0]],
    ]
    _simple_run(dataset, model, 100)


def main_xor_single_model():
    model = EDModelSingle(
        input_num=2,
        layers=[
            (8, "sigmoid"),
            (8, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.8,
    )
    dataset = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    _simple_run(dataset, model, 400)


def main_xor():
    model = EDModel(
        input_num=2,
        output_num=1,
        layers=[
            (8, "sigmoid"),
            (8, "sigmoid"),
        ],
        out_type="sigmoid",
        training_mode="mse",
        lr=0.1,
        quantization=False,
    )
    dataset = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    _simple_run(dataset, model, 400)


def main_xor_quantization():
    model = EDModel(
        input_num=2,
        output_num=1,
        layers=[
            (16, "sigmoid"),
            (16, "sigmoid"),
        ],
        out_type="sigmoid",
        training_mode="mse",
        lr=0.1,
        quantization=True,
    )
    dataset = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    _simple_run(dataset, model, 1000)


if __name__ == "__main__":
    # main_not_single_model()
    # main_not()
    # main_not_quantization()
    # main_xor_single_model()
    main_xor()
    # main_xor_quantization()
