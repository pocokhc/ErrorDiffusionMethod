import random

import numpy as np
import torch
from ed_model import EDModel
from ed_model_single import EDModelSingle

random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=2, suppress=True, linewidth=200)


def _simple_run(dataset, model, N):
    for i in range(N):
        # x, target = dataset[i % len(dataset)]  # debug
        x, target = dataset[random.randint(0, len(dataset)) - 1]

        torch_x = torch.FloatTensor([x])
        torch_target = torch.FloatTensor([target])

        y1 = model(torch_x).cpu().detach().numpy()[0][0]
        loss = model.train(torch_x, torch_target)
        y2 = model(torch_x).cpu().detach().numpy()[0][0]

        print(f"{i} {x} {y1:8.5f} -> {y2:8.5f}, target {target}, loss {loss}")

    print("--- weights ---")
    for p in model.parameters():
        print(p)

    print("--- result ---")
    for x, target in dataset:
        y = model(torch.FloatTensor([x])).cpu().detach().numpy()[0][0]
        print(f"{x} -> {y:8.5f}, target {target}")


def main_not(is_single_model: bool):
    if is_single_model:
        model = EDModelSingle(
            input_num=1,
            layers=[(2, "sigmoid")],
            out_type="sigmoid",
            lr=0.1,
        )
    else:
        model = EDModel(
            input_num=1,
            output_num=1,
            layers=[(2, "sigmoid")],
            out_type="sigmoid",
            training_mode="mse",
            lr=0.1,
        )
    dataset = [
        [[0], [1]],
        [[1], [0]],
    ]
    _simple_run(dataset, model, 100)


def main_xor(is_single_model: bool):
    if is_single_model:
        model = EDModelSingle(
            input_num=2,
            layers=[(8, "sigmoid"), (8, "sigmoid")],
            out_type="sigmoid",
            lr=0.8,
        )
    else:
        model = EDModel(
            input_num=2,
            output_num=1,
            layers=[(8, "sigmoid"), (8, "sigmoid")],
            out_type="sigmoid",
            training_mode="mse",
            lr=0.8,
        )
    dataset = [
        [[0, 0], [0]],
        [[1, 0], [1]],
        [[0, 1], [1]],
        [[1, 1], [0]],
    ]
    _simple_run(dataset, model, 100)


if __name__ == "__main__":
    # main_not(is_single_model=True)
    # main_not(is_single_model=False)
    # main_xor(is_single_model=True)
    main_xor(is_single_model=False)
