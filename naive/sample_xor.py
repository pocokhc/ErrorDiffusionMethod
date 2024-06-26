import random

from ed_model_list import EDModel

random.seed(10)


def _simple_run(dataset, model: EDModel, N):
    for i in range(N):
        # x, target = dataset[i % len(dataset)]  # debug
        x, target = dataset[random.randint(0, len(dataset)) - 1]

        y1 = model.forward(x)
        loss = model.update_weights(x, target)
        y2 = model.forward(x)

        print(f"{i} {x} {y1:8.5f} -> {y2:8.5f}, target {target}, loss {loss}")

    print("--- weights ---")
    for neurons in model.hidden_neurons_list:
        for n in neurons:
            print(n)
    print(model.out_neuron)

    print("--- result ---")
    for x, target in dataset:
        y = model.forward(x)
        print(f"{x} -> {y:8.5f}, target {target}")


def main_not():
    model = EDModel(
        input_num=1,
        layers=[
            (2, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.1,
    )
    dataset = [
        [[0], 1],
        [[1], 0],
    ]
    _simple_run(dataset, model, 100)


def main_not_quantization():
    model = EDModel(
        input_num=1,
        layers=[
            (4, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.1,
        quantization=True,
    )
    dataset = [
        [[0], 1],
        [[1], 0],
    ]
    _simple_run(dataset, model, 100)


def main_xor():
    model = EDModel(
        input_num=2,
        layers=[
            (8, "sigmoid"),
            (8, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.4,
    )

    dataset = [
        [[0, 0], 0],
        [[1, 0], 1],
        [[0, 1], 1],
        [[1, 1], 0],
    ]
    _simple_run(dataset, model, 500)


def main_xor_quantization():
    model = EDModel(
        input_num=2,
        layers=[
            (16, "sigmoid"),
            (16, "sigmoid"),
        ],
        out_type="sigmoid",
        lr=0.1,
        quantization=True,
    )

    dataset = [
        [[0, 0], 0],
        [[1, 0], 1],
        [[0, 1], 1],
        [[1, 1], 0],
    ]
    _simple_run(dataset, model, 1000)


if __name__ == "__main__":
    # main_not()
    # main_not_quantization()
    main_xor()
    # main_xor_quantization()
