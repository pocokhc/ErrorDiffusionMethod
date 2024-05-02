import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ed_model import EDModel
from ed_model_single import EDModelSingle
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
random.seed(10)
np.random.seed(10)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _create_dataset(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_indices = torch.where((train_dataset.targets == 4) | (train_dataset.targets == 9))[0]
    test_indices = torch.where((test_dataset.targets == 4) | (test_dataset.targets == 9))[0]
    train_image = train_dataset.data[train_indices]
    train_label = train_dataset.targets[train_indices]
    test_image = test_dataset.data[test_indices]
    test_label = test_dataset.targets[test_indices]

    train_image = train_image.float() / 255.0
    test_image = test_image.float() / 255.0
    train_image = train_image.view(train_image.size(0), -1)
    test_image = test_image.view(test_image.size(0), -1)

    train_label = torch.where((train_label == 9), 1, 0).unsqueeze(1).float()
    test_label = torch.where((test_label == 9), 1, 0).unsqueeze(1).float()

    # debug
    # train_image = train_image[:1000]
    # train_label = train_label[:1000]

    train_dataset = torch.utils.data.TensorDataset(train_image, train_label)
    test_dataset = torch.utils.data.TensorDataset(test_image, test_label)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_torch_model(layer_num, unit_num, activation, lr, batch_size, epochs):

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.h_layers = nn.ModuleList([nn.Linear(28 * 28, unit_num), nn.Sigmoid()])
            for _ in range(layer_num):
                self.h_layers.append(nn.Linear(unit_num, unit_num))
                if activation == "sigmoid":
                    self.h_layers.append(nn.Sigmoid())
                elif activation == "relu":
                    self.h_layers.append(nn.ReLU())
            self.h_layers.append(nn.Linear(unit_num, 1))
            self.h_layers.append(nn.Sigmoid())

        def forward(self, x):
            for h in self.h_layers:
                x = h(x)
            return x

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    train_loader, test_loader = _create_dataset(batch_size)

    def _eval():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = net(inputs.to(device))
                predicted = torch.where(outputs > 0.5, 1, 0).cpu()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    print("Training start")
    total_time = 0
    acc_list = []
    for epoch in tqdm(range(epochs)):
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            t0 = time.time()
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_time += time.time() - t0

            acc_list.append(_eval())
    print(f"Finished Training {total_time}s")

    return acc_list, total_time


def train_single_ed_model(layer_num, unit_num, activation, lr, batch_size, epochs):
    layers = [(unit_num, activation) for _ in range(layer_num)]
    model = EDModelSingle(
        input_num=28 * 28,
        layers=layers,
        out_type="sigmoid",
        lr=lr,
        device=device,
    )

    train_loader, test_loader = _create_dataset(batch_size)

    def _eval():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs.to(device))
                predicted = torch.where(outputs > 0.5, 1, 0).cpu()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    print("Training start")
    total_time = 0
    acc_list = []
    for epoch in tqdm(range(epochs)):
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            t0 = time.time()
            model.train(inputs, labels)
            total_time += time.time() - t0

            acc_list.append(_eval())
    print(f"Finished Training {total_time}s")

    return acc_list, total_time


def train_ed_model(layer_num, unit_num, activation, lr, batch_size, epochs):
    layers = [(unit_num, activation) for _ in range(layer_num)]
    model = EDModel(
        input_num=28 * 28,
        output_num=1,
        layers=layers,
        out_type="sigmoid",
        training_mode="mse",
        lr=lr,
        device=device,
    )

    train_loader, test_loader = _create_dataset(batch_size)

    def _eval():
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                outputs = model(inputs.to(device))
                predicted = torch.where(outputs > 0.5, 1, 0).cpu()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        return correct / total

    print("Training start")
    total_time = 0
    acc_list = []
    for epoch in tqdm(range(epochs)):
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            t0 = time.time()
            model.train(inputs, labels)
            total_time += time.time() - t0

            acc_list.append(_eval())
    print(f"Finished Training {total_time}s")

    return acc_list, total_time


def main():
    layer_num = 5
    unit_num = 32
    activation = "sigmoid"
    lr = 0.01
    batch_size = 128
    epochs = 10

    name_list = []
    time_list = []

    acc_list, total_time = train_torch_model(layer_num, unit_num, activation, lr, batch_size, epochs)
    plt.plot(acc_list, label="Torch")
    name_list.append("Torch")
    time_list.append(total_time)

    acc_list, total_time = train_single_ed_model(layer_num, unit_num, activation, lr, batch_size, epochs)
    plt.plot(acc_list, label="EDMethod(single)")
    name_list.append("EDMethod(single)")
    time_list.append(total_time)

    acc_list, total_time = train_ed_model(layer_num, unit_num, activation, lr, batch_size, epochs)
    plt.plot(acc_list, label="EDMethod")
    name_list.append("EDMethod")
    time_list.append(total_time)

    plt.ylim(0.49, 1)
    plt.grid()
    plt.legend()
    plt.show()

    plt.bar(name_list, time_list)
    plt.show()


if __name__ == "__main__":
    main()
