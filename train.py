from typing import Tuple

import click
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CIFAR10
from torchinfo import summary
from PIL import Image
import pandas as pd
import numpy as np


class cnn(nn.Module):
    def __init__(self, include_classifier=True):
        """
        Create a small CNN model based on the All-CNN-C architecture.
        """
        super().__init__()
        self.include_classifier = include_classifier

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.block1.apply(self._init_weights)

        self.block2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.block2.apply(self._init_weights)

        if include_classifier:
            self.classifier = nn.Sequential(
                nn.Conv2d(192, 192, 3, stride=1, padding="valid"),
                nn.ReLU(),
                nn.Conv2d(192, 192, 1, stride=1, padding="same"),
                nn.ReLU(),
                nn.Conv2d(192, 10, 1, stride=1, padding="same"),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier.apply(self._init_weights)

    def _init_weights(module, layer):
        # Initialize weights with He Normal
        if isinstance(layer, nn.Conv2d):
            nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
            layer.bias.data.fill_(0)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        if self.include_classifier:
            x = self.classifier(x)

        return x


class cnn_simCLR(nn.Module):
    def __init__(self, dense_units=128, use_encoder_dense=False):
        """
        Create a small CNN model based on the All-CNN-C architecture and set it
        up for simCLR training. The dense layers are attached right before the
        classifier, unless use_encoder_dense is set to True.
        """
        super().__init__()

        if use_encoder_dense:
            raise NotImplementedError("Using encoder dense layers not implemented yet")

        self.encoder = cnn(include_classifier=False)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(dense_units),
            nn.ReLU(),
            nn.LazyLinear(dense_units),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)

        return x


class CIFAR10_Paired(CIFAR10):
    def __getitem__(self, idx):
        """
        Return two copies of the same image for simCLR training.
        """
        # Get the image and target
        img, target = self.data[idx], self.targets[idx]

        # Convert to PIL image and apply transformations twice
        img = Image.fromarray(img)
        imgs = torch.stack([self.transform(img), self.transform(img)])

        return imgs, target


def simCLR_transform(
    scaleRange: Tuple[float, float] = (0.1, 1.0),
    ratioRange: Tuple[float, float] = (0.75, 1.33),
    colorJitter: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.2),
    guassianKernelSize: int = 3,
) -> transforms.Compose:
    """
    Return a composed set of augmentation transformations for simCLR training.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(32, scale=scaleRange, ratio=ratioRange),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(*colorJitter)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=guassianKernelSize)]
            ),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
        ]
    )


def nt_xent(x, t=0.5):
    """
    Return NT-Xent loss given the batch. This implementation may be slow but
    each step is very explicit for pedagogical purposes. See other
    implementations for a more optimized version (using cross_entropy()
    directly).
    """
    # Scaled (t) Cosine similarity between every pair
    x = F.normalize(x, dim=1)
    sim = (x @ x.T) / t

    # Exponentiate it all for softmax later
    sim = torch.exp(sim)

    # Loop through each element to calculate the loss
    loss = 0.0
    for idx in range(x.shape[0] // 2):
        i = idx * 2
        j = idx * 2 + 1

        # Get the postive pair from i, j
        pos = sim[i, j]

        # Get the negative pairs for i, k where k != i
        neg = sim[i, torch.arange(x.shape[0]) != i]
        neg = torch.sum(neg)

        # Calculate the loss
        loss += -torch.log(pos / (pos + neg))

        # Get the postive pair from j, i
        pos = sim[j, i]

        # Get the negative pairs for j, k where k != j
        neg = sim[j, torch.arange(x.shape[0]) != j]
        neg = torch.sum(neg)

        # Calculate the loss
        loss += -torch.log(pos / (pos + neg))

    # Mean of the function across the batch
    return loss / x.shape[0]


@click.group()
def cli():
    pass


@cli.command()
@click.option("--epochs", default=20, help="Number of epochs")
@click.option("--batch_size", default=128, help="Batch size")
@click.option("--learning_rate", default=0.01, help="Learning rate")
@click.option("--momentum", default=0.9, help="Momentum")
@click.option("--weight_decay", default=1e-5, help="Weight decay")
def supervised(
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
):
    """
    Train a small CNN on CIFAR-10 dataset using a completely supervised
    strategy. Architecture is modeled after All-CNN-C.
    """
    model = cnn()
    click.echo(summary(model, input_size=(1, 3, 32, 32)))

    transform = transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]
    )
    cifarTrain = CIFAR10(root="./data", train=True, download=True, transform=transform)
    cifarTest = CIFAR10(root="./data", train=False, download=True, transform=transform)

    trainLoader = DataLoader(cifarTrain, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(cifarTest, batch_size=batch_size, shuffle=False)

    lossFun = nn.CrossEntropyLoss()
    sgd = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Create csv file to store the results
    trainingLog = pd.DataFrame(
        columns=["Epoch", "Train_Loss", "Train_Accuracy", "Test_Loss", "Test_Accuracy"]
    )

    for epoch in range(epochs):
        trainLosses = []
        trainAccs = []

        model.train()

        with click.progressbar(
            length=len(trainLoader),
            label="Train",
            item_show_func=lambda x: (
                f"Loss: {x[0]:5.3f}  Acc: {x[1]:5.3f}" if x is not None else None
            ),
        ) as bar:
            for images, labels in trainLoader:
                images = images.cuda()
                labels = labels.cuda()

                # Zero out gradients
                sgd.zero_grad()

                # Get outputs
                output = model(images)

                # Calculate loss
                loss = lossFun(output, labels)

                # Record metrics
                trainLosses += [loss.item()]
                trainAccs += [(output.argmax(1) == labels).sum().item() / len(labels)]

                # Update weights
                loss.backward()
                sgd.step()

                bar.update(
                    1,
                    [
                        np.mean(trainLosses),
                        np.mean(trainAccs),
                    ],
                )

        # Validation loop
        model.eval()
        with torch.no_grad():
            testLoss = 0.0
            testAcc = 0.0

            with click.progressbar(testLoader, label="Test ") as bar:
                for images, labels in bar:
                    images = images.cuda()
                    labels = labels.cuda()

                    output = model(images)
                    loss = lossFun(output, labels)

                    testLoss += loss.item()
                    testAcc += (output.argmax(1) == labels).sum().item()

            testLoss /= len(testLoader)
            testAcc /= len(cifarTest)

        click.echo(
            f"Epoch {epoch + 1}: Train Loss: {np.mean(trainLosses):5.3f}, Train Accuracy: {np.mean(trainAccs):5.3f}, Test Loss: {testLoss:5.3f}, Test Accuracy: {testAcc:5.3f}"
        )

        # Add data to csv
        trainingLog = pd.concat(
            [
                trainingLog,
                pd.DataFrame(
                    {
                        "Epoch": [epoch],
                        "Train_Loss": [np.mean(trainLosses)],
                        "Train_Accuracy": [np.mean(trainAccs)],
                        "Test_Loss": [testLoss],
                        "Test_Accuracy": [testAcc],
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Save model
    torch.save(model.state_dict(), f"supervised.pth")

    # Save csv
    trainingLog.to_csv("supervisedTrainingLog.csv", index=False)


@cli.command()
@click.option("--batch_size", default=128, help="Batch size")
@click.option(
    "--scale_range",
    default=(0.1, 1.0),
    nargs=2,
    help="Scale range for random resized crop",
    type=(float, float),
)
@click.option(
    "--ratio_range",
    default=(0.75, 1.33),
    nargs=2,
    help="Ratio range for random resized crop",
    type=(float, float),
)
@click.option(
    "--color_jitter",
    default=(0.8, 0.8, 0.8, 0.2),
    nargs=4,
    help="Color jitter parameters",
    type=(float, float, float, float),
)
@click.option("--guassian_kernel_size", default=3, help="Gaussian kernel size")
@click.option("--epochs", default=20, help="Number of epochs")
@click.option("--temperature", default=0.5, help="Temperature for NT-Xent loss")
@click.option("--learning_rate", default=0.01, help="Learning rate")
@click.option("--momentum", default=0.9, help="Momentum")
@click.option("--weight_decay", default=1e-5, help="Weight decay")
def unsupervised(
    batch_size: int = 128,
    scale_range: Tuple[float, float] = (0.1, 1.0),
    ratio_range: Tuple[float, float] = (0.75, 1.33),
    color_jitter: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.2),
    guassian_kernel_size: int = 3,
    epochs: int = 20,
    temperature: float = 0.5,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
):
    """
    Train a small CNN on CIFAR-10 dataset using the simCLR strategy.
    Architecture is modeled after All-CNN-C with the dense layers attached right
    before the dense layers.
    """
    model = cnn_simCLR()
    click.echo(summary(model, input_size=(1, 3, 32, 32)))

    augTransform = simCLR_transform(
        scaleRange=scale_range,
        ratioRange=ratio_range,
        colorJitter=color_jitter,
        guassianKernelSize=guassian_kernel_size,
    )
    cifarTrain = CIFAR10_Paired(
        root="./data", train=True, download=True, transform=augTransform
    )

    trainLoader = DataLoader(cifarTrain, batch_size=batch_size, shuffle=True)

    lossFun = lambda x: nt_xent(x, t=temperature)
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=learning_rate,
    #     momentum=momentum,
    #     weight_decay=weight_decay,
    # )
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=len(trainLoader), eta_min=0, last_epoch=-1
    # )

    trainingLog = pd.DataFrame(columns=["Epoch", "Train_Loss"])
    for epoch in range(epochs):
        trainLosses = []

        model.train()
        with click.progressbar(
            length=len(trainLoader),
            label="Train",
            item_show_func=lambda x: (f"Loss: {x:5.3f}" if x is not None else None),
        ) as bar:
            for images, _ in trainLoader:
                # Reshape images to go through the model (every 2 images are a positive pair)
                shape = images.shape
                images = images.view(shape[0] * 2, shape[2], shape[3], shape[4])
                images = images.cuda()

                # Zero out gradients
                optimizer.zero_grad()

                # Get outputs
                output = model(images)

                # Calculate loss
                loss = lossFun(output)

                # Record metric
                trainLosses += [loss.item()]

                # Update weights
                loss.backward()
                optimizer.step()
                # scheduler.step()

                bar.update(
                    1,
                    np.mean(trainLosses),
                )

        click.echo(f"Epoch {epoch + 1}: Train Loss: {np.mean(trainLosses):5.3f}")

        # Add data to csv
        trainingLog = pd.concat(
            [
                trainingLog,
                pd.DataFrame(
                    {
                        "Epoch": [epoch],
                        "Train_Loss": [np.mean(trainLosses)],
                    },
                    index=[0],
                ),
            ],
            ignore_index=True,
        )

    # Save model
    torch.save(model.state_dict(), f"unsupervised.pth")
    trainingLog.to_csv("unsupervisedTrainingLog.csv", index=False)


if __name__ == "__main__":
    cli()
