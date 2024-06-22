from typing import Tuple

import click
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchinfo import summary
from PIL import Image
import pandas as pd


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
    def __init__(self, dense_units=2048, use_encoder_dense=False):
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
            nn.ReLU(),
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
    scaleRange: Tuple[float, float] = (0.2, 1.0),
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
            transforms.RandomApply([transforms.ColorJitter(*colorJitter)]),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=guassianKernelSize)]
            ),
            transforms.ToTensor(),
        ]
    )


def nt_xent_slow(x, t=0.5):
    """
    Return NT-Xent loss given the batch. This implementation is pretty slow but
    each step is very explicit for pedagogical purposes. See other
    implementation for a more optimized version (using cross_entropy()
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

    return loss / x.shape[0]


def nt_xent(x, t=0.5):
    """https://github.com/p3i0t/SimCLR-CIFAR10/blob/2f449c2e39666a5c3439859347e3f1aced67b17d/simclr.py#L52C1-L65C71"""
    x = F.normalize(x, dim=1)
    x_scores = x @ x.t()  # normalized cosine similarity scores
    x_scale = x_scores / t  # scale with temperature

    # (2N-1)-way softmax without the score of i-th entry itself.
    # Set the diagonals to be large negative values, which become zeros after softmax.
    x_scale = x_scale - torch.eye(x_scale.size(0)).to(x_scale.device) * 1e5

    # targets 2N elements.
    targets = torch.arange(x.size()[0])
    targets[::2] += 1  # target of 2k element is 2k+1
    targets[1::2] -= 1  # target of 2k+1 element is 2k
    return F.cross_entropy(x_scale, targets.long().to(x_scale.device))


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
    print(summary(model, input_size=(1, 3, 32, 32)))

    cifarTrain = CIFAR10(root="./data", train=True, download=True, transform=ToTensor())
    cifarTest = CIFAR10(root="./data", train=False, download=True, transform=ToTensor())

    trainLoader = DataLoader(cifarTrain, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(cifarTest, batch_size=batch_size, shuffle=False)

    lossFun = nn.CrossEntropyLoss()
    sgd = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        trainLoss = 0.0
        trainAcc = 0.0

        model.train()

        with click.progressbar(trainLoader, label="Training") as bar:
            for i, (images, labels) in enumerate(bar):
                images = images.cuda()
                labels = labels.cuda()

                # Zero out gradients
                sgd.zero_grad()

                # Get outputs
                output = model(images)

                # Calculate loss
                loss = lossFun(output, labels)

                # Record metrics
                trainLoss += loss.item()
                trainAcc += (output.argmax(1) == labels).sum().item()

                # Update weights
                loss.backward()
                sgd.step()

        trainLoss /= len(cifarTrain)
        trainAcc /= len(cifarTrain)

        # Validation loop
        model.eval()
        with torch.no_grad():
            testLoss = 0.0
            testAcc = 0.0

            with click.progressbar(testLoader, label="Testing") as bar:
                for images, labels in bar:
                    images = images.cuda()
                    labels = labels.cuda()

                    output = model(images)
                    loss = lossFun(output, labels)

                    testLoss += loss.item()
                    testAcc += (output.argmax(1) == labels).sum().item()

            testLoss /= len(testLoader)
            testAcc /= len(testLoader)

        print(
            f"Epoch {epoch}: Train Loss: {trainLoss}, Train Accuracy: {trainAcc}, Test Loss: {testLoss}, Test Accuracy: {testAcc}"
        )

    # Save model
    torch.save(model.state_dict(), f"supervised.pth")


@cli.command()
@click.option("--batch_size", default=128, help="Batch size")
@click.option(
    "--scale_range",
    default=(0.2, 1.0),
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
@click.option("--learning_rate", default=0.01, help="Learning rate")
@click.option("--momentum", default=0.9, help="Momentum")
@click.option("--weight_decay", default=1e-5, help="Weight decay")
def unsupervised(
    batch_size: int = 128,
    scale_range: Tuple[float, float] = (0.2, 1.0),
    ratio_range: Tuple[float, float] = (0.75, 1.33),
    color_jitter: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 0.2),
    guassian_kernel_size: int = 3,
    epochs: int = 20,
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
    print(summary(model, input_size=(1, 3, 32, 32)))

    augTransform = simCLR_transform(
        scaleRange=scale_range,
        ratioRange=ratio_range,
        colorJitter=color_jitter,
        guassianKernelSize=guassian_kernel_size,
    )
    cifarTrain = CIFAR10_Paired(
        root="./data", train=True, download=True, transform=augTransform
    )
    cifarTest = CIFAR10_Paired(
        root="./data", train=False, download=True, transform=augTransform
    )

    trainLoader = DataLoader(cifarTrain, batch_size=batch_size, shuffle=True)
    testLoader = DataLoader(cifarTest, batch_size=batch_size, shuffle=False)

    lossFun = nt_xent_slow
    sgd = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for epoch in range(epochs):
        trainLoss = 0.0

        model.train()

        with click.progressbar(trainLoader, label="Training") as bar:
            for i, (images, labels) in enumerate(bar):
                # Reshape images to go through the model (every 2 images are a positive pair)
                shape = images.shape
                images = images.view(shape[0] * 2, shape[2], shape[3], shape[4])
                images = images.cuda()

                # Zero out gradients
                sgd.zero_grad()

                # Get outputs
                output = model(images)

                # Calculate loss
                loss = lossFun(output)
                print(f"Step loss: {loss.item()}")

                # Record metrics
                trainLoss += loss.item()

                # Update weights
                loss.backward()
                sgd.step()

        trainLoss /= len(trainLoader)
        print(f"Epoch {epoch + 1}: Train Loss: {trainLoss}")


if __name__ == "__main__":
    cli()
