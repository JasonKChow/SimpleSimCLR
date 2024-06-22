import click
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchinfo import summary


class cnn(nn.Module):
    def __init__(self):
        super().__init__()

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
        x = self.classifier(x)

        return x


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
    epochs=20, batch_size=128, learning_rate=0.01, momentum=0.9, weight_decay=1e-5
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

    loss = nn.CrossEntropyLoss()
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
                lossVal = loss(output, labels)

                # Record metrics
                trainLoss += lossVal.item()
                trainAcc += (output.argmax(1) == labels).sum().item()

                # Update weights
                lossVal.backward()
                sgd.step()

        trainLoss /= len(cifarTrain)
        trainAcc /= len(cifarTrain)

        model.eval()
        with torch.no_grad():
            testLoss = 0.0
            testAcc = 0.0

            with click.progressbar(testLoader, label="Testing") as bar:
                for images, labels in bar:
                    images = images.cuda()
                    labels = labels.cuda()

                    output = model(images)
                    lossVal = loss(output, labels)

                    testLoss += lossVal.item()
                    testAcc += (output.argmax(1) == labels).sum().item()

            testLoss /= len(cifarTest)
            testAcc /= len(cifarTest)

        print(
            f"Epoch {epoch}: Train Loss: {trainLoss}, Train Accuracy: {trainAcc}, Test Loss: {testLoss}, Test Accuracy: {testAcc}"
        )

    # Save model
    torch.save(model.state_dict(), f"supervised.pth")


if __name__ == "__main__":
    cli()
