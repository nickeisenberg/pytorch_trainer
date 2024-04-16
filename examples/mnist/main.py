import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from src.trainer.trainer import Trainer
from src.callbacks import (
    Accuracy,
    CSVLogger,
    SaveBestCheckoint,
    ConfusionMatrix,
    ProgressBarUpdater
)


class MnistClassifier(nn.Module):
    """
    Mnist images are (1, 28, 28)
    """
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 2)
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(3872, 256)
        self.lin2 = nn.Linear(256, 10)

        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.lin1(x)
        x = self.lin2(x)
        return x


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, predictions, targets) -> tuple[torch.Tensor, dict]:
        loss = self.entropy(predictions, targets)

        history = {
            "total_loss": loss.item(),
        }
        return loss, history


mnist = MNIST(
    "/mnt/c/Users/EISENBNT/Datasets/MNIST", 
    train=True,
    transform=ToTensor(),
    download=True
)
tloader = DataLoader(Subset(mnist, range(50000)), 64)
vloader = DataLoader(Subset(mnist, range(50000, 60000)), 64)
unpacker = lambda data, device: (data[0].to(device), data[1].to(device))

model = MnistClassifier()

accuracy = Accuracy()
conf_mat = ConfusionMatrix([*range(10)], "./examples/mnist/metrics")
logger = CSVLogger("./examples/mnist/loss_logs")
save_best = SaveBestCheckoint("./examples/mnist/state_dicts", "total_loss")
pbar_updater = ProgressBarUpdater()


class TrainModule(nn.Module):
    def __init__(self):

        super().__init__()
        
        self.model = model.to("cuda")

        self.loss_fn = Loss()
        self.optimizer = Adam(self.model.parameters(), lr=.0001)
        
        self.accuracy = accuracy 
        self.conf_mat = conf_mat 
        self.logger = logger
        self.save_best = save_best
        self.progress_bar_updater = pbar_updater 

    def callbacks(self):
        return [
            self.accuracy,
            self.conf_mat,
            self.logger,
            self.save_best,
            self.progress_bar_updater
        ]


    def forward(self, x):
        return self.model(x)


    def train_batch_pass(self, *unpacked_loader_data):
        if not self.model.training:
            self.model.train()

        inputs, targets = unpacked_loader_data 
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss, history = self.loss_fn.forward(outputs, targets)
        loss.backward()
        self.optimizer.step()

        targets = targets.detach()
        predictions = torch.argmax(outputs, 1).detach()

        self.accuracy.log(predictions, targets)
        self.conf_mat.log(predictions, targets)

        history["accuracy"] = self.accuracy.accuracy
        self.logger.log(history)


    def validation_batch_pass(self, *unpacked_loader_data):
        if self.model.training:
            self.model.eval()

        inputs, targets = unpacked_loader_data

        with torch.no_grad():
            outputs: torch.Tensor = self.model(inputs)

        _, history = self.loss_fn(outputs, targets)

        targets = targets.detach()
        predictions = torch.argmax(outputs, 1).detach()

        self.accuracy.log(predictions, targets)
        self.conf_mat.log(predictions, targets)

        history["accuracy"] = self.accuracy.accuracy
        self.logger.log(history)


train_module = TrainModule()

trainer = Trainer(train_module)

trainer.fit(
    train_loader=tloader,
    num_epochs=2,
    device="cuda",
    unpacker=unpacker,
    val_loader=vloader
)
