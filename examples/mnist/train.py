import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from src.trfc.trainer import Trainer
from src.trfc.callbacks.progress_bar.progress_bar import ProgressBar
from src.trfc.callbacks.base import Callback


class DummyCallback(Callback):
    def __init__(self):
        super().__init__()
    
    def on_fit_start(self, trainer):
        print("on_fit_start")

    def before_train_epoch_pass(self, trainer):
        print("before_train_epoch_pass")

    def after_train_epoch_pass(self, trainer):
        print("after_train_epoch_pass")

    def before_validation_epoch_pass(self, trainer):
        print("before_validation_epoch_pass")

    def after_validation_epoch_pass(self, trainer):
        print("after_validation_epoch_pass")

    def on_fit_end(self, trainer):
        print("on_fit_end")

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 0)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 0)
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 0)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(30976, 256)
        self.linear2 = nn.Linear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.linear1(self.flatten(x)))
        return self.linear2(x)

class TMod(nn.Module):
    def __init__(self, progress_bar):
        super().__init__()

        self.classifier = Classifier()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=.0001)

        self.progress_bar = progress_bar 

    def forward(self, x):
        return self.classifier(x)

    def train_batch_pass(self, batch, batch_idx):
        if not self.classifier.training:
            self.classifier.train() 

        imgs, labs = batch

        self.optim.zero_grad()
        preds = self.classifier(imgs)
        pred_labels = torch.argmax(preds, dim=1)
        loss = self.cross_entropy(preds, labs)
        loss.backward()
        self.optim.step()

        accuracy = round(float((pred_labels == labs).sum() * 100 / len(labs)), 2)

        self.progress_bar.log("accuracy", accuracy)
        self.progress_bar.log("loss", loss.item())

    def validation_batch_pass(self, batch, batch_idx):
        if self.classifier.training:
            self.classifier.eval() 

        imgs, labs = batch
        preds = self.classifier(imgs)
        pred_labels = torch.argmax(preds, dim=1)

        loss = self.cross_entropy(preds, labs)
        accuracy = round(float((pred_labels == labs).sum() * 100 / len(labs)), 2)

        self.progress_bar.log("accuracy", accuracy)
        self.progress_bar.log("loss", loss.item())

def loaders():
    mnist = MNIST(os.path.expanduser("~/datasets/mnist"), transform=ToTensor())
    train_dataset = Subset(mnist, range(50000))
    val_dataset = Subset(mnist, range(50000, 60000))
    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=19, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=True, num_workers=19
    )
    return train_loader, val_loader

def get_trainer():
    progress_bar = ProgressBar(log_to_bar_every=15)
    dummy = DummyCallback()
    module = TMod(progress_bar)
    trainer = Trainer(
        module, 
        device="gpu",
        ddp=False,
        callbacks=[progress_bar, dummy]
    )
    return trainer

def main(trainer):
    train_loader, val_loader = loaders()
    data_devicer = lambda batch, device: (batch[0].to(device), batch[1].to(device))
    trainer.fit(
        train_loader=train_loader, 
        num_epochs=2, 
        data_devicer=data_devicer,
        val_loader=val_loader
    )

main(get_trainer())
