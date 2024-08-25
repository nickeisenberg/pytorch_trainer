import os
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from src.trnr.trainer import Trainer
from src.trnr.callbacks.base import Callback
from src.trnr.callbacks.data_iterator.progress_bar import ProgressBar
from src.trnr.callbacks.save_best_checkpoint import SaveBestCheckpoint
from src.trnr.callbacks.basic_lr_scheduler import BasicLRScheduler 
from src.trnr.callbacks.logger.classification_logger import ClassificationLogger

class DummyCallback(Callback):
    def __init__(self):
        super().__init__()

    def after_train_epoch_pass(self, trainer: Trainer):
        print("after_train_epoch_pass")
        print("LR", trainer.module.optim.param_groups[0]["lr"])


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


class Module(nn.Module):
    def __init__(self, 
                 progress_bar: ProgressBar, 
                 logger: ClassificationLogger):
        super().__init__()

        self.classifier = Classifier()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=.0001)

        self.cross_entropy = nn.CrossEntropyLoss()

        self.progress_bar = progress_bar 
        self.logger = logger 

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

        self.logger.log_item("loss", loss.item())
        self.logger.log_targs_and_preds(targets=labs, predictions=pred_labels)
        
        num_correct = (self.logger.epoch_predictions == self.logger.epoch_targets).sum()
        accuracy = round(float(num_correct * 100 / len(self.logger.epoch_predictions)), 2)

        avg_loss = sum(self.logger.epoch_history["loss"])
        avg_loss /= len(self.logger.epoch_history["loss"])
        avg_loss = round(avg_loss * 100, 2)

        self.progress_bar.log("avg_loss", avg_loss)
        self.progress_bar.log("acc", accuracy)


    def validation_batch_pass(self, batch, batch_idx):
        if self.classifier.training:
            self.classifier.eval() 

        imgs, labs = batch
        preds = self.classifier(imgs)
        pred_labels = torch.argmax(preds, dim=1)
        loss = self.cross_entropy(preds, labs)

        self.logger.log_item("loss", loss.item())
        self.logger.log_targs_and_preds(targets=labs, predictions=pred_labels)

        num_correct = (self.logger.epoch_predictions == self.logger.epoch_targets).sum()
        accuracy = round(float(num_correct * 100 / len(self.logger.epoch_predictions)), 2)

        avg_loss = sum(self.logger.epoch_history["loss"])
        avg_loss /= len(self.logger.epoch_history["loss"])
        avg_loss = round(avg_loss * 100, 2)

        self.progress_bar.log("avg_loss", avg_loss)
        self.progress_bar.log("acc", accuracy)


def loaders():
    mnist = MNIST(os.path.expanduser("~/datasets/mnist"), transform=ToTensor(), download=True)
    train_dataset = Subset(mnist, range(50000))
    val_dataset = Subset(mnist, range(50000, 60000))
    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=19, shuffle=True
    )
    validation_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=True, num_workers=19
    )
    return train_loader, validation_loader

def get_trainer():
    progress_bar = ProgressBar(log_to_bar_every=15)
    logger = ClassificationLogger(labels=list(range(10)))
    dummy = DummyCallback()
    save_best_checkpoint = SaveBestCheckpoint("loss", "decrease", "loss", "decrease")
    module = Module(progress_bar, logger)
    scheduler = BasicLRScheduler(ExponentialLR(module.optim, gamma=.8))
    trainer = Trainer(
        module, 
        device="gpu",
        ddp=False,
        callbacks=[
            progress_bar, logger, save_best_checkpoint, scheduler
        ],
        save_root="examples/mnist/mnist_classifier"
    )
    return trainer

def main():
    trainer = get_trainer()
    train_loader, validation_loader = loaders()
    data_devicer = lambda batch, device: (batch[0].to(device), batch[1].to(device))
    trainer.fit(
        train_loader=train_loader, 
        num_epochs=2, 
        train_data_devicer=data_devicer,
        validation_loader=validation_loader,
        validation_data_devicer=data_devicer,
    )

if __name__ == "__main__":
    main()
