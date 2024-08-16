import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from trnr.trainer import Trainer
from trnr.callbacks.data_iterator.progress_bar import ProgressBar
from trnr.callbacks.logger.csv_logger import CSVLogger
from trnr.callbacks.save_best_checkpoint import SaveBestCheckpoint


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
    def __init__(self, progress_bar: ProgressBar, logger: CSVLogger):
        super().__init__()

        self.classifier = Classifier()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(self.classifier.parameters(), lr=.0001)

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

        accuracy = round(float((pred_labels == labs).sum() * 100 / len(labs)), 2)

        self.progress_bar.log("accuracy", accuracy)
        self.progress_bar.log("loss", loss.item())

        self.logger.log("loss", loss.item())
        self.logger.log("accuracy", accuracy)

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

        self.logger.log("loss", loss.item())
        self.logger.log("accuracy", accuracy)

def get_loaders():
    mnist = MNIST(os.path.expanduser("~/datasets/mnist"), transform=ToTensor())
    train_dataset = Subset(mnist, range(50000))
    val_dataset = Subset(mnist, range(50000, 60000))
    train_loader = DataLoader(
        train_dataset, batch_size=64, num_workers=19, 
        shuffle=False, sampler=DistributedSampler(train_dataset)
    )
    validation_loader = DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=19, 
        sampler=DistributedSampler(val_dataset)
    )
    return train_loader, validation_loader

def get_trainer():
    progress_bar = ProgressBar(log_to_bar_every=15)
    logger = CSVLogger("logs")
    save_best_checkpoint = SaveBestCheckpoint("loss", "decrease", "loss", "decrease")
    module = Module(progress_bar, logger)
    return Trainer(
        module, 
        device="gpu",
        ddp=True,
        callbacks=[progress_bar, logger, save_best_checkpoint],
        save_root="examples/mnist_ddp/mnist_classifier"
    )

def main():
    init_process_group(backend="nccl")
    train_loader, validation_loader = get_loaders()
    trainer = get_trainer()
    data_devicer = lambda batch, device: (batch[0].to(device), batch[1].to(device))
    trainer.fit(
        train_loader=train_loader, 
        train_data_devicer=data_devicer,
        num_epochs=2, 
        validation_loader=validation_loader,
        validation_data_devicer=data_devicer,
    )
    destroy_process_group()

if __name__ == "__main__":
    main()
