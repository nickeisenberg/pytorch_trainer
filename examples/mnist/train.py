import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from tqdm import tqdm

from src.trfc.trainer import Trainer
from src.trfc.callbacks.progress_bar.progress_bar import ProgressBar


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

progress_bar = ProgressBar()

class TMod(nn.Module):
    def __init__(self):
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


mnist = MNIST(os.path.expanduser("~/datasets/mnist"), transform=ToTensor())
train_loader = DataLoader(Subset(mnist, range(50000)), batch_size=64, num_workers=19)
val_loader = DataLoader(Subset(mnist, range(50000, 60000)), batch_size=64, num_workers=19)

trainer_mod = TMod()
trainer = Trainer(trainer_mod, callbacks=[progress_bar])

trainer.fit(train_loader, 2, val_loader)
