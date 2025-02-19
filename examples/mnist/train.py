from examples.mnist.module import Module

import os

from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ExponentialLR

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from src.trnr.trainer import Trainer
from src.trnr.callbacks.data_iterator.progress_bar import ProgressBar
from src.trnr.callbacks.save_best_checkpoint import SaveBestCheckpoint
from src.trnr.callbacks.basic_lr_scheduler import BasicLRScheduler 
from src.trnr.callbacks.logger.classification_logger import ClassificationLogger


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
    save_best_checkpoint = SaveBestCheckpoint("loss", "decrease", "loss", "decrease")
    module = Module(progress_bar, logger)
    scheduler = BasicLRScheduler(ExponentialLR(module.optim, gamma=.8))
    trainer = Trainer(
        module=module, 
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
