from examples.mnist.module import Module

import os

import torch
from torch.utils.data import DataLoader, Subset

from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST

from src.trnr.trainer import Trainer
from src.trnr.callbacks.data_iterator.progress_bar import ProgressBar
from src.trnr.callbacks.save_best_checkpoint import SaveBestCheckpoint
from src.trnr.callbacks.logger.classification_logger import ClassificationLogger


def loaders():
    mnist = MNIST(os.path.expanduser("~/datasets/mnist"), transform=ToTensor(), download=True)
    dataset = Subset(mnist, range(50000, 60000))
    loader = DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=19
    )
    return loader


def get_trainer():
    progress_bar = ProgressBar(log_to_bar_every=15)
    logger = ClassificationLogger(labels=list(range(10)))
    save_best_checkpoint = SaveBestCheckpoint("loss", "decrease", "loss", "decrease")
    try:
        try:
            sd = torch.load(
                os.path.expanduser(
                    "./examples/mnist/mnist_classifier/state_dicts/validation_ep_2.pth"
                )
            )
        except:
            sd = torch.load(
                os.path.expanduser(
                    "./mnist_classifier/state_dicts/validation_ep_2.pth"
                )
            )

    except Exception as e:
        raise e

    module = Module(progress_bar, logger)
    module.load_state_dict(sd)
    trainer = Trainer(
        module, 
        device="gpu",
        ddp=False,
        callbacks=[
            progress_bar, logger, save_best_checkpoint
        ],
        save_root="examples/mnist/mnist_classifier_evaluation"
    )
    return trainer


def main():
    trainer = get_trainer()
    loader = loaders()
    data_devicer = lambda batch, device: (batch[0].to(device), batch[1].to(device))
    trainer.evaluate(
        loader=loader, 
        data_devicer=data_devicer
    )


if __name__ == "__main__":
    main()
