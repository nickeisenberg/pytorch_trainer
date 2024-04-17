from sys import path

path.append(__file__.split("examples")[0])

from config import config_trainer
from trfc import Trainer

config = config_trainer()

if __name__ == "__main__":
    trainer = Trainer(
        train_module=config["train_module"],
    )

    trainer.fit(
        train_loader=config["train_loader"],
        num_epochs=config["num_epochs"],
        val_loader=config["val_loader"]
    )
