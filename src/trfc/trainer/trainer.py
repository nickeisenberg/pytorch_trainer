import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    def __init__(self, trainer_module: nn.Module):
        self.trainer_module = trainer_module 
        
        self.current_epoch = 0 
        self.which_pass = "N/A" 


    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            val_loader: DataLoader | None = None):

        self.call("before_all_epochs")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self.which_pass = "train" 

            self.pbar = tqdm(train_loader, leave=True)

            self.call(f"before_{self.which_pass}_epoch_pass")
            self.epoch_pass(loader=self.pbar)
            self.call(f"after_{self.which_pass}_epoch_pass")

            if val_loader is not None:
                self.which_pass = "validation"
                self.pbar = tqdm(val_loader, leave=True)

                self.call(f"before_{self.which_pass}_epoch_pass")
                self.epoch_pass(loader=self.pbar)
                self.call(f"after_{self.which_pass}_epoch_pass")

        self.call("after_all_epochs")


    def epoch_pass(self, loader: tqdm):

        batch_pass = getattr(self.trainer_module, f"{self.which_pass}_batch_pass")
        for batch_idx, data in enumerate(loader):
            self.call(f"before_{self.which_pass}_batch_pass")
            batch_pass(data)
            self.call(f"after_{self.which_pass}_batch_pass")


    def call(self, where_at):
        for callback in self.trainer_module.callbacks():
            if hasattr(callback, where_at):
                method = getattr(callback, where_at)
                method(self)


    def call2(self, where_at):
        for callback in self.trainer_module.callbacks():
            if callback._callbacks[where_at] is not None:
                callback._callbacks[where_at](self)
