from collections.abc import Callable
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks.base import Callback
from ..callbacks.progress_bar import ProgressBar


class Trainer:
    def __init__(self, 
                 trainer_module: nn.Module,
                 callbacks: list[Callback]):

        self.trainer_module = trainer_module 
        
        # register all callbacks
        self._register_callbacks(callbacks)

    @property
    def callbacks(self):
        return {
            "on_fit_start": [],
            "before_train_epoch_pass": [],
            "before_train_batch_pass": [],
            "after_train_batch_pass": [],
            "after_train_epoch_pass": [],
            "before_validation_epoch_pass": [],
            "before_validation_batch_pass": [],
            "after_validation_batch_pass": [],
            "after_validation_epoch_pass": [],
            "on_fit_end": [],
        }

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            val_loader: DataLoader | None = None):
        
        # some "env" variables
        self.current_epoch = 0 
        self.which_pass = "N/A" 

        self.call("on_fit_start", self, train_loader=train_loader, 
                  num_epochs=num_epochs, val_loader=val_loader)

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            self.which_pass = "train" 

            self.call(f"before_{self.which_pass}_epoch_pass", self)
            self.epoch_pass(
                loader=self.progress_bar_callback.train_progress_bar,
                batch_pass=getattr(self.trainer_module, f"{self.which_pass}_batch_pass")
            )
            self.call(f"after_{self.which_pass}_epoch_pass", self)

            if val_loader is not None:
                self.which_pass = "validation"

                self.call(f"before_{self.which_pass}_epoch_pass", self)
                self.epoch_pass(
                    loader=self.progress_bar_callback.val_progress_bar,
                    batch_pass=getattr(self.trainer_module, f"{self.which_pass}_batch_pass")
                )
                self.call(f"after_{self.which_pass}_epoch_pass", self)

        self.call("on_fit_end", self)

    def epoch_pass(self, loader: tqdm, batch_pass: Callable):
        for batch_idx, data in enumerate(loader):
            self.call(f"before_{self.which_pass}_batch_pass", self, batch_idx=batch_idx)
            batch_pass(data)
            self.call(f"after_{self.which_pass}_batch_pass", self, batch_idx=batch_idx)

    def call(self, where_at: str, *args, **kwargs):
        for callback in self.callbacks[where_at]:
            callback(*args, **kwargs)

    def _register_callbacks(self, callbacks: list[Callback]):
        for callback in callbacks:
            for k, v in callback.callbacks:
                self.callbacks[k].append(v)

    @property 
    def progress_bar_callback(self) -> ProgressBar:
        for callback in self.callbacks:
            if isinstance(callback, ProgressBar):
                return callback
        return ProgressBar()
