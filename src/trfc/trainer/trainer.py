from collections.abc import Callable
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks.base import Callback
from ..callbacks.progress_bar import ProgressBar



class Trainer:
    def __init__(self, 
                 trainer_module: nn.Module,
                 callbacks: list[Callback] | None = None):

        self.trainer_module = trainer_module 
        
        self._callbacks = {
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

        self._env = {}

        self._progress_bar = ProgressBar()
        # register all callbacks
        if callbacks:
            self._register_callbacks(callbacks)
        else:
            self._register_callbacks([self.progress_bar_callback])
    
    @property
    def env(self):
        """Enviornment variables to pass around"""
        return self._env

    def put_to_env(self, **kwargs):
        for k, v in kwargs.items():
            self.env[k] = v

    @property
    def callbacks(self):
        return self._callbacks

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            val_loader: DataLoader | None = None):
        
        self.put_to_env(
            current_pass="N/A", 
            current_epoch="N/A", 
            current_batch_idx="N/A" ,
            train_loader=train_loader,
            num_epochs=num_epochs,
            val_loader=val_loader
        )

        self.call("on_fit_start", self)

        for epoch in range(1, num_epochs + 1):
            self.put_to_env(current_epoch=epoch, current_pass="train")

            self.call("before_train_epoch_pass", self)
            self.epoch_pass(
                loader=self.progress_bar_callback.train_progress_bar,
                batch_pass=getattr(self.trainer_module, "train_batch_pass")
            )
            self.call("after_train_epoch_pass", self)

            if val_loader is not None:
                self.current_pass = "validation"

                self.call("before_validation_epoch_pass", self)
                self.epoch_pass(
                    loader=self.progress_bar_callback.val_progress_bar,
                    batch_pass=getattr(self.trainer_module, "validation_batch_pass")
                )
                self.call(f"after_validatin_epoch_pass", self)

        self.call("on_fit_end", self)

    def epoch_pass(self, loader: tqdm, batch_pass: Callable):
        for batch_idx, data in enumerate(loader):
            self.call(f"before_{self.current_pass}_batch_pass", self, batch_idx=batch_idx)
            batch_pass(data, batch_idx)
            self.call(f"after_{self.current_pass}_batch_pass", self, batch_idx=batch_idx)

    def call(self, where_at: str, *args, **kwargs):
        for callback in self.callbacks[where_at]:
            if callback:
                callback(*args, **kwargs)

    def _register_callbacks(self, callbacks: list[Callback]):
        for callback in callbacks:
            for k, v in callback.callbacks.items():
                self._callbacks[k].append(v)

    @property 
    def progress_bar_callback(self) -> ProgressBar:
        for callback in self._callbacks:
            if isinstance(callback, ProgressBar):
                self._progress_bar = callback
                return callback
        return self._progress_bar
