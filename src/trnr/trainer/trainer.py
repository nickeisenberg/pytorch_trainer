from typing import Callable, Iterable, Literal
import os

import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import (
    increment_save_root,
    device_and_module_setup, 
    Variables
)
from ..callbacks.base import Callback
from ..callbacks.progress_bar.base import ProgressBar
from ..callbacks.logger.base import Logger 


class Trainer:
    def __init__(self, 
                 module: nn.Module,
                 device: Literal["cpu", "gpu", "mps"] = "cpu",
                 ddp: bool = False,
                 callbacks: list[Callback] | None = None,
                 save_root: str = "./trainer_data"):
        
        self.ddp = ddp
        self.module, self.device = device_and_module_setup(module, device, ddp)
        self.save_root = increment_save_root(save_root)

        self.rank = int(os.getenv("RANK", 0))
        self.local_rank = int(os.getenv("LOCAL_RANK", 0))

        self.variables = Variables()
        
        self._callbacks: dict[str, list[Callable]] = {
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

        # register all callbacks
        if callbacks:
            self._register_callbacks(callbacks)

        self._logger = None
        self._progress_bar = None

    @property
    def callbacks(self):
        return self._callbacks

    def fit(self, 
            train_loader: DataLoader,
            num_epochs: int,
            data_devicer: Callable | None = None,
            validation_loader: DataLoader | None = None):

        self.variables.train_loader = train_loader
        self.variables.num_epochs = num_epochs
        if validation_loader:
            self.variables.validation_loader = validation_loader
        
        self.call("on_fit_start", self)

        for epoch in range(1, num_epochs + 1):
            self.variables.current_epoch = epoch
            self.variables.current_pass = "train"

            train_batch_pass = self._get_batch_pass()

            self.call("before_train_epoch_pass", self)
            self.epoch_pass(
                data_iterator=self.progress_bar_callback.train_progress_bar,
                data_devicer=data_devicer,
                batch_pass=train_batch_pass
            )
            self.call("after_train_epoch_pass", self)

            if validation_loader is not None:
                self.variables.current_pass = "validation"

                validation_batch_pass = self._get_batch_pass()

                self.call("before_validation_epoch_pass", self)
                self.epoch_pass(
                    data_iterator=self.progress_bar_callback.validation_progress_bar,
                    data_devicer=data_devicer,
                    batch_pass=validation_batch_pass
                )
                self.call(f"after_validation_epoch_pass", self)

        self.call("on_fit_end", self)

    def epoch_pass(self, data_iterator: Iterable, 
                   data_devicer: Callable | None, 
                   batch_pass: Callable):

        for batch_idx, data in enumerate(data_iterator):
            self.variables.current_batch_idx = batch_idx
            if data_devicer:
                data = data_devicer(data, self.device)
            else:
                data = (data[0].to(self.device), data[1].to(self.device))

            self.call(f"before_{self.variables.current_pass}_batch_pass", self)
            batch_pass(data, batch_idx)
            self.call(f"after_{self.variables.current_pass}_batch_pass", self)

    def predict(self, *_):
        raise Exception("ERROR: predict is not implemented yet")

    def call(self, where_at: str, *args, **kwargs):
        for callback in self.callbacks[where_at]:
            if callback:
                callback(*args, **kwargs)

    def _register_callbacks(self, callbacks: list[Callback]):
        for callback in callbacks:
            for k, v in callback.callbacks.items():
                self._callbacks[k].append(v)

            # handle special callbacks
            if isinstance(callback, ProgressBar):
                self.progress_bar_callback = callback
            elif isinstance(callback, Logger):
                self.logger_callback = callback

    def _get_batch_pass(self):
        if not self.ddp:
            batch_pass = getattr(
                self.module, f"{self.variables.current_pass}_batch_pass"
            )
        else:
            batch_pass = getattr(
                self.module.module, f"{self.variables.current_pass}_batch_pass"
            )
        return batch_pass

    @property 
    def progress_bar_callback(self) -> ProgressBar:
        if self._progress_bar is not None:
            return self._progress_bar
        else:
            raise Exception("ERROR: pbar called before being set.")

    @progress_bar_callback.setter 
    def progress_bar_callback(self, pbar: ProgressBar):
        if isinstance(pbar, ProgressBar):
            self._progress_bar = pbar 
        else:
            raise Exception("ERROR: pbar must be a ProgressBar")

    @property 
    def logger_callback(self) -> Logger:
        if self._logger is not None:
            return self._logger
        else:
            raise Exception("ERROR: pbar called before being set.")

    @logger_callback.setter 
    def logger_callback(self, logger: Logger):
        if isinstance(logger, Logger):
            self._logger = logger 
        else:
            raise Exception("ERROR: logger must be a Logger")
