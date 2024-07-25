from typing import Callable, Literal

import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from .utils import device_and_module_setup, Variables
from ..callbacks.base import Callback
from ..callbacks.progress_bar.base import ProgressBar


class Trainer:
    def __init__(self, 
                 module: nn.Module,
                 device: Literal["cpu", "gpu", "mps"] = "cpu",
                 ddp: bool = False,
                 callbacks: list[Callback] | None = None):
        
        self.ddp = ddp
        self.module, self.device = device_and_module_setup(module, device, ddp)
        
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

        self.variables = Variables()

        # register all callbacks
        if callbacks:
            self._register_callbacks(callbacks)

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
            
            if not self.ddp:
                train_batch_pass = getattr(self.module, "train_batch_pass")
            else:
                train_batch_pass = getattr(self.module.module, "train_batch_pass")

            self.call("before_train_epoch_pass", self)
            self.epoch_pass(
                loader=self.progress_bar_callback.train_progress_bar,
                data_devicer=data_devicer,
                batch_pass=train_batch_pass
            )
            self.call("after_train_epoch_pass", self)

            if validation_loader is not None:
                self.variables.current_pass = "validation"

                if not self.ddp:
                    validation_batch_pass = getattr(self.module, "validation_batch_pass")
                else:
                    validation_batch_pass = getattr(self.module.module, "validation_batch_pass")

                self.call("before_validation_epoch_pass", self)
                self.epoch_pass(
                    loader=self.progress_bar_callback.validation_progress_bar,
                    data_devicer=data_devicer,
                    batch_pass=validation_batch_pass
                )
                self.call(f"after_validation_epoch_pass", self)

        self.call("on_fit_end", self)

    def epoch_pass(self, loader: tqdm, data_devicer: Callable | None, batch_pass: Callable):
        for batch_idx, data in enumerate(loader):
            if data_devicer:
                data = data_devicer(data, self.device)
            else:
                data = (data[0].to(self.device), data[1].to(self.device))

            self.variables.current_batch_idx = batch_idx

            self.call(f"before_{self.variables.current_pass}_batch_pass", self)
            batch_pass(data, batch_idx)
            self.call(f"after_{self.variables.current_pass}_batch_pass", self)

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

    @property 
    def progress_bar_callback(self) -> ProgressBar:
        return self._progress_bar

    @progress_bar_callback.setter 
    def progress_bar_callback(self, pbar: ProgressBar):
        if isinstance(pbar, ProgressBar):
            self._progress_bar = pbar 
        else:
            raise Exception("ERROR: pbar must be a ProgressBar")
