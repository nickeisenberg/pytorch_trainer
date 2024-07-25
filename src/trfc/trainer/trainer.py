import os
from typing import Callable, Iterable, Literal

import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from ..callbacks.base import Callback
from ..callbacks.progress_bar.base import ProgressBar


class Variables:
    def __init__(self):
        self._current_pass = None
        self._current_epoch = -1
        self._current_batch_idx = -1
        self._train_loader = None
        self._num_epochs = None
        self._val_loader = None
    
    @property
    def current_pass(self):
        return self._current_pass

    @current_pass.setter
    def current_pass(self, current_pass: str):
        if isinstance(current_pass, str):
            if current_pass in ["train", "validation", "test"]:
                self._current_pass = current_pass
        else:
            raise Exception("ERROR: current_pass must be train, val or test.")

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch: int):
        if isinstance(current_epoch, int):
            self._current_epoch = current_epoch
        else:
            raise Exception("ERROR: current_epoch must be an int")

    @property
    def current_batch_idx(self) -> int:
        return self._current_batch_idx

    @current_batch_idx.setter
    def current_batch_idx(self, current_batch_idx: int):
        if isinstance(current_batch_idx, int):
            self._current_batch_idx = current_batch_idx
        else:
            raise Exception("ERROR: current_batch_idx must be an int")

    @property
    def train_loader(self):
        return self._train_loader

    @train_loader.setter
    def train_loader(self, train_loader: Iterable):
        if isinstance(train_loader, Iterable):
            self._train_loader = train_loader 
        else:
            raise Exception("ERROR: train_loader must be an Iterable")

    @property
    def val_loader(self):
        return self._val_loader

    @val_loader.setter
    def val_loader(self, val_loader: Iterable):
        if isinstance(val_loader, Iterable):
            self._val_loader = val_loader 
        else:
            raise Exception("ERROR: val_loader must be an Iterable")

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int):
        if isinstance(num_epochs, int):
            self._num_epochs = num_epochs 
        else:
            raise Exception("ERROR: num_epochs must be an int")

def device_and_module_setup(module: nn.Module, 
                            device: Literal["cpu", "gpu", "mps"], 
                            ddp: bool):
    if device == "cpu":
        if ddp:
            raise Exception("ERROR: Distribution across CPUs is not supported")
        else:
            return module, device

    elif device == "mps":
        if ddp:
            raise Exception("ERROR: Distribution across mps is not supported")
        else:
            return module.to("mps"), device 

    elif device == "gpu":
        if not ddp:
            return module.to(0), 0 
        else:
            local_rank = int(os.environ["LOCAL_RANK"])
            return nn.parallel.DistributedDataParallel(
                module.to(local_rank), device_ids=[local_rank]
            ), local_rank

    else:
        raise Exception("ERROR: Device not supported.")

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
            val_loader: DataLoader | None = None):

        self.variables.train_loader = train_loader
        self.variables.num_epochs = num_epochs
        if val_loader:
            self.variables.val_loader = val_loader
        
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

            if val_loader is not None:
                self.variables.current_pass = "validation"

                if not self.ddp:
                    validation_batch_pass = getattr(self.module, "validation_batch_pass")
                else:
                    validation_batch_pass = getattr(self.module.module, "validation_batch_pass")

                self.call("before_validation_epoch_pass", self)
                self.epoch_pass(
                    loader=self.progress_bar_callback.val_progress_bar,
                    data_devicer=data_devicer,
                    batch_pass=validation_batch_pass
                )
                self.call(f"after_validation_epoch_pass", self)

        self.call("on_fit_end", self)

    def epoch_pass(self, loader: tqdm, data_devicer: Callable | None, batch_pass: Callable):
        for batch_idx, data in enumerate(loader):
            if data_devicer:
                data = data_devicer(data, self.device)

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
    
