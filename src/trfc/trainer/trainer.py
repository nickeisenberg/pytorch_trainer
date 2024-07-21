from collections.abc import Callable, Iterable
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..callbacks.base import Callback
from ..callbacks.progress_bar.base import ProgressBar


class Variables:
    def __init__(self):
        self._current_pass = None
        self._current_epoch = None
        self._current_batch_idx = None
        self._train_loader = None
        self._num_epochs = None
        self._val_loader = None
    
    @property
    def current_pass(self):
        return self._current_pass

    @current_pass.setter
    def current_pass(self, current_pass: str):
        if isinstance(current_pass, str):
            if current_pass in ["train", "val", "test"]:
                self._current_pass = current_pass
        else:
            raise Exception("ERROR: current_pass must be train, val or test.")

    @property
    def current_epoch(self):
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, current_epoch: int):
        if isinstance(current_epoch, int):
            self._current_epoch = current_epoch
        else:
            raise Exception("ERROR: current_epoch must be an int")

    @property
    def current_batch_idx(self):
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
            val_loader: DataLoader | None = None):

        self.variables.train_loader = train_loader,
        self.variables.num_epochs = num_epochs
        if val_loader:
            self.variables.val_loader = val_loader

        self.call("on_fit_start", self)

        for epoch in range(1, num_epochs + 1):
            self.variables.current_epoch = epoch
            self.variables.current_pass = "train"

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
            self.variables.current_batch_idx = batch_idx

            self.call(f"before_{self.current_pass}_batch_pass", self)
            batch_pass(data, batch_idx)
            self.call(f"after_{self.current_pass}_batch_pass", self)

    def call(self, where_at: str, *args, **kwargs):
        for callback in self.callbacks[where_at]:
            if callback:
                callback(*args, **kwargs)

    def _register_callbacks(self, callbacks: list[Callback]):
        for callback in callbacks:
            for k, v in callback.callbacks.items():
                self._callbacks[k].append(v)

                # handle special callbacks
                if isinstance(v, ProgressBar):
                    self.progress_bar_callback = v

    @property 
    def progress_bar_callback(self) -> ProgressBar:
        return self._progress_bar

    @progress_bar_callback.setter 
    def progress_bar_callback(self, pbar: ProgressBar):
        if isinstance(pbar, ProgressBar):
            self._progress_bar = pbar 
        else:
            raise Exception("ERROR: pbar must be a ProgressBar")
