from collections.abc import Iterable
import os
import torch.nn as nn
from typing import Literal
import re


def increment_save_root(save_root: str):
    if save_root.endswith(os.sep):
        save_root = save_root[:-len(os.sep)]

    if os.sep not in save_root:
        save_root = "./" + save_root

    if save_root.startswith("~"):
        save_root = os.path.expanduser(save_root)

    base_dir = save_root.split(os.sep)[-1]
    root_dir = os.path.join(*save_root.split(os.sep)[:-1])
    
    is_dir = os.path.isdir(save_root)
    while is_dir:
        if re.search(r"_\d+$", base_dir):
            num = int(base_dir.split("_")[-1])
            base_dir = "_".join(base_dir.split("_")[0:-1]) + f"_{num + 1}"
            save_root = os.path.join(root_dir, base_dir)
            is_dir = os.path.isdir(save_root)
        else:
            base_dir = base_dir + "_1"
            save_root = os.path.join(root_dir, base_dir)
            is_dir = os.path.isdir(save_root)

    return save_root


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


class Variables:
    def __init__(self):
        self._current_pass = None
        self._current_epoch = -1
        self._current_batch_idx = -1
        self._train_loader = None
        self._num_epochs = None
        self._validation_loader = None
    
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
    def validation_loader(self):
        return self._validation_loader

    @validation_loader.setter
    def validation_loader(self, validation_loader: Iterable):
        if isinstance(validation_loader, Iterable):
            self._validation_loader = validation_loader 
        else:
            raise Exception("ERROR: validation_loader must be an Iterable")

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, num_epochs: int):
        if isinstance(num_epochs, int):
            self._num_epochs = num_epochs 
        else:
            raise Exception("ERROR: num_epochs must be an int")
