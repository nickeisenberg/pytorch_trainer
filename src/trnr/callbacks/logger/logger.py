from collections import defaultdict
import os
import csv

from ...trainer import Trainer
from .base import Logger as _Logger
from ..utils import rank_zero_only

class CSVLogger(_Logger):
    def __init__(self, log_root: str = "logs"):
        super().__init__()
    
        self.epoch_log = defaultdict(list)

        self.log_root = log_root 
        self._train_log_path = None
        self._validation_log_path = None

    @rank_zero_only
    def log(self, name: str, value: float):
        self.epoch_log[name].append(value)
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.log_root = os.path.join(trainer.save_root, self.log_root)
        if not os.path.isdir(self.log_root):
            os.makedirs(self.log_root)

    @rank_zero_only
    def before_train_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
        self.train_log_path = os.path.join(
            self.log_root, f"train_ep{trainer.variables.current_epoch}.csv"
        )

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        pass

    @rank_zero_only
    def before_validation_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
        self.validation_log_path = os.path.join(
            self.log_root, f"validation_ep{trainer.variables.current_epoch}.csv"
        )

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        pass

    def _write_headers(self, path_to_csv: str):
        with open(path_to_csv, 'a', newline='',) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(self.epoch_log.keys()))
            writer.writeheader()

    @property
    def train_log_path(self):
        return self._train_log_path

    @train_log_path.setter
    def train_log_path(self, x: str):
        if isinstance(x, str):
            if not os.path.isfile(x):
                self._train_log_path = x
        else:
            if not isinstance(x, str):
                raise Exception(f"ERROR: {x} is not a string")
            else:
                raise Exception(f"ERROR: {x} already exists.")

    @property
    def validation_log_path(self):
        return self._validation_log_path

    @validation_log_path.setter
    def validation_log_path(self, x: str):
        if isinstance(x, str):
            if not os.path.isfile(x):
                self._validation_log_path = x
        else:
            if not isinstance(x, str):
                raise Exception(f"ERROR: {x} is not a string")
            else:
                raise Exception(f"ERROR: {x} already exists.")

