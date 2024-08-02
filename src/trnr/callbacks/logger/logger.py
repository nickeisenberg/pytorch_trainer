from collections import defaultdict
import os
import csv

from .base import Logger as _Logger
from ..utils import rank_zero_only
from ...trainer import Trainer


class CSVLogger(_Logger):
    def __init__(self, log_root: str = "logs"):
        super().__init__()

        self.log_root = log_root 

        self.batch_log = defaultdict(float)

    @rank_zero_only
    def log(self, name: str, value: float):
        self.batch_log[name] = value
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.log_root = os.path.join(trainer.save_root, self.log_root)
        if not os.path.isdir(self.log_root):
            os.makedirs(self.log_root)

        self.train_log_path = os.path.join(
            self.log_root, f"train.csv"
        )
        self.validation_log_path = os.path.join(
            self.log_root, f"validation.csv"
        )
        self.train_headers_written = False
        self.validation_headers_written = False

    @rank_zero_only
    def before_train_epoch_pass(self, trainer: Trainer):
        pass

    @rank_zero_only
    def after_train_batch_pass(self, trainer: Trainer):
        with open(self.train_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_log.keys())
            if not self.train_headers_written:
                _ = writer.writeheader()
                self.train_headers_written = True
            _ = writer.writerow(self.batch_log)
        self.batch_log = defaultdict(float)

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        pass

    @rank_zero_only
    def before_validation_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
    
    @rank_zero_only
    def after_validation_batch_pass(self, trainer: Trainer):
        with open(self.validation_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_log.keys())
            if not self.validation_headers_written:
                _ = writer.writeheader()
                self.validation_headers_written = True
            _ = writer.writerow(self.batch_log)
        self.batch_log = defaultdict(float)

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        pass
