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
        self.epoch_log = defaultdict(list)

        self.train_log = defaultdict(list)
        self.validation_log = defaultdict(list)

    @rank_zero_only
    def log(self, name: str, value: float):
        self.batch_log[name] = value
        self.epoch_log[name].append(value)
    
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

    @rank_zero_only
    def before_train_epoch_pass(self, trainer: Trainer):
        self.headers_written = False

    @rank_zero_only
    def after_train_batch_pass(self, trainer: Trainer):
        with open(self.train_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_log.keys())
            if not self.headers_written:
                _ = writer.writeheader()
                self.headers_written = True
            _ = writer.writerow(self.batch_log)
        self.batch_log = defaultdict(float)

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_log:
            self.train_log[key].append(
                round(sum(self.epoch_log[key]) / len(self.epoch_log[key]), 4)
            )
        self.epoch_log = defaultdict(list)
        pass

    @rank_zero_only
    def before_validation_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
    
    @rank_zero_only
    def after_validation_batch_pass(self, trainer: Trainer):
        with open(self.validation_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_log.keys())
            if not self.headers_written:
                _ = writer.writeheader()
                self.headers_written = True
            _ = writer.writerow(self.batch_log)
        self.batch_log = defaultdict(float)

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_log:
            self.validation_log[key].append(
                round(sum(self.epoch_log[key]) / len(self.epoch_log[key]), 4)
            )
        self.epoch_log = defaultdict(list)

    def _write_headers(self, path_to_csv: str):
        with open(path_to_csv, 'a', newline='',) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(self.epoch_log.keys()))
            writer.writeheader()
