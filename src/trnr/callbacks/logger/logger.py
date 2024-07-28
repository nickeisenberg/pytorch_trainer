from collections import defaultdict
import os
import csv

from ...trainer import Trainer
from .base import Logger as _Logger
from ..utils import rank_zero_only

class CSVLogger(_Logger):
    def __init__(self, log_dir: str = "logs"):
        super().__init__()
    
        self.log_dir = log_dir
        self.epoch_log = defaultdict(list)

    @rank_zero_only
    def log(self, name: str, value: float):
        self.epoch_log[name].append(value)
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.log_dir = os.path.join(trainer.save_root, self.log_dir)
        if not os.path.isdir(self.log_dir):
            os.makedirs(self.log_dir)

    @rank_zero_only
    def before_train_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
        pass

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        pass

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        pass

    def _write_headers(self, which: str):
        with open(f'{which}.csv', 'a', newline='',) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(self.epoch_log.keys()))
            writer.writeheader()
        self._headers_written = True
