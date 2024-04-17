import os
import pandas as pd
from collections import defaultdict

from ..trainer.trainer import Trainer

from .base import Callback

class CSVLogger(Callback):
    def __init__(self, save_root):
        self.save_root = save_root
        self.train_history = defaultdict(list)
        self.validation_history = defaultdict(list)

        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)


    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "train_module")
        assert hasattr(trainer.train_module, "logger")


    def after_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        self._after_epoch(trainer.which_pass)


    def after_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        self._after_epoch(trainer.which_pass)


    def log(self, loss_dict: dict) -> None:
        """
        Log loss after each batch. Update the CSV or whatever file.
        """

        for key in loss_dict:
            self._epoch_history[key].append(loss_dict[key])

            self._avg_epoch_history[key] = sum(
                self._epoch_history[key]
            ) / len(self._epoch_history[key])
        

    def _after_epoch(self, which: str) -> None:
        file_name = f"{which}_log.csv"
        loss_log_file_path  = os.path.join(
            self.save_root, file_name
        )

        df = pd.DataFrame(self._epoch_history)

        if not os.path.isfile(loss_log_file_path):
            df.to_csv(loss_log_file_path, index=False)

        else:
            df.to_csv(loss_log_file_path, mode='a', header=False, index=False)

        for k in self._avg_epoch_history:
            if which == "train":
                self.train_history[k].append(self._avg_epoch_history[k])
            elif which == "validation":
                self.validation_history[k].append(self._avg_epoch_history[k])
        
        self._epoch_history = defaultdict(list)
        self._avg_epoch_history = defaultdict(float)
