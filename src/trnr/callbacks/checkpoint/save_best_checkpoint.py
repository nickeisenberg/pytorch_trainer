from typing import Literal

from ...trainer import Trainer
from ..base import Callback

class SaveBestCheckpoint(Callback):
    """A checkpoint saveer"""
    def __init__(self,
                 train_metric: str,
                 train_desire: Literal["increase", "decrease"],
                 validation_metric,
                 validation_desire: Literal["increase", "decrease"]):
        super().__init__()

        self.train_metric = train_metric
        self.validation_metric = validation_metric
        
        if train_desire == "increase":
            self.best_train_metric = -1e6
        else:
            self.best_train_metric = 1e6

        if validation_desire == "increase":
            self.best_validation_metric = -1e6
        else:
            self.best_validation_metric = 1e6

    def after_train_epoch_pass(self, trainer: Trainer):
        pass

    def after_validation_epoch_pass(self, trainer: Trainer):
        pass
