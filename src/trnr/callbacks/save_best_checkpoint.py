import os
import torch

from typing import Literal

from ..trainer import Trainer
from .base import Callback
from .utils import rank_zero_only

class SaveBestCheckpoint(Callback):
    """A checkpoint saveer"""
    def __init__(self,
                 train_metric: str,
                 train_metric_desire: Literal["increase", "decrease"],
                 validation_metric: str,
                 validation_metric_desire: Literal["increase", "decrease"],
                 state_dict_dir_name: str = "state_dicts"):
        super().__init__()
        
        self.state_dict_root = state_dict_dir_name 
        self.train_metric = train_metric
        self.train_metric_desire = train_metric_desire
        self.validation_metric = validation_metric
        self.validation_metric_desire = validation_metric_desire
        
        if self.train_metric_desire == "increase":
            self.best_train_metric = -1e6
        else:
            self.best_train_metric = 1e6

        if self.validation_metric_desire == "increase":
            self.best_validation_metric = -1e6
        else:
            self.best_validation_metric = 1e6
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.state_dict_root = os.path.join(trainer.save_root, self.state_dict_root)
        if not os.path.isdir(self.state_dict_root):
            os.makedirs(self.state_dict_root)

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        last_epoch_val = trainer.logger_callback.train_log[self.train_metric][-1]

        if self.train_metric_desire == "decrease" and last_epoch_val <= self.best_train_metric:
            self.best_train_metric = last_epoch_val
            torch.save(
                self._get_module_state_dict(trainer),
                os.path.join(
                    self.state_dict_root, 
                    f"train_ep_{trainer.variables.current_epoch}.pth"
                )
            )

        elif self.train_metric_desire == "increase" and last_epoch_val >= self.best_train_metric:
            self.best_train_metric = last_epoch_val
            torch.save(
                self._get_module_state_dict(trainer),
                os.path.join(
                    self.state_dict_root, 
                    f"train_ep_{trainer.variables.current_epoch}.pth"
                )
            )

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        last_epoch_val = trainer.logger_callback.validation_log[self.validation_metric][-1]

        if self.validation_metric_desire == "decrease" and last_epoch_val <= self.best_validation_metric:
            self.best_validation_metric = last_epoch_val
            torch.save(
                self._get_module_state_dict(trainer),
                os.path.join(
                    self.state_dict_root, 
                    f"validation_ep_{trainer.variables.current_epoch}.pth"
                )
            )

        elif self.validation_metric_desire == "increase" and last_epoch_val >= self.best_validation_metric:
            self.best_validation_metric = last_epoch_val
            torch.save(
                self._get_module_state_dict(trainer),
                os.path.join(
                    self.state_dict_root, 
                    f"validation_ep_{trainer.variables.current_epoch}.pth"
                )
            )
    
    @staticmethod
    def _get_module_state_dict(trainer: Trainer):
        if not trainer.ddp:
            return trainer.module.state_dict()
        else:
            return trainer.module.module.state_dict()
