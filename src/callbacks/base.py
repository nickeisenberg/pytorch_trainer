from ..trainer.trainer import Trainer
from abc import ABC


class Callback(ABC):
    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_train_batch_pass(self, trainer: Trainer, *args, **kwargs):
        pass
    
    def after_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_validation_batch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def after_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def after_all_epochs(self, trainer: Trainer, *args, **kwargs):
        pass
