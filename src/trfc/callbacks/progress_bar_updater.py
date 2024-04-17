from ..trainer.trainer import Trainer

from .base import Callback

class ProgressBarUpdater(Callback):
    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "train_module")
        assert hasattr(trainer.train_module, "logger")
        assert hasattr(trainer.train_module.logger, "_avg_epoch_history")

        
    def before_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "pbar")
        assert hasattr(trainer, "current_epoch")


    def after_train_batch_pass(self, trainer: Trainer, *args, **kwargs):
        trainer.pbar.set_postfix(
            None, 
            True,
            EPOCH=trainer.current_epoch,
            **trainer.train_module.logger._avg_epoch_history
        )


    def before_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "pbar")
        assert hasattr(trainer, "current_epoch")


    def after_validation_batch_pass(self, trainer: Trainer, *args, **kwargs):
        trainer.pbar.set_postfix(
            None, 
            True,
            EPOCH=trainer.current_epoch,
            **trainer.train_module.logger._avg_epoch_history
        )
