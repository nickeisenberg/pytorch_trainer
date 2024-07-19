import os
from torch import save
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from ..trainer.trainer import Trainer

from .base import Callback

class SaveBestCheckoint(Callback):
    def __init__(self, 
                 state_dict_root: str, 
                 key: str,
                 best_train_val = 1e6,
                 train_check = lambda cur, prev: cur < prev,
                 best_validation_val = 1e6,
                 validation_check = lambda cur, prev: cur < prev,
                 ):
        self.state_dict_root = state_dict_root 

        self.key = key

        self.best_train_val = best_train_val
        self.train_check = train_check
        self.best_validation_val = best_validation_val
        self.validation_check = validation_check


    def on_fit_start(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "trainer_module")

        assert hasattr(trainer.trainer_module, "logger")

        assert hasattr(trainer.trainer_module.logger, "train_history")
        assert hasattr(trainer.trainer_module.logger, "validation_history")

        assert hasattr(trainer.trainer_module, "model")
        assert hasattr(trainer.trainer_module, "optimizer")

        assert hasattr(trainer, "current_pass")
        assert hasattr(trainer, "current_epoch")


    def after_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        assert self.key in trainer.trainer_module.logger.train_history

        if self.save_checkpoint_flag(trainer, trainer.current_pass):
            self.save_checkpoint(trainer, trainer.current_pass)


    def after_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        assert self.key in trainer.trainer_module.logger.validation_history
        if self.save_checkpoint_flag(trainer, trainer.current_pass):
            self.save_checkpoint(trainer, trainer.current_pass)

    
    def save_checkpoint_flag(self, trainer: Trainer, which):
        save_ckp = False
    
        if which == "train":
            value = trainer.trainer_module.logger.train_history[self.key][-1]
            if self.train_check(value, self.best_train_val):
                save_ckp = True
                self.best_train_val = value
    
        elif which == "validation":
            value = trainer.trainer_module.logger.train_history[self.key][-1]
            if self.validation_check(value, self.best_validation_val):
                save_ckp = True
                self.best_validation_val = value
        
        return save_ckp


    def save_checkpoint(self, trainer: Trainer, *args, **kwargs):
        model = trainer.trainer_module.model
        optimizer = trainer.trainer_module.optimizer
    
        checkpoint = {}
    
        save_to = os.path.join(
            self.state_dict_root, f"{trainer.current_pass}_ckp.pth"
        )
    
        if isinstance(model, DataParallel):
            checkpoint["MODEL_STATE"] = model.module.state_dict()
            checkpoint["OPTIMIZER_STATE"] = optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = trainer.current_epoch 
        elif isinstance(model, DistributedDataParallel):
            checkpoint["MODEL_STATE"] = model.module.state_dict()
            checkpoint["OPTIMIZER_STATE"] = optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = trainer.current_epoch 
        else:
            checkpoint["MODEL_STATE"] = model.state_dict()
            checkpoint["OPTIMIZER_STATE"] = optimizer.state_dict()
            checkpoint["EPOCHS_RUN"] = trainer.current_epoch 
    
        save(checkpoint, save_to)
        print(f"EPOCH {trainer.current_epoch} checkpoint saved at {save_to}")
