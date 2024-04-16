import os
from torch import load, Tensor
from torch.nn import DataParallel

from ..trainer.trainer import Trainer

from .base import Callback

class LoadCheckpoint(Callback):
    def __init__(self, load_from: str):
        self.load_from = load_from

    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "train_module")

        assert hasattr(trainer.train_module, "model")
        assert hasattr(trainer.train_module, "optimizer")
        assert hasattr(trainer.train_module, "state_dict_root")
        assert hasattr(trainer.train_module, "device")

        self.state_dict_root = trainer.train_module.state_dict_root

        self.load_checkpoint(trainer)


    def load_checkpoint(self, trainer: Trainer, *args, **kwargs):
        train_checkpoint = load(self.load_from)
    
        for state in train_checkpoint["OPTIMIZER_STATE"]["state"].values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(trainer.train_module.device)
    
        if isinstance(trainer.train_module.model, DataParallel):
            trainer.train_module.model.module.load_state_dict(
                train_checkpoint["MODEL_STATE"]
            )
        else:
            trainer.train_module.model.load_state_dict(
                train_checkpoint["MODEL_STATE"]
            )
    
        trainer.train_module.model.optimizer.load_state_dict(
            train_checkpoint["OPTIMIZER_STATE"]
        )
        trainer.train_module.model.epochs_run = train_checkpoint["EPOCHS_RUN"]
