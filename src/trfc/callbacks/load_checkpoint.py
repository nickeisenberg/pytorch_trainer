import os
from torch import load, Tensor
from torch.nn import DataParallel

from ..trainer.trainer import Trainer

from .base import Callback

class LoadCheckpoint(Callback):
    def __init__(self, load_from: str):
        self.load_from = load_from

    def on_fit_start(self, trainer: Trainer, *args, **kwargs):
        assert hasattr(trainer, "module")

        assert hasattr(trainer.module, "model")
        assert hasattr(trainer.module, "optimizer")
        assert hasattr(trainer.module, "state_dict_root")
        assert hasattr(trainer.module, "device")

        self.state_dict_root = trainer.module.state_dict_root

        self.load_checkpoint(trainer)


    def load_checkpoint(self, trainer: Trainer, *args, **kwargs):
        train_checkpoint = load(self.load_from)
    
        for state in train_checkpoint["OPTIMIZER_STATE"]["state"].values():
            for k, v in state.items():
                if isinstance(v, Tensor):
                    state[k] = v.to(trainer.module.device)
    
        if isinstance(trainer.module.model, DataParallel):
            trainer.module.model.module.load_state_dict(
                train_checkpoint["MODEL_STATE"]
            )
        else:
            trainer.module.model.load_state_dict(
                train_checkpoint["MODEL_STATE"]
            )
    
        trainer.module.model.optimizer.load_state_dict(
            train_checkpoint["OPTIMIZER_STATE"]
        )
        trainer.module.model.epochs_run = train_checkpoint["EPOCHS_RUN"]
