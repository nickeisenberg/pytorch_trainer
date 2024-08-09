from torch.optim.lr_scheduler import LRScheduler as _LRScheduler
from ..base import Callback

class LRScheduler(Callback):
    def __init__(self, scheduler: _LRScheduler):
        super().__init__()
        self.scheduler = scheduler

    def after_train_epoch_pass(self, *_):
        self.scheduler.step()
