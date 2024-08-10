from torch.optim.lr_scheduler import LRScheduler
from ..base import Callback

class BasicLRScheduler(Callback):
    def __init__(self, scheduler: LRScheduler):
        super().__init__()
        self.scheduler = scheduler

    def after_train_epoch_pass(self, *_):
        self.scheduler.step()
