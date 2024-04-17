from torch import Tensor, tensor, hstack
from numpy import round

from .base import Callback


class Accuracy(Callback):
    def __init__(self):
        self.accuracy = 0.0
        self._running_total = tensor([])


    def log(self, predictions: Tensor, targets: Tensor):
        predictions, targets = predictions.cpu(), targets.cpu()
        acc = ((predictions == targets) * 1).float()
        self._running_total = hstack([self._running_total, acc])
        self.accuracy = round(
             self._running_total.mean().item() * 100, 2
         )


    def after_train_epoch_pass(self, *args, **kwargs):
        self.__init__()


    def after_validation_epoch_pass(self, *args, **kwargs):
        self.__init__()
