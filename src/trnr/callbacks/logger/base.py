from abc import ABC
from ..base import Callback

class Logger(Callback, ABC):
    """Logger base class"""
    def __init__(self):
        super().__init__()

        self._train_log = []
        self._validation_log = []
    
    @property
    def train_log(self):
        return self._train_log

    @property
    def validation_log(self):
        return self._validation_log
