from abc import ABC
from collections import defaultdict
from ..base import Callback


class Logger(Callback, ABC):
    """Logger base class"""

    def __init__(self, priority=0):
        super().__init__(priority)
        self._train_log = defaultdict(list[float])
        self._validation_log = defaultdict(list[float])
        self._evaluation_log = defaultdict(list[float])

    @property
    def train_log(self):
        return self._train_log

    @property
    def validation_log(self):
        return self._validation_log

    @property
    def evaluation_log(self):
        return self._evaluation_log
