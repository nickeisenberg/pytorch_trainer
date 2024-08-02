from abc import ABC
from collections import defaultdict
from ..base import Callback


class Logger(Callback, ABC):
    """Logger base class"""

    def __init__(self):
        super().__init__()
        self._train_log = defaultdict(list)
        self._validation_log = defaultdict(list)

    @property
    def train_log(self):
        return self._train_log

    @property
    def validation_log(self):
        return self._validation_log
