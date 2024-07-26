from ...trainer import Trainer
from .base import Logger as _Logger

class Logger(_Logger):
    def __init__(self):
        super().__init__()

    def on_fit_start(self, trainer: Trainer):
        pass

