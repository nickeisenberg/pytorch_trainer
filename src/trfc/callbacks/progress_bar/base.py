from abc import ABC, abstractmethod

from tqdm import tqdm
from ..base import Callback

class ProgressBar(Callback, ABC):
    """ A name so that the Trainer knows what this callback is """
    
    @property
    @abstractmethod
    def train_progress_bar(self) -> tqdm:
        pass

    @train_progress_bar.setter
    @abstractmethod
    def train_progress_bar(self, pbar):
        pass

    @property
    @abstractmethod
    def val_progress_bar(self) -> tqdm:
        pass

    @val_progress_bar.setter
    @abstractmethod
    def val_progress_bar(self, pbar):
        pass
