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
    def validation_progress_bar(self) -> tqdm:
        pass

    @validation_progress_bar.setter
    @abstractmethod
    def validation_progress_bar(self, pbar):
        pass
