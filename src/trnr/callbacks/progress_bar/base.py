from abc import ABC
from tqdm import tqdm
from ..base import Callback

class ProgressBar(Callback, ABC):
    """ A name so that the Trainer knows what this callback is """

    def __init__(self):
        super().__init__()
        self._train_progress_bar = None
        self._validation_progress_bar = None
    
    @property
    def train_progress_bar(self) -> tqdm:
        if self._train_progress_bar is None:
            raise Exception("ERROR: train_progress_bar called before being set.")
        return self._train_progress_bar

    @train_progress_bar.setter
    def train_progress_bar(self, pbar: tqdm):
        if isinstance(pbar, tqdm):
            self._train_progress_bar = pbar
        else:
            raise Exception("ERROR: pbar must be a tqdm")

    @property
    def validation_progress_bar(self) -> tqdm:
        if self._validation_progress_bar is None:
            raise Exception("ERROR: validation_progress_bar called before set")
        return self._validation_progress_bar

    @validation_progress_bar.setter
    def validation_progress_bar(self, pbar: tqdm):
        if isinstance(pbar, tqdm):
            self._validation_progress_bar = pbar
        else:
            raise Exception("ERROR: pbar must be a tqdm")
