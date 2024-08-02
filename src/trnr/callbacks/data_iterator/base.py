from abc import ABC
from typing import Iterable

from ..base import Callback


class DataIterator(Callback, ABC):
    """A name so that the Trainer knows what this callback is"""

    def __init__(self):
        super().__init__()
        self._train_data_iterator = None
        self._validation_data_iterator = None
    
    @property
    def train_data_iterator(self) -> Iterable:
        if self._train_data_iterator is None:
            raise Exception("ERROR: train_data_iterator called before being set.")
        return self._train_data_iterator

    @train_data_iterator.setter
    def train_data_iterator(self, train_data_iterator: Iterable):
        if isinstance(train_data_iterator, Iterable):
            self._train_data_iterator = train_data_iterator 
        else:
            raise Exception("ERROR: train_data_iterator must be a Iterable")

    @property
    def validation_data_iterator(self) -> Iterable:
        if self._validation_data_iterator is None:
            raise Exception("ERROR: validation_data_iterator called before set")
        return self._validation_data_iterator

    @validation_data_iterator.setter
    def validation_data_iterator(self, validation_data_iterator: Iterable):
        if isinstance(validation_data_iterator, Iterable):
            self._validation_data_iterator = validation_data_iterator 
        else:
            raise Exception("ERROR: validation_data_iterator must be a Iterable")
