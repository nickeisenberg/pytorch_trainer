from collections.abc import Iterable
from tqdm import tqdm

from .base import ProgressBar as _ProgressBar
from ...trainer import Trainer


def tqdm_postfix_to_dictionary(postfix: str):
    return dict([x.strip().split("=") for x in postfix.split(",")]) if postfix else {}

def append_tqdm_postfix(pbar: tqdm, **kwargs):
    pbar.set_postfix(
        {**tqdm_postfix_to_dictionary(pbar.postfix), **{k: v for k, v in kwargs.items()}}
    )

class ProgressBar(_ProgressBar):
    """The progress bar"""

    def __init__(self):
        super().__init__()

        self._train_loader = None
        self._val_loader = None
        self._postfix = {}
        self._train_progress_bar = None
        self._val_progress_bar = None

    def log(self, name, value):
        """Log values intra batch to be displayed to the tqdm bar"""
        self.postfix[name] = value
    
    @property
    def train_loader(self):
        if self._train_loader is None:
            raise Exception("ERROR: train_loader is called before being set.")
        return self._train_loader
    
    @train_loader.setter
    def train_loader(self, train_loader):
        if isinstance(train_loader, Iterable):
            self._train_loader = train_loader
        else:
            raise Exception("ERROR: train_loader must be a Iterable")

    @property
    def val_loader(self):
        if self._val_loader is None:
            raise Exception("ERROR: train_loader is called before being set.")
        return self._val_loader

    @val_loader.setter
    def val_loader(self, val_loader):
        if isinstance(val_loader, Iterable):
            self._val_loader = val_loader
        else:
            raise Exception("ERROR: val_loader must be a Iterable")
    
    @property
    def postfix(self):
        return self._postfix
    
    @postfix.setter
    def postfix(self, postfix_dict: dict):
        if isinstance(postfix_dict, dict):
            self._postfix = postfix_dict
        else:
            raise Exception("ERROR: postfix_dict must be a dict[str, value]")

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
    def val_progress_bar(self) -> tqdm:
        if self._val_progress_bar is None:
            raise Exception("ERROR: val_progress_bar called before set")
        return self._val_progress_bar

    @val_progress_bar.setter
    def val_progress_bar(self, pbar: tqdm):
        if isinstance(pbar, tqdm):
            self._val_progress_bar = pbar
        else:
            raise Exception("ERROR: pbar must be a tqdm")
    
    def on_fit_start(self, trainer: Trainer):
        self.train_loader = trainer.variables.train_loader
        if trainer.variables.val_loader is not None:
            self.val_loader = trainer.variables.val_loader

    def before_train_epoch_pass(self, trainer: Trainer):
        self.train_progress_bar = tqdm(self.train_loader, leave=True)

    def before_validation_epoch_pass(self, trainer: Trainer):
        self.val_progress_bar = tqdm(self.val_loader, leave=True)

    def after_train_batch_pass(self, trainer: Trainer) -> None:
        append_tqdm_postfix(self.train_progress_bar, **self.postfix)
        self.postfix = {}

    def after_validation_batch_pass(self, trainer: Trainer) -> None:
        if self.val_progress_bar is not None:
            append_tqdm_postfix(self.val_progress_bar, **self.postfix)
            self.postfix = {}
