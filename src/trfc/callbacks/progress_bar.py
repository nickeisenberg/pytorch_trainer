from collections.abc import Iterable
from tqdm import tqdm
from .base import Callback
from ..trainer import Trainer


def tqdm_postfix_to_dictionary(postfix: str):
    return dict([x.strip().split("=") for x in postfix.split(",")]) if postfix else {}


def append_tqdm_postfix(pbar: tqdm, **kwargs):
    pbar.set_postfix(
        {**tqdm_postfix_to_dictionary(pbar.postfix), **{k: v for k, v in kwargs.items()}}
    )


class ProgressBar(Callback):
    """The progress bar"""

    def __init__(self):
        super().__init__()

        self.postfix = {}
        self._train_progress_bar = None
        self._val_progress_bar = None

    def log(self, name, value):
        self.postfix[name] = value

    @property
    def train_progress_bar(self) -> tqdm:
        if self._train_progress_bar is None:
            raise Exception("Progress bar is only set in fit")
        else:
            return self._train_progress_bar

    @property
    def val_progress_bar(self) -> tqdm:
        if self._val_progress_bar is None:
            raise Exception("Progress bar is only set in fit")
        else:
            return self._val_progress_bar
    
    def on_fit_start(self, trainer: Trainer, train_loader: Iterable, 
                    val_loader: Iterable | None = None, *args, **kwargs):
        self._train_progress_bar = tqdm(train_loader, leave=True)
        if val_loader is not None:
            self._val_progress_bar = tqdm(val_loader, leave=True)

    def on_train_batch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        append_tqdm_postfix(self.train_progress_bar, **self.postfix)
        self.postfix = {}

    def on_validation_batch_end(self, trainer: Trainer, *args, **kwargs) -> None:
        if self.val_progress_bar is not None:
            append_tqdm_postfix(self.val_progress_bar, **self.postfix)
            self.postfix = {}
