import os
import shutil

from .base import Callback
from ..trainer import Trainer

class FileCopier(Callback):
    """
    copy any wanted files to the experiment results dir of the trainer.
    """
    def __init__(self, files_to_copy: list[str]):
        super().__init__()

        self.files_to_copy = files_to_copy

    def on_fit_start(self, trainer: Trainer):
        for f in self.files_to_copy:
            if os.path.isfile(f):
                base_name = os.path.basename(f)
                shutil.copyfile(
                    src=f,
                    dst=os.path.join(trainer.save_root, base_name)
                )

    def on_evaluation_start(self, trainer: Trainer):
        for f in self.files_to_copy:
            if os.path.isfile(f):
                base_name = os.path.basename(f)
                shutil.copyfile(
                    src=f,
                    dst=os.path.join(trainer.save_root, base_name)
                )