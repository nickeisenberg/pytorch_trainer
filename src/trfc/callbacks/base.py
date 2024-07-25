from abc import abstractmethod
from collections.abc import Callable


def _callback_not_implemented(*args, **kwargs):
    return None

_callbacks = {
    "on_fit_start": _callback_not_implemented,
    "before_train_epoch_pass": _callback_not_implemented,
    "before_train_batch_pass": _callback_not_implemented,
    "after_train_batch_pass": _callback_not_implemented,
    "after_train_epoch_pass": _callback_not_implemented,
    "before_validation_epoch_pass": _callback_not_implemented,
    "before_validation_batch_pass": _callback_not_implemented,
    "after_validation_batch_pass": _callback_not_implemented,
    "after_validation_epoch_pass": _callback_not_implemented,
    "on_fit_end": _callback_not_implemented,
}


class Callback:
    """Handles all registered callbacks for Hooks."""

    def __init__(self):
        """Initializes a Callbacks object to manage registered event hooks."""

        self._callbacks = _callbacks
        self.register_all_actions()

    @property
    def callbacks(self) -> dict[str, Callable]:
        return self._callbacks
    
    def register_all_actions(self) -> None:
        atlease_one_callback_set = False
        for callback in self.callbacks.keys():
            if hasattr(self, callback):
                self.callbacks[callback] = getattr(self, callback)
                atlease_one_callback_set = True
        if not atlease_one_callback_set:
            error_message = """ No callback was set. Ensure to set one of the 
            following:
                - on_fit_start
                - before_train_epoch_pass
                - before_train_batch_pass
                - after_train_batch_pass
                - after_train_epoch_pass
                - before_validation_epoch_pass
                - before_validation_batch_pass
                - after_validation_batch_pass
                - after_validation_epoch_pass
                - on_fit_end
            """
            raise Exception(error_message)
