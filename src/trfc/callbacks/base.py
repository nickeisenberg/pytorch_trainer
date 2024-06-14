from ..trainer.trainer import Trainer
from abc import ABC, abstractmethod


class Callback(ABC):
    def before_all_epochs(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_train_batch_pass(self, trainer: Trainer, *args, **kwargs):
        pass
    
    def after_train_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def before_validation_batch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def after_validation_epoch_pass(self, trainer: Trainer, *args, **kwargs):
        pass

    def after_all_epochs(self, trainer: Trainer, *args, **kwargs):
        pass


# TODO: Implement a more robust callback system
class Callbacks:
    """Handles all registered callbacks for Hooks."""

    def __init__(self):
        """Initializes a Callbacks object to manage registered event hooks."""
        self._callbacks = {
            "before_all_epochs": [],
            "before_train_epoch_pass": [],
            "before_train_batch_pass": [],
            "after_train_batch_pass": [],
            "after_train_epoch_pass": [],
            "before_validation_epoch_pass": [],
            "before_validation_batch_pass": [],
            "after_validation_batch_pass": [],
            "after_validation_epoch_pass": [],
            "after_all_epochs": [],
        }
        self.stop_training = False  # set True to interrupt training

        self.register_all_actions()

    
    @abstractmethod
    def register_all_actions(self) -> None:
        raise Exception("register_all_actions was not implemented")


    def register_action(self, hook, name="", callback=None):
        """
        Register a new action to a callback hook.

        Args:
            hook: The callback hook name to register the action to
            name: The name of the action for later reference
            callback: The callback to fire
        """

        try:
            assert hook in self._callbacks
        except:
            raise Exception(f"hook '{hook}' not found in callbacks {self._callbacks}")

        try:
            assert callable(callback)
        except:
            raise Exception(f"callback '{callback}' is not callable")

        self._callbacks[hook].append({"name": name, "callback": callback})


    def get_registered_actions(self, hook=None):
        """
        Returns all the registered actions by callback hook.

        Args:
            hook: The name of the hook to check, defaults to all
        """
        return self._callbacks[hook] if hook else self._callbacks


    def run(self, hook, *args, **kwargs):
        """
        Loop through the registered actions and fire all callbacks on main thread.

        Args:
            hook: The name of the hook to check, defaults to all
            args: Arguments to receive
            kwargs: Keyword Arguments to receive 
        """
        
        try:
            assert hook in self._callbacks
        except:
            raise Exception(f"hook '{hook}' not found in callbacks {self._callbacks}")

        for logger in self._callbacks[hook]:
            logger["callback"](*args, **kwargs)
