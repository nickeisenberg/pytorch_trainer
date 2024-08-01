from collections import defaultdict
from abc import ABC
from ..base import Callback

class Logger(Callback, ABC):
    """Logger base class"""
