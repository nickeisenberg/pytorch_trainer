from ..base import Callback

class SaveBestCheckpoint(Callback):
    """A checkpoint saveer"""
    def __init__(self):
        super().__init__()
