import os
from matplotlib.figure import Figure
import numpy as np
from numpy import ndarray
from matplotlib import colormaps
import matplotlib.pyplot as plt
from torch import Tensor

from ..trainer.trainer import Trainer
from .base import Callback
from .utils import rank_zero_only


class ConfusionMatrix(Callback):
    def __init__(self, labels: list, save_root: str = "metrics"):
        super().__init__()
        self.labels = labels

        self.save_root = save_root
        
        self.predictions = []
        self.targets = []
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.save_root = os.path.join(trainer.save_root, self.save_root)
        if not os.path.isdir(self.save_root):
            os.makedirs(self.save_root)

    @rank_zero_only
    def log(self, predictions: Tensor, targets: Tensor):
        self.predictions += predictions.tolist()
        self.targets += targets.tolist()

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        self.reset_state(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        self.reset_state(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )

    def reset_state(self, which, epoch):
        matrix = self.compute_confusion_matrix(self.targets, self.predictions)
        fig = self.make_confusion_matrix_fig(matrix, self.labels)

        save_to = os.path.join(
            self.save_root, f"{which}_ep{epoch}.png"
        )
        fig.savefig(save_to)

        self.predictions = []
        self.targets = []
        return None

    
    def compute_confusion_matrix(self, targets: list[int], predictions: list[int]):
        """
        Compute a confusion matrix for multi-class classification.
    
        Parameters:
        - targets: A list of actual target values.
        - predictions: A list of predictions.
        - num_classes: The number of classes in the classification problem.
    
        Returns:
        - A 2D numpy array representing the confusion matrix.
        """
        num_classes = len(self.labels)
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        for actual, predicted in zip(targets, predictions):
            matrix[actual, predicted] += 1
        return matrix

    @staticmethod 
    def make_confusion_matrix_fig(matrix:ndarray, class_names: list) -> Figure:
        """
        Plot the confusion matrix using matplotlib.
    
        Parameters:
        - matrix: A 2D numpy array representing the confusion matrix.
        - class_names: A list of names for the classes.
        """
        try:
            class_names = list(np.array(class_names).astype(str))
        except:
            raise Exception("class names could not be converted to string")
    
        accuracy = np.round(np.einsum("ii->i", matrix).sum() / matrix.sum() * 100, 2)
    
        fig, ax = plt.subplots(figsize=(10, 7))
        percentages = matrix / matrix.sum(axis=1)
        cax = ax.matshow(percentages, cmap=colormaps["Blues"])
        plt.title(f'Confusion Matrix: Accuracy {accuracy}%', pad=20)
        fig.colorbar(cax)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45)
        ax.set_yticklabels(class_names)
    
        # Loop over data dimensions and create text annotations.
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j, i, 
                    f"{np.round(percentages[i, j] * 100 ,2 )}%", 
                    ha="center", va="center", color="Black",
                    size=12
                )
    
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
    
        return fig
