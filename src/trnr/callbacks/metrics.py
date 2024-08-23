import os
import pandas as pd

import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from torch import Tensor

from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
)

from ..trainer.trainer import Trainer
from .base import Callback
from .utils import rank_zero_only

class ClassificationSummary(Callback):
    def __init__(self, labels: list, save_root: str = "metrics",
                 normalize_conf_mat: bool = False):
        super().__init__()
        self.labels = labels
        self.normalize_conf_mat = normalize_conf_mat

        self.conf_mat_save_root = os.path.join(save_root, "conf_mat")
        self.report_save_root = os.path.join(save_root, "report")
        
        self.predictions = []
        self.targets = []
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.conf_mat_save_root = os.path.join(
            trainer.save_root, self.conf_mat_save_root
        )
        self.report_save_root = os.path.join(
            trainer.save_root, self.report_save_root
        )
        for dir in [self.conf_mat_save_root, self.report_save_root]:
            if not os.path.isdir(dir):
                os.makedirs(dir)

    @rank_zero_only
    def log(self, predictions: Tensor, targets: Tensor):
        self.predictions += predictions.tolist()
        self.targets += targets.tolist()

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        self.save_confusion_matrix_csv_and_fig(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )
        self.save_classification_report(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )
        self.predictions = []
        self.targets = []

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        self.save_confusion_matrix_csv_and_fig(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )
        self.save_classification_report(
            trainer.variables.current_pass, trainer.variables.current_epoch
        )
        self.predictions = []
        self.targets = []

    def save_classification_report(self, which, epoch):
        report = classification_report(
            self.targets, self.predictions, labels=self.labels, zero_division=0, output_dict=True
        )
        report["accuracy"] = {
            "precision": accuracy_score(self.targets, self.predictions),
            "recall": np.nan,
            "f1-score": np.nan,
            "support": np.nan,
        }
        report = pd.DataFrame(report).transpose()
        acc = report.loc["accuracy"]
        report.drop("accuracy", inplace=True)
        report = pd.concat((report, pd.DataFrame(acc).T), axis=0)
        save_to = os.path.join(
            self.report_save_root, f"{which}_ep{epoch}.csv"
        )
        report.to_csv(save_to)

    def save_confusion_matrix_csv_and_fig(self, which, epoch):

        cm = confusion_matrix(self.targets, self.predictions, labels=self.labels)
        cm_df = pd.DataFrame(cm, index=self.labels, columns=self.labels)

        if self.normalize_conf_mat:
            with np.errstate(all='ignore'):
                cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
                cm = np.nan_to_num(cm)  # Replace NaNs with 0

        cm_df = pd.DataFrame(cm, index=self.labels, columns=self.labels)

        acc = float(round(np.einsum("ii->i", cm).sum() / cm.sum() * 100, 2))

        fig_title = 'Confusion Matrix\n'
        fig_title += f"Accuracy: {acc}" 
        cmap = 'Blues'
        fig = plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='.2f' if self.normalize_conf_mat else 'd', cmap=cmap, cbar=True)
        plt.title(fig_title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        save_to = os.path.join(
            self.conf_mat_save_root, f"{which}_ep{epoch}.png"
        )
        fig.savefig(save_to)
        plt.close()

        save_to = os.path.join(
            self.conf_mat_save_root, f"{which}_ep{epoch}.csv"
        )
        cm_df.to_csv(save_to)
