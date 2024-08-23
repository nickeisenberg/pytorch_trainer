from collections import defaultdict
import os
import csv

import numpy as np
import pandas as pd
from torch import Tensor

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score
)

from .base import Logger as _Logger
from ..utils import rank_zero_only
from ...trainer import Trainer


def compute_weighted_precision_score(y_true, y_pred, labels):
    return precision_score(
        y_true=y_true, y_pred=y_pred, labels=labels, zero_division=0, average="weighted"
    )

def compute_weighted_recall_score(y_true, y_pred, labels):
    return recall_score(
        y_true=y_true, y_pred=y_pred, labels=labels, zero_division=0, average="weighted"
    )

def compute_classification_summary_csv(y_true, y_pred, labels):
    report = classification_report(
        y_true=y_true, y_pred=y_pred, labels=labels, zero_division=0, output_dict=True
    )
    report["accuracy"] = {
        "precision": accuracy_score(y_true, y_pred),
        "recall": np.nan,
        "f1-score": np.nan,
        "support": np.nan,
    }
    report = pd.DataFrame(report).transpose()
    acc = report.loc["accuracy"]
    report.drop("accuracy", inplace=True)
    report = pd.concat((report, pd.DataFrame(acc).T), axis=0)
    return report


class ClassificationLogger(_Logger):
    """The logger with functionality suited for classification models"""
    def __init__(self, log_root: str = "logs", priority=0):
        super().__init__(priority=priority)

        self.log_root = log_root 

        self.batch_history = defaultdict(float)
        self.epoch_history = defaultdict(list)

        self.epoch_predictions = []
        self.epoch_targets = []

    @rank_zero_only
    def log_item(self, name: str, value: float):
        self.batch_history[name] = value
        self.epoch_history[name].append(value)

    @rank_zero_only
    def log_targs_and_preds(self, targets: Tensor, predictions: Tensor):
        self.epoch_targets += targets.tolist()
        self.epoch_predictions += predictions.tolist()
    
    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):
        self.log_root = os.path.join(trainer.save_root, self.log_root)
        if not os.path.isdir(self.log_root):
            os.makedirs(self.log_root)

        self.train_log_path = os.path.join(
            self.log_root, f"train.csv"
        )
        self.validation_log_path = os.path.join(
            self.log_root, f"validation.csv"
        )
        self.train_headers_written = False
        self.validation_headers_written = False

    @rank_zero_only
    def before_train_epoch_pass(self, trainer: Trainer):
        pass

    @rank_zero_only
    def after_train_batch_pass(self, trainer: Trainer):
        with open(self.train_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_history.keys())
            if not self.train_headers_written:
                _ = writer.writeheader()
                self.train_headers_written = True
            _ = writer.writerow(self.batch_history)
        self.batch_history = defaultdict(float)

    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_history:
            self.train_log[key].append(round(sum(self.epoch_history[key]) / len(self.epoch_history[key]), 4))
        self.epoch_history = defaultdict(list)

    @rank_zero_only
    def before_validation_epoch_pass(self, trainer: Trainer):
        self._headers_written = False
    
    @rank_zero_only
    def after_validation_batch_pass(self, trainer: Trainer):
        with open(self.validation_log_path, "a", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=self.batch_history.keys())
            if not self.validation_headers_written:
                _ = writer.writeheader()
                self.validation_headers_written = True
            _ = writer.writerow(self.batch_history)
        self.batch_history = defaultdict(float)

    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_history:
            self.validation_log[key].append(round(sum(self.epoch_history[key]) / len(self.epoch_history[key]), 4))
        self.epoch_history = defaultdict(list)
