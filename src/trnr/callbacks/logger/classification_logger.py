from collections import defaultdict
import os
import csv
import pandas as pd
import numpy as np

import torch
from torch import Tensor

from .base import Logger as _Logger
from ...metrics import (
    compute_confusion_matrix_fig_and_csv,
    compute_classification_report_csv, 
)
from ..utils import rank_zero_only
from ...trainer import Trainer


class ClassificationLogger(_Logger):
    """The logger with functionality suited for classification models"""
    def __init__(self, labels: list[int], log_root: str = "logs", 
                 metrics_root: str = "metrics", priority=0):
        super().__init__(priority=priority)
        
        self.labels = labels

        self.log_root = log_root 
        self.metrics_root = metrics_root 

        self.batch_history = defaultdict(float)
        self.epoch_history = defaultdict(list)

        self.epoch_targets = torch.tensor([])
        self.epoch_predictions = torch.tensor([])


    @rank_zero_only
    def log_item(self, name: str, value: float):
        self.batch_history[name] = value
        self.epoch_history[name].append(value)


    @rank_zero_only
    def log_targs_and_preds(self, targets: Tensor, predictions: Tensor):
        self.epoch_targets = torch.hstack(
            (self.epoch_targets, targets.detach().to("cpu"))
        )
        self.epoch_predictions = torch.hstack(
            (self.epoch_predictions, predictions.detach().to("cpu"))
        )
    

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer):

        self.log_root = os.path.join(trainer.save_root, self.log_root)

        self.metrics_root = os.path.join(trainer.save_root, self.metrics_root)
        self.conf_mat_root = os.path.join(self.metrics_root, "conf_mat")
        self.report_root = os.path.join(self.metrics_root, "report")
        
        for dir in [self.log_root, self.conf_mat_root, self.report_root]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
        
        # set the paths to the logs
        self.train_log_path = os.path.join(
            self.log_root, f"train.csv"
        )
        self.train_headers_written = False

        self.train_all_log_path = os.path.join(
            self.log_root, f"train_all.csv"
        )
        self.train_all_headers_written = False

        self.validation_log_path = os.path.join(
            self.log_root, f"validation.csv"
        )
        self.validation_headers_written = False

        self.validation_all_log_path = os.path.join(
            self.log_root, f"validation_all.csv"
        )
        self.validation_all_headers_written = False


    @rank_zero_only
    def after_train_batch_pass(self, trainer: Trainer):
        write_to_log(
            path=self.train_all_log_path, 
            history=self.batch_history, 
            write_headers=not self.train_all_headers_written
        )
        self.train_all_headers_written = True
        self.batch_history = defaultdict(float)


    @rank_zero_only
    def after_train_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_history:
            self.train_log[key].append(
                round(sum(self.epoch_history[key]) / len(self.epoch_history[key]), 4)
            )

        if len(self.epoch_targets) > 0:
            cm_fig, cm_csv = compute_confusion_matrix_fig_and_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels, normalize=True
            )

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.png"
            )
            cm_fig.savefig(save_to)

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            cm_csv.to_csv(save_to)

            save_to = os.path.join(
                self.report_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            report = compute_classification_report_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels
            )
            report.to_csv(save_to)
            update_log_from_classification_report(self.train_log, report)

        _ = pd.DataFrame(self.train_log).to_csv(self.train_log_path)
        
        self.epoch_history = defaultdict(list)
        self.epoch_targets, self.epoch_predictions = torch.tensor([]), torch.tensor([])
    

    @rank_zero_only
    def after_validation_batch_pass(self, trainer: Trainer):
        write_to_log(
            self.validation_all_log_path, 
            self.batch_history, 
            not self.validation_all_headers_written
        )
        self.validation_all_headers_written = True
        self.batch_history = defaultdict(float)


    @rank_zero_only
    def after_validation_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_history:
            self.validation_log[key].append(
                round(sum(self.epoch_history[key]) / len(self.epoch_history[key]), 4)
            )

        if len(self.epoch_targets) > 0:
            cm_fig, cm_csv = compute_confusion_matrix_fig_and_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels, normalize=True
            )

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.png"
            )
            cm_fig.savefig(save_to)

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            cm_csv.to_csv(save_to)

            save_to = os.path.join(
                self.report_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            report = compute_classification_report_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels
            )
            report.to_csv(save_to)
            update_log_from_classification_report(self.validation_log, report)

        _ = pd.DataFrame(self.validation_log).to_csv(self.validation_log_path)
        
        self.epoch_history = defaultdict(list)
        self.epoch_targets, self.epoch_predictions = torch.tensor([]), torch.tensor([])

    @rank_zero_only
    def on_evaluation_start(self, trainer: Trainer):

        self.log_root = os.path.join(trainer.save_root, self.log_root)

        self.metrics_root = os.path.join(trainer.save_root, self.metrics_root)
        self.conf_mat_root = os.path.join(self.metrics_root, "conf_mat")
        self.report_root = os.path.join(self.metrics_root, "report")
        
        for dir in [self.log_root, self.conf_mat_root, self.report_root]:
            if not os.path.isdir(dir):
                os.makedirs(dir)
        
        self.evaluation_log_path = os.path.join(
            self.log_root, f"evaluation.csv"
        )
        self.evaluation_headers_written = False

        self.evaluation_all_log_path = os.path.join(
            self.log_root, f"evaluation_all.csv"
        )
        self.evaluation_all_headers_written = False

    @rank_zero_only
    def after_evaluation_batch_pass(self, trainer: Trainer):
        write_to_log(
            path=self.evaluation_all_log_path, 
            history=self.batch_history, 
            write_headers=not self.evaluation_all_headers_written
        )
        self.evaluation_all_headers_written = True
        self.batch_history = defaultdict(float)

    @rank_zero_only
    def after_evaluation_epoch_pass(self, trainer: Trainer):
        for key in self.epoch_history:
            self.evaluation_log[key].append(
                round(sum(self.epoch_history[key]) / len(self.epoch_history[key]), 4)
            )

        if len(self.epoch_targets) > 0:
            cm_fig, cm_csv = compute_confusion_matrix_fig_and_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels, normalize=True
            )

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.png"
            )
            cm_fig.savefig(save_to)

            save_to = os.path.join(
                self.conf_mat_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            cm_csv.to_csv(save_to)

            save_to = os.path.join(
                self.report_root, 
                f"{trainer.variables.current_pass}_ep{trainer.variables.current_epoch}.csv"
            )
            report = compute_classification_report_csv(
                y_true=self.epoch_targets, y_pred=self.epoch_predictions,
                labels=self.labels
            )
            report.to_csv(save_to)
            update_log_from_classification_report(self.evaluation_log, report)

        _ = pd.DataFrame(self.evaluation_log).to_csv(self.evaluation_log_path)
        
        self.epoch_history = defaultdict(list)
        self.epoch_targets, self.epoch_predictions = torch.tensor([]), torch.tensor([])


def write_to_log(path: str, history: dict, write_headers: bool):
    with open(path, "a", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=history.keys())
        if write_headers:
            _ = writer.writeheader()
        _ = writer.writerow(history)


def update_log_from_classification_report(history: dict[str, list[float]], 
                                          report: pd.DataFrame
    ):
    cols_to_check = [
        "precision", "recall", "f1-score"
    ]
    inds_to_check = ["micro avg", "macro avg", "weighted avg", "accuracy"]

    inds = np.intersect1d(report.index.tolist(), inds_to_check).tolist()
    cols = np.intersect1d(report.columns.tolist(), cols_to_check).tolist()
    for col in cols:
        for ind in inds:
            key = f"{col}_{ind}".replace(" ", "-")
            if "accuracy" in key:
                key = "accuracy"
            value = report.loc[ind][col]
            if np.isnan(value):
                continue
            history[key].append(value)
