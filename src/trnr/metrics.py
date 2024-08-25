import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


def compute_classification_report_csv(y_true, y_pred, labels, **kwargs):
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

def compute_confusion_matrix_fig_and_csv(y_true, y_pred, labels, normalize: bool, **kwargs):
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Replace NaNs with 0

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    acc = float(round(np.einsum("ii->i", cm).sum() / cm.sum() * 100, 2))

    fig_title = 'Confusion Matrix\n'
    fig_title += f"Accuracy: {acc}" 
    cmap = 'Blues'

    plt.close()
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_df, annot=True, fmt='.2f' if normalize else 'd', 
        cmap=cmap, cbar=True
    )
    plt.title(fig_title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    return fig, cm_df

def update_history_from_classification_report(history: dict, report: pd.DataFrame):
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
            history[key] = value

if __name__ == "__main__":
    pass
    y_true = [1, 2, 3, 4]
    y_pred = [2, 2, 3, 3]
    labels = [1, 2, 3, 4, 5]
    
    fig, csv = compute_confusion_matrix_fig_and_csv(y_true, y_pred, labels, True)
    report = compute_classification_report_csv(y_true, y_pred, labels)
    
    
    history = {}
    update_epoch_history_from_classification_report(history, report)
    print(history)
