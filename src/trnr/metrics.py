import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


def compute_classification_report_csv(y_true, y_pred, labels):
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

def compute_confusion_matrix_fig_and_csv(y_true, y_pred, labels, normalize: bool):
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
