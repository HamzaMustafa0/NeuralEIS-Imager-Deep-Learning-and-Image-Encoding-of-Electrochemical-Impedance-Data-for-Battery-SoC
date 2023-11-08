"""Utility metrics and reporting wrappers."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray, target_names: list[str] | None = None):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    return cm, report

def accuracy_from_confusion(cm: np.ndarray):
    tot = cm.sum()
    acc = np.trace(cm) / tot * 100.0 if tot else 0.0
    row_acc = []
    for i in range(cm.shape[0]):
        row_sum = cm[i].sum()
        row_acc.append((cm[i, i] / row_sum * 100.0) if row_sum else 0.0)
    return acc, row_acc
