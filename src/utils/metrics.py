"""Utility wrappers around sklearn metrics for convenience."""
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    y_prob: List[List[float]],
) -> Dict[str, float]:
    """
    Compute a standard set of classification metrics.

    Parameters
    ----------
    y_true : ground-truth integer labels
    y_pred : predicted integer labels
    y_prob : softmax probabilities per class

    Returns
    -------
    dict with accuracy, f1_macro, f1_weighted, precision, recall, roc_auc
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "f1_macro":    float(f1_score(y_true, y_pred, average="macro",    zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision":   float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, average="macro",    zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        metrics["roc_auc"] = 0.0

    return metrics
