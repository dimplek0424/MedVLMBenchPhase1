"""
medvlm_core.metrics
-------------------
Metrics utilities for projection-type classification (frontal vs lateral).
"""

from typing import Iterable, List, Tuple
from sklearn.metrics import accuracy_score, confusion_matrix

def projection_accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    """
    Compute accuracy for projection type.

    y_* are iterables of strings: 'frontal' or 'lateral'
    """
    return float(accuracy_score(list(y_true), list(y_pred)))

def projection_confusion(
    y_true: Iterable[str], y_pred: Iterable[str],
    labels: Tuple[str, str] = ("frontal", "lateral")
) -> List[List[int]]:
    """
    Return confusion matrix as a nested list for easy JSON serialization.
    """
    cm = confusion_matrix(list(y_true), list(y_pred), labels=list(labels))
    return cm.tolist()
