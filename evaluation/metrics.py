"""
Evaluation metrics for the Risk-Aware Active Learning framework.

Computes:
- Overall accuracy
- F1 score (macro and per-class)
- False-negative rate on malignant classes (PRIMARY SAFETY METRIC)
- False-negative rate on melanoma specifically
- Confusion matrix
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, precision_recall_fscore_support
)

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from constants import HIGH_RISK_CLASSES
from config import CLASS_NAMES, CLASS_TO_IDX


def compute_fn_rate_malignant(y_true, y_pred):
    """
    Compute false-negative rate on HIGH-RISK (malignant) classes.

    FN Rate = cases where true label is malignant but prediction is benign,
              divided by total actual malignant cases.

    This is the PRIMARY SAFETY METRIC of the paper.
    A missed melanoma is far more dangerous than a false alarm.

    Parameters
    ----------
    y_true : np.ndarray
        True label indices.
    y_pred : np.ndarray
        Predicted label indices.

    Returns
    -------
    float
        False-negative rate on malignant classes [0, 1].
    """
    high_risk_indices = {CLASS_TO_IDX[c] for c in HIGH_RISK_CLASSES}

    # True malignant cases
    is_malignant = np.array([y in high_risk_indices for y in y_true])
    total_malignant = is_malignant.sum()

    if total_malignant == 0:
        return 0.0

    # Of those, how many were predicted as benign (false negatives)?
    predicted_benign = np.array([y not in high_risk_indices for y in y_pred])
    false_negatives = (is_malignant & predicted_benign).sum()

    return false_negatives / total_malignant


def compute_fn_rate_melanoma(y_true, y_pred):
    """
    Compute false-negative rate specifically for melanoma (mel).
    Same logic as above but only for the most dangerous class.
    """
    mel_idx = CLASS_TO_IDX['mel']

    is_melanoma = (y_true == mel_idx)
    total_melanoma = is_melanoma.sum()

    if total_melanoma == 0:
        return 0.0

    predicted_not_melanoma = (y_pred != mel_idx)
    false_negatives = (is_melanoma & predicted_not_melanoma).sum()

    return false_negatives / total_melanoma


def compute_all_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    Compute all evaluation metrics for a round.

    Parameters
    ----------
    y_true : np.ndarray
        True label indices, shape (N,).
    y_pred : np.ndarray
        Predicted label indices, shape (N,).
    y_probs : np.ndarray
        Predicted probabilities, shape (N, num_classes).
    class_names : list of str, optional
        Class names for per-class metrics.

    Returns
    -------
    dict
        Dictionary of all computed metrics.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Core metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Per-class F1
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))),
        zero_division=0
    )

    # Safety metrics
    fn_rate_malignant = compute_fn_rate_malignant(y_true, y_pred)
    fn_rate_melanoma = compute_fn_rate_melanoma(y_true, y_pred)

    # Confusion matrix
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(class_names)))
    )

    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'fn_rate_malignant': float(fn_rate_malignant),
        'fn_rate_melanoma': float(fn_rate_melanoma),
    }

    # Add per-class metrics
    for i, name in enumerate(class_names):
        metrics[f'f1_{name}'] = float(f1_per[i])
        metrics[f'precision_{name}'] = float(precision[i])
        metrics[f'recall_{name}'] = float(recall[i])

    return metrics
