"""
Clinical Risk Score — Metric 2 of the Dual-Metric Framework.

Risk Score = P(mel) + P(bcc) + P(akiec)

This is the sum of the model's predicted probabilities for all HIGH-RISK
(malignant) classes, regardless of which class the model thinks is most likely.

Example:
  Model predicts: nv=0.45, mel=0.25, bkl=0.15, bcc=0.10, akiec=0.03, df=0.01, vasc=0.01
  Top prediction: nv (benign mole) — model is "confident"
  Risk score: 0.25 + 0.10 + 0.03 = 0.38 — HIGH RISK!

  → Baseline (uncertainty only): auto-accepts (model seems confident)
  → Our 2×2 system: forces oracle review (risk score > threshold)
  → This catches the dangerous melanoma that the baseline missed.

Range: [0.0, 1.0]. Higher = more likely malignant.
"""

import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import CLASS_NAMES
from constants import HIGH_RISK_CLASSES


# Precompute which indices in CLASS_NAMES are high-risk
_HIGH_RISK_INDICES = [
    i for i, name in enumerate(CLASS_NAMES) if name in HIGH_RISK_CLASSES
]


def compute_risk_score(probabilities):
    """
    Compute clinical risk score by summing malignant class probabilities.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape (B, num_classes) or (num_classes,).
        Class order must match config.CLASS_NAMES:
        ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

    Returns
    -------
    float or np.ndarray
        Risk score(s) in [0.0, 1.0].
        Sum of P(mel) + P(bcc) + P(akiec).
    """
    probs = np.array(probabilities)
    single = probs.ndim == 1

    if single:
        probs = probs.reshape(1, -1)

    # Sum probabilities of high-risk classes
    risk = probs[:, _HIGH_RISK_INDICES].sum(axis=1)

    return float(risk[0]) if single else risk


def get_risk_category(risk_score, threshold=0.3):
    """
    Categorize a risk score as 'high_risk' or 'low_risk'.

    Parameters
    ----------
    risk_score : float
        The computed risk score.
    threshold : float
        Risk threshold (default: 0.3).

    Returns
    -------
    str
        'high_risk' or 'low_risk'.
    """
    return 'high_risk' if risk_score >= threshold else 'low_risk'
