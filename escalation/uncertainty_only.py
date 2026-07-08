"""
Uncertainty-Only Escalation Policy (BASELINE).

This is the standard approach from existing literature:
- If uncertainty > threshold → escalate to human expert (oracle)
- If uncertainty ≤ threshold → auto-accept the model's prediction

The PROBLEM with this policy (which our research exposes):
- A melanoma image that the model confidently misclassifies as a benign mole
  has LOW uncertainty → gets auto-accepted → patient is at risk.
- This policy has no way to catch "confidently wrong on a dangerous case."
"""

import numpy as np


def decide(uncertainty_scores, threshold):
    """
    Baseline escalation: escalate all cases above the uncertainty threshold.

    Parameters
    ----------
    uncertainty_scores : np.ndarray or list
        Uncertainty score for each image, shape (N,).
    threshold : float
        Uncertainty threshold. Above this → escalate.

    Returns
    -------
    decisions : np.ndarray of str
        'escalate' or 'auto_accept' for each image.
    escalate_indices : np.ndarray
        Indices of images to escalate (query the oracle).
    auto_accept_indices : np.ndarray
        Indices of images to auto-accept.
    """
    scores = np.array(uncertainty_scores)
    decisions = np.where(scores > threshold, 'escalate', 'auto_accept')

    escalate_idx = np.where(scores > threshold)[0]
    auto_accept_idx = np.where(scores <= threshold)[0]

    return decisions, escalate_idx, auto_accept_idx
