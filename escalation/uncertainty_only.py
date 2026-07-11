"""
Uncertainty-Only Escalation Policy (BASELINE).

Escalates the K most-uncertain images each round — a fixed query budget,
so this policy and dual_metric always spend the same number of queries
per round and produce comparable, evenly-spaced learning curves — PLUS any
image whose uncertainty exceeds this round's recalibrated threshold, even
past K. That overflow rule means a round with unusually many genuinely
confusing images is never silently capped and losing real signal; K only
sets a floor, not a ceiling.

The PROBLEM with this policy (which our research exposes):
- A melanoma image that the model confidently misclassifies as a benign
  mole has LOW uncertainty → never escalated by this rule alone → patient
  is at risk.
- This policy has no way to catch "confidently wrong on a dangerous case" —
  that's what the risk signal in dual_metric is for.
"""

import numpy as np


def decide(uncertainty_scores, threshold, k_budget):
    """
    Baseline escalation: top-K most uncertain, plus anything above
    threshold (uncapped).

    Parameters
    ----------
    uncertainty_scores : np.ndarray or list
        Uncertainty score for each image, shape (N,).
    threshold : float
        This round's recalibrated uncertainty threshold. Above this →
        escalate, regardless of the K budget.
    k_budget : int
        Number of top-uncertainty images escalated each round, at minimum.
        0 means threshold-only (no budget floor).

    Returns
    -------
    decisions : np.ndarray of str
        'escalate' or 'auto_accept' for each image.
    escalate_idx : np.ndarray
        Indices of images to escalate (query the oracle).
    auto_accept_idx : np.ndarray
        Indices of images to auto-accept.
    """
    scores = np.array(uncertainty_scores)
    n = len(scores)

    over_threshold = np.where(scores > threshold)[0]

    if k_budget > 0 and n > 0:
        k = min(k_budget, n)
        top_k = np.argsort(scores)[::-1][:k]
    else:
        top_k = np.array([], dtype=int)

    escalate_idx = np.union1d(top_k, over_threshold).astype(int)

    decisions = np.array(['auto_accept'] * n)
    decisions[escalate_idx] = 'escalate'
    auto_accept_idx = np.where(decisions == 'auto_accept')[0]

    return decisions, escalate_idx, auto_accept_idx
