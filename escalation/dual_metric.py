"""
Dual-Metric Escalation Policy (OUR CONTRIBUTION).

Combines uncertainty score + clinical risk score in a 2×2 grid:

                    Low Risk          High Risk
                ┌─────────────────┬─────────────────┐
  Low           │                 │                 │
  Uncertainty   │  AUTO-ACCEPT    │  ESCALATE ⚠️    │
                │  (efficient)    │  (SAFETY        │
                │                 │   OVERRIDE)     │
                ├─────────────────┼─────────────────┤
  High          │                 │                 │
  Uncertainty   │  AUTO-ACCEPT    │  ESCALATE 🚨    │
                │  (batch review  │  (always send   │
                │   optional)     │   to oracle)    │
                └─────────────────┴─────────────────┘

KEY DIFFERENCE from baseline:
  Row 1, Col 2 → "Low uncertainty + High risk"
  The model is CONFIDENT but the case is DANGEROUS.
  Baseline: auto-accepts (model seems sure) → MISSES THE CANCER
  Ours: forces escalation (risk score overrides confidence) → CATCHES IT
"""

import numpy as np


def decide(uncertainty_scores, risk_scores, unc_threshold, risk_threshold):
    """
    Dual-metric escalation: use both uncertainty AND risk to decide.

    Parameters
    ----------
    uncertainty_scores : np.ndarray
        Uncertainty score for each image, shape (N,).
    risk_scores : np.ndarray
        Clinical risk score for each image, shape (N,).
    unc_threshold : float
        Uncertainty threshold. Above this → "high uncertainty".
    risk_threshold : float
        Risk threshold. Above this → "high risk".

    Returns
    -------
    decisions : np.ndarray of str
        'escalate' or 'auto_accept' for each image.
    escalate_indices : np.ndarray
        Indices of images to escalate.
    auto_accept_indices : np.ndarray
        Indices of images to auto-accept.
    categories : np.ndarray of str
        The 2×2 category for each image:
        'low_unc_low_risk', 'low_unc_high_risk',
        'high_unc_low_risk', 'high_unc_high_risk'
    """
    unc = np.array(uncertainty_scores)
    risk = np.array(risk_scores)
    n = len(unc)

    decisions = np.array(['auto_accept'] * n)
    categories = np.array([''] * n, dtype='U25')

    for i in range(n):
        high_unc = unc[i] > unc_threshold
        high_risk = risk[i] > risk_threshold

        if not high_unc and not high_risk:
            # Low uncertainty + Low risk → auto-accept (safe & efficient)
            decisions[i] = 'auto_accept'
            categories[i] = 'low_unc_low_risk'

        elif not high_unc and high_risk:
            # Low uncertainty + High risk → ESCALATE (safety override!)
            # This is the critical case that baseline misses.
            decisions[i] = 'escalate'
            categories[i] = 'low_unc_high_risk'

        elif high_unc and not high_risk:
            # High uncertainty + Low risk → auto-accept (efficiency gain)
            decisions[i] = 'auto_accept'
            categories[i] = 'high_unc_low_risk'

        elif high_unc and high_risk:
            # High uncertainty + High risk → ESCALATE (always)
            decisions[i] = 'escalate'
            categories[i] = 'high_unc_high_risk'

    escalate_idx = np.where(decisions == 'escalate')[0]
    auto_accept_idx = np.where(decisions == 'auto_accept')[0]

    return decisions, escalate_idx, auto_accept_idx, categories
