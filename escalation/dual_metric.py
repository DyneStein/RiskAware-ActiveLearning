"""
Dual-Metric Escalation Policy (OUR CONTRIBUTION).

Two genuinely independent escalation routes, combined with OR:

1. Uncertainty route (same mechanism as the baseline in
   uncertainty_only.py): the K most-uncertain images this round, plus
   anything above this round's recalibrated uncertainty threshold even
   past K.
2. Risk route: ANY image where the independent risk head's P(malignant)
   (see risk_score/clinical_risk.py) exceeds this round's recalibrated
   risk threshold — uncapped, no budget. A genuinely dangerous case must
   never be skipped just because the uncertainty budget K is full that
   round.

An image can be escalated by either route, both, or neither:

                    Low Risk          High Risk
                ┌─────────────────┬─────────────────┐
  Low           │  AUTO-ACCEPT    │  ESCALATE ⚠️     │
  Uncertainty   │                 │  (risk route)    │
                ├─────────────────┼─────────────────┤
  High          │  ESCALATE       │  ESCALATE 🚨     │
  Uncertainty   │  (unc. route)   │  (both routes)   │
                └─────────────────┴─────────────────┘

KEY DIFFERENCE from the baseline:
  Low uncertainty + High risk → the model is CONFIDENT but the case is
  DANGEROUS. Baseline: never escalated by uncertainty alone → misses the
  cancer. Ours: the risk route escalates it regardless of confidence.

  High uncertainty + Low risk → the model is genuinely unsure, but the
  case looks safe. Still escalated — this is real active learning, using
  the label to improve the model, not just a safety catch. (Earlier
  versions of this policy auto-accepted this cell, which meant uncertainty
  never actually participated in the decision — see
  DUAL_METRIC_ANALYSIS.md.)

`categories` records which of the four quadrants each image falls into,
for the scatter plots and analysis — descriptive only, it no longer
solely drives the decision the way it originally did.
"""

import numpy as np


def decide(uncertainty_scores, risk_scores, unc_threshold, risk_threshold,
           k_budget):
    """
    Dual-metric escalation: uncertainty (budgeted, top-K + overflow) OR
    risk (uncapped).

    Parameters
    ----------
    uncertainty_scores : np.ndarray
        Uncertainty score for each image, shape (N,).
    risk_scores : np.ndarray
        Risk score (P(malignant) from the independent risk head) for each
        image, shape (N,).
    unc_threshold : float
        This round's recalibrated uncertainty threshold.
    risk_threshold : float
        This round's recalibrated risk threshold (or a manual override).
    k_budget : int
        Number of top-uncertainty images escalated each round, at minimum.
        0 means uncertainty is threshold-only (no budget floor); the risk
        route is always uncapped regardless of this value.

    Returns
    -------
    decisions : np.ndarray of str
        'escalate' or 'auto_accept' for each image.
    escalate_idx : np.ndarray
        Indices of images to escalate.
    auto_accept_idx : np.ndarray
        Indices of images to auto-accept.
    categories : np.ndarray of str
        The 2×2 category for each image (descriptive, for plots/analysis):
        'low_unc_low_risk', 'low_unc_high_risk',
        'high_unc_low_risk', 'high_unc_high_risk'
    """
    unc = np.array(uncertainty_scores)
    risk = np.array(risk_scores)
    n = len(unc)

    over_unc_threshold = np.where(unc > unc_threshold)[0]
    if k_budget > 0 and n > 0:
        k = min(k_budget, n)
        top_k = np.argsort(unc)[::-1][:k]
    else:
        top_k = np.array([], dtype=int)
    uncertainty_escalate = np.union1d(top_k, over_unc_threshold)

    risk_escalate = np.where(risk > risk_threshold)[0]

    escalate_idx = np.union1d(uncertainty_escalate, risk_escalate).astype(int)

    decisions = np.array(['auto_accept'] * n)
    decisions[escalate_idx] = 'escalate'
    auto_accept_idx = np.where(decisions == 'auto_accept')[0]

    categories = np.array([''] * n, dtype='U25')
    high_unc_mask = unc > unc_threshold
    high_risk_mask = risk > risk_threshold
    categories[~high_unc_mask & ~high_risk_mask] = 'low_unc_low_risk'
    categories[~high_unc_mask & high_risk_mask] = 'low_unc_high_risk'
    categories[high_unc_mask & ~high_risk_mask] = 'high_unc_low_risk'
    categories[high_unc_mask & high_risk_mask] = 'high_unc_high_risk'

    return decisions, escalate_idx, auto_accept_idx, categories
