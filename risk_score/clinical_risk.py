"""
Clinical Risk Score — Metric 2 of the Dual-Metric Framework.

Risk Score = P(malignant), from the model's independent risk head (see
models/base_model.py: _build_risk_head(), predict_risk()) — a second
classification head sharing the backbone with the 7-class head but trained
on its own binary malignant-vs-non-malignant objective, with its own
parameters.

This replaces an earlier version that summed P(mel) + P(bcc) + P(akiec)
straight out of the 7-way classification softmax. That risk score was not
independent of the model's classification confidence: when the classifier
was confidently wrong, both signals failed together, since they were two
different readings off the same probability vector. See
DUAL_METRIC_ANALYSIS.md for the full analysis and worked examples.

  → Baseline (uncertainty only): auto-accepts when the model seems
    confident, regardless of danger.
  → Dual-metric: the independent risk head can still flag danger even when
    the classification head is confidently (and wrongly) sure of itself,
    because it isn't reading the same softmax.

Range: [0.0, 1.0]. Higher = more likely malignant.
"""


def get_risk_category(risk_score, threshold=0.3):
    """
    Categorize a risk score as 'high_risk' or 'low_risk'.

    Parameters
    ----------
    risk_score : float
        The computed risk score (P(malignant) from the risk head).
    threshold : float
        Risk threshold (default: 0.3).

    Returns
    -------
    str
        'high_risk' or 'low_risk'.
    """
    return 'high_risk' if risk_score >= threshold else 'low_risk'
