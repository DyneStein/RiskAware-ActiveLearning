"""
MC Dropout uncertainty measure.

Runs N stochastic forward passes with dropout enabled.
Uncertainty = mean variance across all class predictions.

Higher variance = predictions disagree across passes = more uncertain.

Range: theoretical ceiling ~0.25 per class (max variance of a Bernoulli at
p=0.5), reported RAW — not rescaled to [0, 1]. Each uncertainty method is
left in its own natural scale; the escalation threshold for this method is
calibrated separately from the seed data (see
active_learning/al_loop.py calibrate_thresholds()) rather than assuming a
shared [0, 1] range across methods.
"""

import numpy as np


def compute_mc_dropout_uncertainty(all_probs):
    """
    Compute raw (unnormalized) MC Dropout uncertainty from multiple forward
    pass predictions.

    Parameters
    ----------
    all_probs : np.ndarray
        Predictions from N forward passes.
        Shape: (n_passes, num_classes) for single image, or
               (n_passes, B, num_classes) for a batch.

    Returns
    -------
    float or np.ndarray
        Mean variance across classes. Higher = more uncertain.
    """
    probs = np.array(all_probs)
    single = probs.ndim == 2  # (n_passes, num_classes)

    if single:
        probs = probs[:, np.newaxis, :]  # (n_passes, 1, num_classes)

    # Variance across forward passes for each class
    variance = probs.var(axis=0)  # (B, num_classes)

    # Mean variance across classes as the uncertainty score
    uncertainty = variance.mean(axis=1)  # (B,)

    return float(uncertainty[0]) if single else uncertainty
