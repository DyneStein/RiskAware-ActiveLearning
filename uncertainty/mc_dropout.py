"""
MC Dropout uncertainty measure.

Runs N stochastic forward passes with dropout enabled.
Uncertainty = mean variance across all class predictions.

Higher variance = predictions disagree across passes = more uncertain.
"""

import numpy as np


def compute_mc_dropout_uncertainty(all_probs):
    """
    Compute MC Dropout uncertainty from multiple forward pass predictions.

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
        Range: approximately [0, 0.25] (normalized to [0, 1]).
    """
    probs = np.array(all_probs)
    single = probs.ndim == 2  # (n_passes, num_classes)

    if single:
        probs = probs[:, np.newaxis, :]  # (n_passes, 1, num_classes)

    # Variance across forward passes for each class
    variance = probs.var(axis=0)  # (B, num_classes)

    # Mean variance across classes as the uncertainty score
    uncertainty = variance.mean(axis=1)  # (B,)

    # Normalize: max variance of a Bernoulli is 0.25 (at p=0.5)
    # Scale to [0, 1] range
    normalized = np.clip(uncertainty / 0.25, 0.0, 1.0)

    return float(normalized[0]) if single else normalized
