"""
Softmax Entropy uncertainty measure.

H = -Σ p_i · log(p_i)

Higher entropy = model is spread across multiple classes = more uncertain.
Lower entropy = model is concentrated on one class = more confident.

Range: [0, log(num_classes)] — reported RAW (not rescaled to [0, 1]). We
deliberately do not divide by max entropy: each uncertainty method is left
in its own natural scale, and the escalation threshold for this method is
calibrated separately from the seed data (see
active_learning/al_loop.py calibrate_thresholds()) rather than assuming a
shared [0, 1] range across methods.
"""

import numpy as np


def compute_entropy(probabilities):
    """
    Compute raw (unnormalized) softmax entropy for each sample.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape (B, num_classes) or (num_classes,).

    Returns
    -------
    np.ndarray or float
        Entropy score(s) in [0, log(num_classes)]. Higher = more uncertain.
    """
    probs = np.array(probabilities)
    single = probs.ndim == 1

    if single:
        probs = probs.reshape(1, -1)

    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    # Entropy: H = -Σ p_i * log(p_i)
    entropy = -np.sum(probs * np.log(probs), axis=1)

    return float(entropy[0]) if single else entropy
