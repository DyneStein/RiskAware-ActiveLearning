"""
Least Confidence uncertainty measure.

Uncertainty = 1 - max(p)

If the model's best guess is only 40% confident, uncertainty = 0.6.
If the model is 99% confident, uncertainty = 0.01.

Range: [0, 1 - 1/num_classes]. Higher = more uncertain. This formula was
already raw/unnormalized (no artificial rescaling applied) — it just happens
to fall naturally in a bounded range for a softmax distribution, unlike
entropy or MC dropout.
"""

import numpy as np


def compute_least_confidence(probabilities):
    """
    Compute least-confidence uncertainty for each sample.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape (B, num_classes) or (num_classes,).

    Returns
    -------
    float or np.ndarray
        Least confidence score(s) in [0, 1]. Higher = more uncertain.
    """
    probs = np.array(probabilities)
    single = probs.ndim == 1

    if single:
        probs = probs.reshape(1, -1)

    # Uncertainty = 1 - highest class probability
    uncertainty = 1.0 - probs.max(axis=1)

    return float(uncertainty[0]) if single else uncertainty
