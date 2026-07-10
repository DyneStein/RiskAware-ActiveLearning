"""
Margin sampling uncertainty measure.

Margin = p_top1 - p_top2
Uncertainty = 1 - margin

Small margin = model can't decide between top two classes = more uncertain.
Large margin = model is clearly picking one class = more confident.

Range: [0, 1]. Higher = more uncertain. This formula was already raw/
unnormalized (no artificial rescaling applied) — it just happens to fall
naturally in [0, 1] for a softmax distribution, unlike entropy or MC dropout.
"""

import numpy as np


def compute_margin(probabilities):
    """
    Compute margin-based uncertainty for each sample.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape (B, num_classes) or (num_classes,).

    Returns
    -------
    float or np.ndarray
        Margin uncertainty score(s) in [0, 1]. Higher = more uncertain.
    """
    probs = np.array(probabilities)
    single = probs.ndim == 1

    if single:
        probs = probs.reshape(1, -1)

    # Sort probabilities in descending order
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]

    # Margin = difference between top-1 and top-2 probabilities
    margin = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Uncertainty = 1 - margin (small margin → high uncertainty)
    uncertainty = 1.0 - margin

    return float(uncertainty[0]) if single else uncertainty
