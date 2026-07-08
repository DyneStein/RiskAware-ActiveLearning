"""
Softmax Entropy uncertainty measure.

H = -Σ p_i · log(p_i)

Higher entropy = model is spread across multiple classes = more uncertain.
Lower entropy = model is concentrated on one class = more confident.

Range: [0, log(num_classes)] → normalized to [0, 1].
"""

import numpy as np


def compute_entropy(probabilities):
    """
    Compute normalized softmax entropy for each sample.

    Parameters
    ----------
    probabilities : np.ndarray
        Softmax probabilities, shape (B, num_classes) or (num_classes,).

    Returns
    -------
    np.ndarray or float
        Entropy score(s) normalized to [0, 1].
    """
    probs = np.array(probabilities)
    single = probs.ndim == 1

    if single:
        probs = probs.reshape(1, -1)

    # Clip to avoid log(0)
    probs = np.clip(probs, 1e-10, 1.0)

    # Entropy: H = -Σ p_i * log(p_i)
    entropy = -np.sum(probs * np.log(probs), axis=1)

    # Normalize by max possible entropy (uniform distribution)
    max_entropy = np.log(probs.shape[1])
    normalized = entropy / max_entropy

    return float(normalized[0]) if single else normalized
