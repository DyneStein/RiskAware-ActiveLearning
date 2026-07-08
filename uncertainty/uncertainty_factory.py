"""
Uncertainty method factory.
Returns the appropriate uncertainty function by name string.
"""

from .entropy import compute_entropy
from .mc_dropout import compute_mc_dropout_uncertainty
from .margin import compute_margin
from .least_confidence import compute_least_confidence


_UNCERTAINTY_REGISTRY = {
    'entropy': compute_entropy,
    'mc_dropout': compute_mc_dropout_uncertainty,
    'margin': compute_margin,
    'least_confidence': compute_least_confidence,
}


def get_uncertainty_method(name):
    """
    Get an uncertainty computation function by name.

    Parameters
    ----------
    name : str
        One of: 'entropy', 'mc_dropout', 'margin', 'least_confidence'.

    Returns
    -------
    callable
        The uncertainty function.

    Note
    ----
    - entropy, margin, least_confidence: take probabilities (B, num_classes)
    - mc_dropout: takes all_probs from MC passes (n_passes, B, num_classes)
    """
    if name not in _UNCERTAINTY_REGISTRY:
        raise ValueError(
            f"Unknown uncertainty method: '{name}'. "
            f"Available: {list(_UNCERTAINTY_REGISTRY.keys())}"
        )
    return _UNCERTAINTY_REGISTRY[name]


def list_uncertainty_methods():
    """Return list of available uncertainty method names."""
    return list(_UNCERTAINTY_REGISTRY.keys())
