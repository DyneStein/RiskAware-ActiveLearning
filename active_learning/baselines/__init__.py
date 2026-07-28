"""
Recent active-learning baselines, for comparison against the dual-metric
escalation policy.

    coreset  Sener & Savarese, ICLR 2018       — pure diversity
    badge    Ash et al., ICLR 2020             — uncertainty x diversity
    clue     Prabhu et al., ICCV 2021          — entropy-weighted clustering
    vaal     Sinha et al., ICCV 2019           — task-agnostic, adversarial

WHAT THESE ARE, AND WHAT THEY ARE NOT
-------------------------------------
None of these is a pretrained model. There is nothing to download. Each is
an *algorithm* that replaces one step of the active-learning loop — the
step that decides which images to send to the oracle — and each therefore
requires a complete 15-round run with the backbone retrained from scratch,
exactly like the original experiments.

They also do not multiply out against the existing 24-experiment matrix.
That matrix is 3 backbones x 4 uncertainty methods x 2 policies, but the
uncertainty method and the policy are dials that only our framework has:
CoreSet never computes an uncertainty score, CLUE uses entropy as a fixed
internal component rather than a swappable one, BADGE builds its own
gradient embedding, and VAAL does not look at the classifier at all. Each
baseline *is* a policy. So the correct comparison matrix is

    3 backbones x 4 baselines = 12 runs

not 96.

COST-MATCHING
-------------
Every baseline is given exactly the number of labels the dual-metric
policy spent in the same round on the same backbone — see budgets.py for
why that is the only fair way to compare an acquisition strategy against
an escalation policy.
"""

import numpy as np

from .budgets import load_matched_budgets, reference_experiment_id
from .features import extract_features

STRATEGIES = ("coreset", "badge", "clue", "vaal")

# Human-readable, for logs, plot legends and table headers — so the paper
# and the console agree on what each method is called.
STRATEGY_LABELS = {
    "coreset": "CoreSet (Sener & Savarese, 2018)",
    "badge": "BADGE (Ash et al., 2020)",
    "clue": "CLUE (Prabhu et al., 2021)",
    "vaal": "VAAL (Sinha et al., 2019)",
}


def select_batch(strategy, k, *, model, labeled_dataset, unlabeled_dataset,
                 rng, batch_size=32, num_workers=2, vaal_epochs=5,
                 verbose=True):
    """
    Choose k images from the unlabelled pool using the named strategy.

    Each strategy needs a different view of the data, so the expensive
    parts are computed only when the chosen strategy actually needs them:
    VAAL needs raw images and no classifier features at all; CoreSet needs
    features for both the labelled and unlabelled sets; BADGE and CLUE
    need only the unlabelled side.

    Parameters
    ----------
    strategy : str
        One of STRATEGIES.
    k : int
        Budget for this round (cost-matched — see budgets.py).
    model : BaseModel
        The classifier, already trained on the current labelled set.
    labeled_dataset, unlabeled_dataset : HAM10000Dataset
        Built with evaluation transforms, in pool order.
    rng : np.random.Generator
        Seeded per round by the caller, so selection is reproducible.

    Returns
    -------
    np.ndarray of int — indices into the unlabelled pool.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Available: {', '.join(STRATEGIES)}"
        )

    n = len(unlabeled_dataset)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    if verbose:
        print(f"  Strategy: {STRATEGY_LABELS[strategy]} | budget this round: {k}")

    if strategy == "vaal":
        from . import vaal
        return vaal.select(
            k, labeled_dataset, unlabeled_dataset, model.device,
            epochs=vaal_epochs, batch_size=batch_size,
            num_workers=num_workers, rng=rng, verbose=verbose,
        )

    unlabeled = extract_features(
        model, unlabeled_dataset, batch_size=batch_size,
        num_workers=num_workers, desc="  Features (unlabelled pool)",
    )

    if strategy == "coreset":
        from . import coreset
        labeled = extract_features(
            model, labeled_dataset, batch_size=batch_size,
            num_workers=num_workers, desc="  Features (labelled set)",
        )
        return coreset.select(
            k, unlabeled["backbone_features"], labeled["backbone_features"],
            rng=rng, verbose=verbose,
        )

    if strategy == "badge":
        from . import badge
        return badge.select(
            k, unlabeled["probs"], unlabeled["penultimate"],
            rng=rng, verbose=verbose,
        )

    if strategy == "clue":
        from . import clue
        return clue.select(
            k, unlabeled["penultimate"], unlabeled["probs"],
            rng=rng, verbose=verbose,
        )

    raise AssertionError(f"unhandled strategy {strategy}")  # pragma: no cover


__all__ = [
    "STRATEGIES", "STRATEGY_LABELS", "select_batch",
    "load_matched_budgets", "reference_experiment_id", "extract_features",
]
