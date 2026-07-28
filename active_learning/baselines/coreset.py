"""
CoreSet — greedy k-centre selection.

    Sener & Savarese, "Active Learning for Convolutional Neural Networks:
    A Core-Set Approach", ICLR 2018.

THE IDEA IN ONE PARAGRAPH
-------------------------
Forget uncertainty entirely. Ask instead: if the model only ever sees the
labelled images, which unlabelled images are *least well represented* by
what it has already seen? Place every image as a point in feature space.
The labelled set covers a region around each of its points. Repeatedly
pick the unlabelled point that is furthest from anything already labelled,
label it, and the covered region grows. After k picks, no unlabelled point
is far from a labelled one — the labelled set is a "core set" that
summarises the pool.

WHY IT IS A GOOD BASELINE FOR US
--------------------------------
It is the purest possible *diversity* method: it has no notion of
uncertainty and no notion of risk. If our dual-metric policy wins on
safety against CoreSet, it cannot be explained away as "you just picked
more diverse images".

WHY IT MIGHT LOSE ON SAFETY
---------------------------
Melanomas are a small minority of the pool. Coverage-driven selection
spends its budget proportionally across the whole feature space, so it
has no reason to over-sample the dangerous corner of it. That is exactly
the gap the risk head is designed to fill — and CoreSet makes the point
cleanly, because it is not a weak method, just an unaligned one.

COMPLEXITY
----------
O(k · N · D) with a running-minimum trick: each pick updates the distance
of every unlabelled point to the *newly added* centre only, rather than
recomputing distances to all labelled points. For k ≈ 500, N ≈ 7,600 and
D = 2048 this is a few seconds.
"""

import numpy as np

from .features import l2_normalise


def select(k, unlabeled_features, labeled_features, rng=None, verbose=True):
    """
    Greedy k-centre: repeatedly take the unlabelled point furthest from
    the labelled set.

    Parameters
    ----------
    k : int
        Number of images to select.
    unlabeled_features : np.ndarray, shape (N, D)
        Backbone features for the unlabelled pool.
    labeled_features : np.ndarray, shape (M, D)
        Backbone features for the current labelled set. If empty, the
        first centre is chosen at random and the algorithm proceeds from
        there.
    rng : np.random.Generator, optional
        Only used for the cold-start case above.

    Returns
    -------
    np.ndarray of int, shape (min(k, N),)
        Indices into `unlabeled_features`.
    """
    n = len(unlabeled_features)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    rng = rng if rng is not None else np.random.default_rng(42)
    unlabeled = l2_normalise(np.asarray(unlabeled_features, dtype=np.float32))

    # min_dist[i] = distance from unlabelled point i to the closest point
    # currently in the labelled/selected set. This single vector is the
    # whole state of the algorithm.
    if labeled_features is not None and len(labeled_features):
        labeled = l2_normalise(np.asarray(labeled_features, dtype=np.float32))
        min_dist = _min_distance_to_set(unlabeled, labeled)
    else:
        # No labelled data yet: seed with one random point so there is
        # something to measure distance from.
        min_dist = np.full(n, np.inf, dtype=np.float32)
        first = int(rng.integers(n))
        min_dist = np.minimum(
            min_dist, _distance_to_point(unlabeled, unlabeled[first])
        )

    selected = []
    for _ in range(k):
        # The point currently worst-covered by everything labelled so far.
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        # Adding it can only shrink distances, so a running minimum
        # against the new centre alone is exact — no need to revisit the
        # rest of the labelled set.
        min_dist = np.minimum(
            min_dist, _distance_to_point(unlabeled, unlabeled[idx])
        )
        # Guarantee it is never picked twice.
        min_dist[idx] = -np.inf

    if verbose:
        remaining = min_dist[min_dist > -np.inf]
        covered = remaining.max() if len(remaining) else 0.0
        print(f"    CoreSet: selected {len(selected)}; "
              f"largest remaining gap to a labelled point = {covered:.4f}")
    return np.array(selected, dtype=int)


def _distance_to_point(x, point):
    """Euclidean distance from every row of x to a single point."""
    diff = x - point[None, :]
    return np.sqrt(np.einsum("ij,ij->i", diff, diff))


def _min_distance_to_set(x, centres, block=512):
    """
    Distance from every row of x to its nearest row of `centres`.

    Blocked over centres so the (N × M) distance matrix is never
    materialised — with N ≈ 7,600 and M growing past 5,000 that would be
    ~150 MB of float32 for a number we only need the minimum of.
    """
    best = np.full(len(x), np.inf, dtype=np.float32)
    x_sq = np.einsum("ij,ij->i", x, x)
    for start in range(0, len(centres), block):
        chunk = centres[start:start + block]
        chunk_sq = np.einsum("ij,ij->i", chunk, chunk)
        # |a-b|^2 = |a|^2 - 2a·b + |b|^2
        d2 = x_sq[:, None] - 2.0 * (x @ chunk.T) + chunk_sq[None, :]
        np.minimum(best, np.sqrt(np.maximum(d2.min(axis=1), 0.0)), out=best)
    return best
