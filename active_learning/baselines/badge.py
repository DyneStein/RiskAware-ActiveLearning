"""
BADGE — Batch Active learning by Diverse Gradient Embeddings.

    Ash, Zhang, Krishnamurthy, Langford & Agarwal,
    "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower
    Bounds", ICLR 2020.

THE IDEA IN ONE PARAGRAPH
-------------------------
Uncertainty methods pick the k most confused images, and often pick k
near-duplicates of the same confusing case. Diversity methods pick a
spread of images, including easy ones the model already handles. BADGE
gets both at once with one trick: pretend the model's own top guess is the
true label, and compute the gradient that would produce. That single
vector carries two things at the same time —

  * its LENGTH is large when the model is unsure (a confident, correct-
    looking prediction produces almost no gradient, because there is
    nothing to learn), and
  * its DIRECTION says *which way* the image would push the weights, so
    two images that would teach the model the same lesson point the same
    way.

Picking a spread-out set of these vectors therefore selects images that
are individually informative and collectively non-redundant, without ever
tuning a trade-off parameter between the two.

THE MATHS, CONCRETELY
---------------------
For image x, with softmax p ∈ ℝ⁷, top guess ŷ = argmax p, and h the
activation entering the final linear layer, the gradient of the
cross-entropy loss with respect to that layer's weights is the outer
product

    g_x = (p − e_ŷ) hᵀ ,    flattened to length 7 × 256 = 1792

where e_ŷ is the one-hot vector for the guess. Then run k-means++ *seeding*
on {g_x} — only the seeding step, which is a well-known procedure for
picking k mutually distant, high-magnitude points — and label the k images
it chooses.

WHY THIS IS THE BASELINE THAT MATTERS MOST
------------------------------------------
BADGE is the reference point reviewers in active learning expect. It is
strong, it needs no hyperparameter tuning, and it has held up across
years of follow-up work. Beating it on accuracy is not the goal here and
may well not happen — our claim is about clinical safety, an axis BADGE
has no representation of at all. It optimises expected information gain;
nothing in the gradient embedding knows that a melanoma is worse to miss
than a benign nevus.
"""

import numpy as np


def gradient_embeddings(probs, penultimate):
    """
    Build the BADGE gradient embedding for every image.

    Parameters
    ----------
    probs : np.ndarray, shape (N, C)
        Classification softmax probabilities.
    penultimate : np.ndarray, shape (N, H)
        Activation entering the final linear layer (H = 256 here).

    Returns
    -------
    np.ndarray, shape (N, C*H), float32
        Row i is vec((p_i − e_ŷᵢ) h_iᵀ).
    """
    probs = np.asarray(probs, dtype=np.float32)
    penultimate = np.asarray(penultimate, dtype=np.float32)
    n, c = probs.shape
    h = penultimate.shape[1]

    # (p − e_ŷ): the residual against the model's own hard guess.
    residual = probs.copy()
    residual[np.arange(n), probs.argmax(axis=1)] -= 1.0

    # Outer product per row, flattened. einsum avoids building an
    # intermediate (N, C, H) list in Python.
    emb = np.einsum("nc,nh->nch", residual, penultimate)
    return emb.reshape(n, c * h)


def kmeans_plusplus_seed(x, k, rng):
    """
    k-means++ seeding: choose k points, each far from those already
    chosen.

    Standard procedure — first centre uniformly at random, then each
    subsequent centre sampled with probability proportional to its squared
    distance from the nearest existing centre. The squared weighting is
    what makes BADGE prefer long gradient vectors (uncertain images)
    while still spreading out in direction.

    Only the seeding is used. Running k-means to convergence would move
    the centres to cluster means, which are not actual images and so
    cannot be labelled.
    """
    n = len(x)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    # Squared norms cached once; used for the |a-b|^2 expansion below.
    x_sq = np.einsum("ij,ij->i", x, x)

    first = int(rng.integers(n))
    centres = [first]
    closest_sq = _sq_dist_to_point(x, x_sq, x[first])

    for _ in range(1, k):
        total = float(closest_sq.sum())
        if not np.isfinite(total) or total <= 0:
            # Every remaining point coincides with a chosen centre — can
            # happen late in a round when the pool is nearly exhausted or
            # the model has collapsed. Fall back to uniform sampling over
            # whatever has not been picked yet.
            remaining = np.setdiff1d(np.arange(n), np.array(centres))
            if not len(remaining):
                break
            nxt = int(rng.choice(remaining))
        else:
            probabilities = closest_sq / total
            nxt = int(rng.choice(n, p=probabilities))

        centres.append(nxt)
        np.minimum(closest_sq, _sq_dist_to_point(x, x_sq, x[nxt]),
                   out=closest_sq)
        closest_sq[nxt] = 0.0

    return np.array(centres, dtype=int)


def _sq_dist_to_point(x, x_sq, point):
    """Squared Euclidean distance from every row of x to one point."""
    d2 = x_sq - 2.0 * (x @ point) + float(point @ point)
    return np.maximum(d2, 0.0)


def select(k, probs, penultimate, rng=None, verbose=True):
    """
    Select k images by k-means++ seeding over gradient embeddings.

    Parameters
    ----------
    k : int
        Number of images to select.
    probs : np.ndarray, shape (N, C)
    penultimate : np.ndarray, shape (N, H)
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray of int — indices into the unlabelled pool.
    """
    rng = rng if rng is not None else np.random.default_rng(42)
    n = len(probs)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    emb = gradient_embeddings(probs, penultimate)
    chosen = kmeans_plusplus_seed(emb, k, rng)

    if verbose:
        norms = np.linalg.norm(emb, axis=1)
        print(f"    BADGE: selected {len(chosen)} of {n}; "
              f"mean gradient norm chosen={norms[chosen].mean():.4f} "
              f"vs pool mean={norms.mean():.4f} "
              f"(higher means it favoured uncertain images, as intended)")
    return chosen
