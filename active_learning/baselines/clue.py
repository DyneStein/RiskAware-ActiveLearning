"""
CLUE — Clustering Uncertainty-weighted Embeddings.

    Prabhu, Chandrasekaran, Saenko & Hoffman,
    "Active Domain Adaptation via Clustering Uncertainty-weighted
    Embeddings", ICCV 2021.

THE IDEA IN ONE PARAGRAPH
-------------------------
Cluster the unlabelled pool into as many groups as there is budget, then
label one representative image from each group — that gives diversity for
free, because no two picks come from the same group. CLUE's contribution
is *where* the clusters are allowed to form: each image is weighted by how
uncertain the model is about it, so the clustering pulls its groups toward
the confusing regions of feature space and spends barely any of the budget
on regions the model already understands. The result is one representative
from each distinct *kind of confusion*, rather than one representative
from each distinct kind of image.

HOW IT DIFFERS FROM THE OTHER TWO
---------------------------------
  * CoreSet  — diversity only; uncertainty plays no part.
  * BADGE    — combines both implicitly, through the geometry of the
               gradient vector.
  * CLUE     — combines both explicitly: uncertainty is a scalar weight,
               diversity comes from the clustering, and the two are
               separate, legible steps.

Having all three ensures a comprehensive baseline comparison across fundamentally different mechanisms.

IMPLEMENTATION NOTE
-------------------
Clustering runs on the 256-dimensional activation entering the final
linear layer, not the 1,664–2,048-dimensional backbone output. That
follows the paper, which clusters the penultimate representation, and it
also keeps a k-means with several hundred clusters down to seconds rather
than minutes — this runs every round, of every experiment, for every
backbone.
"""

import numpy as np


def predictive_entropy(probs, eps=1e-12):
    """
    Shannon entropy of each row, H(p) = −Σ p log p.

    Zero when the model puts all its mass on one class; maximal
    (log 7 ≈ 1.946) when it is spread evenly over all seven. This is the
    same quantity as uncertainty/entropy.py, recomputed here so the
    baseline is self-contained and does not silently inherit our
    framework's scaling choices.
    """
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def select(k, embeddings, probs, rng=None, verbose=True,
           max_iter=30, n_init=1):
    """
    Entropy-weighted k-means; return the image closest to each centroid.

    Parameters
    ----------
    k : int
        Number of images to select — also the number of clusters, which
        is what makes "one per cluster" exhaust the budget exactly.
    embeddings : np.ndarray, shape (N, H)
        Penultimate-layer features for the unlabelled pool.
    probs : np.ndarray, shape (N, C)
        Classification softmax, used only to derive the weights.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray of int — indices into the unlabelled pool. May contain
    fewer than k entries if two centroids share their nearest image;
    duplicates are removed and the shortfall is topped up by highest
    remaining entropy.
    """
    from sklearn.cluster import KMeans

    rng = rng if rng is not None else np.random.default_rng(42)
    x = np.asarray(embeddings, dtype=np.float32)
    n = len(x)
    k = int(min(k, n))
    if k <= 0:
        return np.array([], dtype=int)

    weights = predictive_entropy(probs)
    # A pool where the model is perfectly confident everywhere would give
    # all-zero weights, which k-means rejects. Falling back to uniform
    # weights degrades CLUE to plain k-means clustering, which is the
    # right behaviour: with no uncertainty signal there is nothing to
    # weight by, and diversity alone is still a sensible choice.
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        weights = np.ones(n, dtype=np.float64)

    seed = int(rng.integers(0, 2 ** 31 - 1))
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=n_init,
                    max_iter=max_iter, random_state=seed)
    kmeans.fit(x, sample_weight=weights)

    # One real image per centroid — a centroid is an average and is not
    # itself an image, so it cannot be sent to the oracle.
    centroids = kmeans.cluster_centers_.astype(np.float32)
    chosen = _nearest_index_per_centroid(x, centroids)

    chosen = np.unique(chosen)
    if len(chosen) < k:
        # Two centroids landed on the same nearest image. Top up with the
        # most uncertain images not already picked, so the budget is spent
        # exactly and the cost-match against dual-metric stays honest.
        remaining = np.setdiff1d(np.arange(n), chosen)
        order = remaining[np.argsort(weights[remaining])[::-1]]
        chosen = np.concatenate([chosen, order[:k - len(chosen)]])

    if verbose:
        print(f"    CLUE: {k} clusters over {n} images; "
              f"mean entropy of picks={weights[chosen].mean():.4f} "
              f"vs pool mean={weights.mean():.4f}")
    return chosen.astype(int)


def _nearest_index_per_centroid(x, centroids, block=64):
    """
    Index of the closest row of x to each centroid.

    Blocked over centroids: with several hundred clusters and thousands of
    images the full distance matrix is large and only its per-column
    argmin is needed.
    """
    x_sq = np.einsum("ij,ij->i", x, x)
    out = np.empty(len(centroids), dtype=int)
    for start in range(0, len(centroids), block):
        chunk = centroids[start:start + block]
        chunk_sq = np.einsum("ij,ij->i", chunk, chunk)
        d2 = x_sq[:, None] - 2.0 * (x @ chunk.T) + chunk_sq[None, :]
        out[start:start + len(chunk)] = d2.argmin(axis=0)
    return out
