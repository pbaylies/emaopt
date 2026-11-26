"""PDDP / kM-PDDP seeding for centroid clustering.

Principal Direction Divisive Partitioning (Boley, 1998) builds a binary
hierarchy by projecting each node onto its first principal direction and
splitting at the projected mean.  kM-PDDP (Zeimpekis & Gallopoulos, 2008)
replaces that fixed split with two-means on the one-dimensional projection:
the cut is placed between the two projected centroids.  The leaf with the
largest total scatter is split next.

For a seed we only need the leaf centroids, not the tree.  The implementation
therefore keeps node indices, centroids, scatters and a warm direction, and
returns the leaf centroids.  It is opt-in through ``KMeans(init="pddp")``;
the default k-means++ behaviour is unchanged.

The exact direction path forms the node covariance and uses ``eigh``.  The
power path is cheaper for wide rows but is deliberately opt-in: on nearly
isotropic spectra its direction converges slowly, so it should be paired with
a residual check/fallback in a production path.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-12


def _leading_direction(
    data: np.ndarray,
    mean: np.ndarray,
    *,
    direction: str,
    power_iters: int,
    warm: np.ndarray | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """First principal direction of ``data`` around ``mean``.

    ``direction="exact"`` is the covariance/eigh path.  ``direction="power"``
    applies power iteration directly to the centered rows, which avoids the
    ``d x d`` covariance and costs ``O(n d)`` per iteration.  The power path
    accepts a parent direction as a warm start.
    """
    if direction not in ("exact", "power"):
        raise ValueError(
            f"unknown direction {direction!r}; available: exact, power"
        )
    centered = data - mean
    d = centered.shape[1]
    if direction == "exact":
        cov = centered.T @ centered
        _evals, evecs = np.linalg.eigh(cov)
        return evecs[:, -1]

    if warm is not None and np.asarray(warm).shape == (d,):
        vector = np.asarray(warm, dtype=np.float64).copy()
        norm = float(np.linalg.norm(vector))
        if norm > _EPS:
            vector /= norm
        else:
            vector = rng.normal(size=d)
    else:
        vector = rng.normal(size=d)
    for _ in range(max(1, int(power_iters))):
        updated = centered.T @ (centered @ vector)
        norm = float(np.linalg.norm(updated))
        if norm <= _EPS:
            break
        vector = updated / norm
    return vector


def _one_dimensional_split(projection: np.ndarray) -> float:
    """Optimal two-means cut for a one-dimensional sample.

    Sorting turns the two-cluster SSE into a prefix-sum scan.  The returned
    threshold is the midpoint between the two rows adjacent to the best cut.
    """
    values = np.sort(np.asarray(projection, dtype=np.float64).ravel())
    n = values.size
    if n < 2:
        return float(values[0]) if n else 0.0
    prefix = np.cumsum(values)
    prefix_sq = np.cumsum(values * values)
    left_count = np.arange(1, n, dtype=np.float64)
    right_count = n - left_count
    left = prefix_sq[:-1] - prefix[:-1] ** 2 / left_count
    right = (
        (prefix_sq[-1] - prefix_sq[:-1])
        - (prefix[-1] - prefix[:-1]) ** 2 / right_count
    )
    cut = int(np.argmin(left + right))
    return float(0.5 * (values[cut] + values[cut + 1]))


def pddp_seed(
    data: np.ndarray,
    k: int,
    *,
    steering: bool = True,
    direction: str = "exact",
    power_iters: int = 8,
    min_size: int = 1,
    seed: int = 0,
) -> np.ndarray:
    """Return ``k`` PDDP/kM-PDDP leaf centroids for k-means initialization.

    Parameters
    ----------
    data:
        ``(n, d)`` rows.
    k:
        Number of centroids.  If the data cannot be split further (for
        example, all rows are identical), the remaining slots are filled with
        data rows deterministically.
    steering:
        If true, use exact one-dimensional two-means for the cut (kM-PDDP).
        If false, split at the projection mean, the original PDDP rule.
    direction:
        ``"exact"`` (covariance/eigh) or ``"power"`` (row power iteration).
    power_iters:
        Iterations for ``direction="power"``; ignored by the exact path.
    min_size:
        Minimum child size.  A node below ``2 * min_size`` is not split.
    seed:
        Seed for the power-iteration start and for padding.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array (n, d)")
    n, d = data.shape
    if n == 0:
        raise ValueError("data is empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return data.mean(axis=0, keepdims=True)
    if n == 1:
        return np.repeat(data, k, axis=0)

    rng = np.random.default_rng(int(seed))
    min_size = max(1, int(min_size))

    def make_node(indices: np.ndarray, warm: np.ndarray | None) -> dict:
        rows = data[indices]
        mean = rows.mean(axis=0)
        centered = rows - mean
        scatter = float(np.einsum("ij,ij->", centered, centered))
        return {
            "indices": indices,
            "mean": mean,
            "scatter": scatter,
            "warm": warm,
        }

    nodes = [make_node(np.arange(n), None)]
    while len(nodes) < k:
        candidates = [
            index for index, node in enumerate(nodes)
            if node["indices"].size >= 2 * min_size
            and node["scatter"] > _EPS
        ]
        if not candidates:
            break
        pick = max(candidates, key=lambda index: nodes[index]["scatter"])
        node = nodes[pick]
        indices = node["indices"]
        mean = node["mean"]
        rows = data[indices]
        vector = _leading_direction(
            rows, mean, direction=direction, power_iters=power_iters,
            warm=node["warm"], rng=rng,
        )
        projection = (rows - mean) @ vector
        if steering:
            threshold = _one_dimensional_split(projection)
        else:
            threshold = 0.0
        left_mask = projection <= threshold
        left_count = int(left_mask.sum())
        if left_count < min_size or indices.size - left_count < min_size:
            order = np.argsort(projection, kind="stable")
            half = indices.size // 2
            left_indices = indices[order[:half]]
            right_indices = indices[order[half:]]
        else:
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]
        if left_indices.size == 0 or right_indices.size == 0:
            node["scatter"] = 0.0
            continue
        nodes[pick:pick + 1] = [
            make_node(left_indices, vector),
            make_node(right_indices, vector),
        ]

    centroids = np.asarray([node["mean"] for node in nodes], dtype=np.float64)
    if centroids.shape[0] < k:
        extra = k - centroids.shape[0]
        indices = rng.choice(n, size=extra, replace=extra > n)
        centroids = np.vstack([centroids, data[indices]])
    return centroids


__all__ = ["pddp_seed"]
