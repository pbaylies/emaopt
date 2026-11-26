"""The e-means drop-in surface.

Mirrors the published reference API::

    from emaopt import emeans
    result = emeans(x, k, nepochs=5)
    result.centroids, result.labels, result.objective

Everything here is a thin layer over :class:`~emaopt.algorithms.kmeans.EMeans`,
so the algorithm has exactly one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import FitConfig, FitResult, assign, fit
from .kmeans import EMeans, batch_size_from_law, alpha0_from_law


@dataclass
class EmeansResult:
    centroids: np.ndarray
    labels: np.ndarray
    objective: float
    history: list[dict[str, Any]]
    state: Any
    batch_size: int
    alpha0: float
    nepochs: int
    examples_processed: int
    updates: int
    wall_time: float
    seed: int
    extras: dict[str, Any] = field(default_factory=dict)

    def curve(self) -> tuple[np.ndarray, np.ndarray]:
        xs = np.array([h["examples_processed"] for h in self.history], dtype=float)
        ys = np.array([h["objective"] for h in self.history], dtype=float)
        return xs, ys

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": "emeans",
            "k": int(self.centroids.shape[0]),
            "d": int(self.centroids.shape[1]),
            "n": int(self.labels.shape[0]),
            "objective": float(self.objective),
            "batch_size": int(self.batch_size),
            "alpha0": float(self.alpha0),
            "nepochs": int(self.nepochs),
            "updates": int(self.updates),
            "examples_processed": int(self.examples_processed),
            "wall_time": float(self.wall_time),
            "seed": int(self.seed),
        }


def emeans(
    x: np.ndarray,
    k: int,
    *,
    nepochs: int = 5,
    batch_size: int | None = None,
    alpha0: float | None = None,
    batch_law: str = "auto",
    alpha_end_ratio: float = 0.1,
    unused_threshold: float | None = 0.1,
    revival: bool | None = None,
    finalize: bool = True,
    init: str = "kmeans++",
    init_centroids: np.ndarray | None = None,
    max_points_per_centroid: int | None = 256,
    seed: int = 0,
    track_memory: bool = False,
) -> EmeansResult:
    """Cluster ``x`` (n, d) into ``k`` clusters with e-means.

    ``batch_size`` and ``alpha0`` default to the fitted e-means scaling laws
    (``batch_law="auto"``, dimension-aware as in the reference).
    ``unused_threshold`` controls dead-cluster revival by proportional split;
    the default 0.1 is the reference release's resolved threshold, ``None``
    disables revival, and revival is off at ``d <= 2`` to match the reference.  If
    ``n > max_points_per_centroid * k``, training uses a subsample of that size
    (Faiss convention) and every point is labelled afterwards; pass
    ``max_points_per_centroid=None`` to train on all data.  ``init="auto"``
    (opt-in) uses the fast coreset-Lloyd seed when the data is wide enough,
    falling back to k-means++ on small or anisotropic data; it stays opt-in
    because its five-draw quality tail failed the pre-committed gates.
    ``init_centroids`` (k, d) starts the run from given centroids — maternal
    init, mirroring the reference ``train(x, init_centroids=...)`` hook — and
    overrides ``init``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be a 2-D array (n, d)")
    n, d = x.shape
    if k < 1:
        raise ValueError("k must be >= 1")
    if n == 0:
        raise ValueError("x is empty")
    if revival is None:
        revival = d > 2 and unused_threshold is not None
    if init_centroids is not None:
        init = "maternal"

    train = x
    if max_points_per_centroid is not None and n > int(max_points_per_centroid) * k:
        rng = np.random.default_rng(seed)
        budget = int(max_points_per_centroid) * k
        train = x[rng.choice(n, size=budget, replace=False)]

    model = EMeans(
        k,
        nepochs=nepochs,
        batch_size=batch_size,
        alpha0=alpha0,
        batch_law=batch_law,
        alpha_end_ratio=alpha_end_ratio,
        revival=revival,
        revival_threshold=(0.1 if unused_threshold is None else float(unused_threshold)),
        finalize=finalize,
        init=init,
        init_centroids=init_centroids,
        seed=seed,
    )
    cfg = FitConfig(
        batch_size=batch_size,
        nepochs=nepochs,
        seed=seed,
        track_memory=track_memory,
        eval_every_updates=None,
    )
    result: FitResult = fit(model, train, cfg, eval_data=train)
    params = result.params
    centroids = params["centroids"]
    labels = assign(x, centroids)
    objective = model.objective(x, params)
    return EmeansResult(
        centroids=centroids,
        labels=labels,
        objective=objective,
        history=result.history,
        state=result.extras.get("state"),
        batch_size=int(result.extras.get("batch_size", 0)),
        alpha0=float(result.extras.get("resolved_alpha0", 0.0)),
        nepochs=int(nepochs),
        examples_processed=int(result.examples_processed),
        updates=int(result.updates),
        wall_time=float(result.wall_time),
        seed=int(seed),
        extras={
            "batch_law": batch_law,
            "trained_on": int(train.shape[0]),
            "batches_per_epoch": int(result.extras.get("updates_per_epoch", 0)),
        },
    )
