"""Strong baselines.

A candidate only becomes interesting after comparison with the established
method for its problem (PLAN hypothesis H6).  Each baseline is wired to the same
driver and the same budget as the research variants.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core import FitConfig, FitResult
from ..state import WeightedMean
from ..algorithms.kmeans import KMeans, LloydKMeans
from ..algorithms.nmf import mu_nmf
from ..algorithms.pca import PCACovariance
from ..algorithms.ica import fastica
from ..core import fit


def lloyd(data: np.ndarray, k: int, *, niter: int = 25, seed: int = 0) -> FitResult:
    """Batch Lloyd k-means (the default k-means in every vector library)."""
    return LloydKMeans(k, niter=niter, seed=seed).fit(data, niter=niter)


def minibatch_kmeans(
    data: np.ndarray,
    k: int,
    *,
    batch_size: int | None = 256,
    nepochs: int = 5,
    seed: int = 0,
    track_memory: bool = False,
) -> FitResult:
    """Sculley/MacQueen mini-batch k-means.

    Per-point MacQueen updates vectorised per cluster are exactly a
    batch-size-weighted cumulative mean of the cluster sums and counts, which is
    why this baseline is the ``WeightedMean`` updater rather than a separate
    implementation.
    """
    model = KMeans(k, updater=WeightedMean("batch_size"), seed=seed)
    cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                    track_memory=track_memory)
    return fit(model, np.asarray(data, dtype=np.float64), cfg)


def batch_em_gmm(data: np.ndarray, n_components: int, *, covariance: str = "diag",
                 niter: int = 50, seed: int = 0) -> FitResult:
    from ..algorithms.gmm import GaussianMixture

    model = GaussianMixture(n_components, covariance=covariance, seed=seed)
    return model.fit(data, batch_size=None, nepochs=niter, seed=seed)


def nmf_mu(data: np.ndarray, n_components: int, *, niter: int = 200, seed: int = 0
           ) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Lee & Seung multiplicative updates (reference NMF)."""
    return mu_nmf(data, n_components, niter=niter, seed=seed)


def fastica_baseline(data: np.ndarray, n_components: int | None = None, *, seed: int = 0):
    return fastica(data, n_components, seed=seed)


def pca_batch(data: np.ndarray, n_components: int, *, seed: int = 0) -> FitResult:
    model = PCACovariance(n_components, seed=seed)
    return model.fit(data, batch_size=None, nepochs=1, seed=seed)


BASELINES: dict[str, Callable[..., Any]] = {
    "lloyd": lloyd,
    "minibatch_kmeans": minibatch_kmeans,
    "batch_em_gmm": batch_em_gmm,
    "nmf_mu": nmf_mu,
    "fastica": fastica_baseline,
    "pca_batch": pca_batch,
}


def get_baseline(name: str) -> Callable[..., Any]:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name]
