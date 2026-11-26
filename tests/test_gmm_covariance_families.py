"""Covariance families: what each decode can and cannot represent.

Two properties are pinned here because the reference comparison found them wrong:

* ``tied`` is a *shared full* covariance (off-diagonals included), not a shared
  diagonal vector;
* ``spherical`` is one variance *per component*, not one pooled over every
  component and dimension.

The third test covers the streaming full-covariance collapse: a centred scatter
in the initial state double-centred the covariance, which made the first
mini-batches hand every point to one near-singular component.
"""

from __future__ import annotations

import numpy as np

from emaopt.algorithms import GaussianMixture
from emaopt.state import EMA

SHARED_COV = np.array(
    [
        [1.0, 0.7, 0.2, 0.1],
        [0.7, 1.0, 0.3, 0.2],
        [0.2, 0.3, 1.0, 0.6],
        [0.1, 0.2, 0.6, 1.0],
    ]
)


def _shared_cov_data(n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(5, SHARED_COV.shape[0]))
    labels = rng.integers(0, 5, size=n)
    rows = [rng.multivariate_normal(means[j], SHARED_COV) for j in labels]
    return np.asarray(rows), labels


def test_tied_recovers_a_shared_full_covariance():
    x, _ = _shared_cov_data()
    tied = GaussianMixture(5, covariance="tied", seed=0).fit(
        x, batch_size=None, nepochs=25, seed=0
    )
    diag = GaussianMixture(5, covariance="diag", seed=0).fit(
        x, batch_size=None, nepochs=25, seed=0
    )
    cov = np.asarray(tied.params["covariances"])
    assert cov.shape == SHARED_COV.shape
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > 0.0
    off_diagonal = np.abs(cov - np.diag(np.diag(cov))).max()
    assert off_diagonal > 0.05, cov
    assert tied.objective < diag.objective


def test_tied_state_is_one_global_second_moment():
    model = GaussianMixture(4, covariance="tied")
    assert "gscatter" in model.state_keys
    assert "scatter" not in model.state_keys


def test_spherical_is_one_variance_per_component():
    x, _ = _shared_cov_data()
    result = GaussianMixture(5, covariance="spherical", seed=0).fit(
        x, batch_size=None, nepochs=15, seed=0
    )
    cov = np.asarray(result.params["covariances"])
    assert cov.shape == (5, SHARED_COV.shape[0])
    assert np.allclose(cov, cov[:, :1])          # constant within a component
    assert np.unique(np.round(cov[:, 0], 9)).size > 1  # distinct across components


def test_streaming_full_covariance_does_not_collapse():
    rng = np.random.default_rng(0)
    k, d, n = 6, 6, 6000
    means = rng.normal(size=(k, d)) * 2.5 / np.sqrt(d)
    labels = rng.integers(0, k, size=n)
    x = np.empty((n, d))
    for j in range(k):
        a = rng.normal(size=(d, d)) * 0.5
        index = labels == j
        x[index] = rng.multivariate_normal(
            means[j], a @ a.T + 0.2 * np.eye(d), size=int(index.sum())
        )

    batch = GaussianMixture(k, covariance="full", seed=0).fit(
        x, batch_size=None, nepochs=8, seed=0
    )
    streaming = GaussianMixture(k, covariance="full", updater=EMA(alpha=0.1), seed=0).fit(
        x, batch_size=256, nepochs=8, seed=0
    )
    weights = np.asarray(streaming.params["weights"])
    assert (weights > 1e-3).sum() >= k - 1, weights
    assert streaming.objective <= batch.objective * 1.05
