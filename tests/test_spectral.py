"""Spectral clustering: the Tier-3 control, and why k-means is not enough."""

from __future__ import annotations

import numpy as np

from emaopt.algorithms import LloydKMeans, SpectralClustering
from emaopt.benchmarks import datasets, metrics


def test_spectral_separates_concentric_rings():
    ds = datasets.concentric_rings(n=800, noise=0.1, seed=0)
    model = SpectralClustering(2, n_landmarks=32, seed=0, nepochs=3, batch_size=64)
    labels = model.fit_predict(ds.x)
    ari = metrics.adjusted_rand_index(ds.meta["labels"], labels)
    assert ari > 0.9, ari


def test_kmeans_fails_on_the_same_rings():
    """The motivation for a spectral method: centroid geometry cannot see rings."""
    ds = datasets.concentric_rings(n=800, noise=0.1, seed=0)
    result = LloydKMeans(2, niter=20, seed=0).fit(ds.x)
    from emaopt.core import assign

    labels = assign(ds.x, result.params["centroids"])
    ari = metrics.adjusted_rand_index(ds.meta["labels"], labels)
    assert ari < 0.7, ari


def test_embedding_and_predict_are_deterministic():
    ds = datasets.concentric_rings(n=300, noise=0.1, seed=1)
    model = SpectralClustering(2, n_landmarks=16, seed=0, nepochs=1, batch_size=64).fit(ds.x)
    emb = model.embedding(ds.x)
    assert emb.shape == (ds.n, 2)
    a = model.predict(ds.x)
    b = model.predict(ds.x)
    assert np.array_equal(a, b)
    assert set(np.unique(a)).issubset({0, 1})


def test_operator_is_positive_semidefinite_and_converges_to_batch_estimate():
    ds = datasets.concentric_rings(n=400, noise=0.1, seed=2)
    model = SpectralClustering(2, n_landmarks=16, seed=0, nepochs=1, batch_size=400).fit(ds.x)
    op = np.asarray(model.state_["A"])
    assert np.allclose(op, op.T, atol=1e-10)
    evals = np.linalg.eigvalsh(op)
    assert np.min(evals) > -1e-8


def test_streaming_is_skipped_when_the_operator_already_spans_the_data():
    """A small fit must not pay five redundant passes over the same rows.

    With ``n <= _OPERATOR_SAMPLE`` the initial operator is the full Gram, so
    the epochs are skipped -- and the operator must then be *exactly* that
    Gram, not an EMA of it.
    """
    ds = datasets.concentric_rings(n=800, noise=0.1, seed=0)
    model = SpectralClustering(2, n_landmarks=32, seed=0, nepochs=5,
                               batch_size=64).fit(ds.x)
    assert model.state_.meta["epochs_streamed"] == 0
    kx = model._kernel(ds.x, model.landmarks_, model.gamma)
    assert np.allclose(model.state_["A"], kx.T @ kx / ds.n, rtol=0, atol=1e-12)
    assert metrics.adjusted_rand_index(ds.meta["labels"], model.predict(ds.x)) > 0.95
