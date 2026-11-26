"""The PLAN's correctness contract (section 14), exercised across families.

Degenerate clusters, empty assignments, singular covariance, zero-variance
dimensions, extreme batch sizes, repeated observations, numerical extremes,
multiple seeds.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import (
    EMeans,
    GaussianMixture,
    LloydKMeans,
    NMF,
    PCACovariance,
    ProductQuantizer,
    SparseDiagTied,
    emeans,
)
from emaopt.benchmarks import datasets
from emaopt.core import assign
from emaopt.state import EMA


def test_single_point_datasets_are_accepted():
    x = np.array([[1.5, -2.0]])
    result = emeans(x, 3, nepochs=2, seed=0)
    assert np.all(np.isfinite(result.centroids))
    assert np.isfinite(result.objective)


def test_k_equals_one_is_the_mean():
    x = np.random.default_rng(0).normal(size=(120, 3))
    result = emeans(x, 1, nepochs=2, seed=0)
    assert np.allclose(result.centroids[0], x.mean(axis=0), atol=1e-2)


def test_repeated_observations_are_idempotent():
    x = np.repeat(np.array([[3.0, 4.0]]), 40, axis=0)
    result = emeans(x, 2, nepochs=2, seed=0)
    assert np.allclose(result.centroids, [3.0, 4.0], atol=1e-9)


def test_zero_variance_dimension_is_finite_everywhere():
    rng = np.random.default_rng(1)
    x = np.column_stack([rng.normal(size=200), np.zeros(200), np.full(200, 7.0)])
    gmm = GaussianMixture(2, covariance="diag", seed=0)
    gmm_result = gmm.fit(x, batch_size=None, nepochs=3, seed=0)
    kmeans_result = emeans(x, 2, nepochs=2, seed=0)
    assert np.isfinite(gmm_result.objective)
    assert np.isfinite(kmeans_result.objective)


def test_numerical_extremes_do_not_produce_nan():
    rng = np.random.default_rng(2)
    for scale in (1e-6, 1e6):
        x = rng.normal(size=(100, 2)) * scale
        result = emeans(x, 3, nepochs=2, seed=0)
        assert np.all(np.isfinite(result.centroids))
        assert np.isfinite(result.objective)
        model = PCACovariance(1, seed=0)
        pca = model.fit(x, batch_size=None, nepochs=1, seed=0)
        assert np.isfinite(pca.objective)


def test_extreme_batch_sizes_agree_on_a_separable_problem():
    ds = datasets.blobs(n=300, k=3, d=2, sep=9.0, seed=3)
    full = EMeans(3, nepochs=3, batch_size=300, seed=0).fit(ds.x, batch_size=300,
                                                            nepochs=3, seed=0)
    single = EMeans(3, nepochs=3, batch_size=1, seed=0).fit(ds.x, batch_size=1,
                                                            nepochs=3, seed=0)
    assert np.isfinite(full.objective) and np.isfinite(single.objective)
    labels_full = assign(ds.x, full.params["centroids"])
    labels_single = assign(ds.x, single.params["centroids"])
    from emaopt.benchmarks.metrics import adjusted_rand_index

    assert adjusted_rand_index(labels_full, labels_single) > 0.8


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multiple_seeds_stay_finite(seed):
    ds = datasets.blobs(n=200, k=4, d=2, sep=5.0, seed=seed)
    result = emeans(ds.x, 4, nepochs=3, seed=seed)
    assert np.isfinite(result.objective)
    assert result.centroids.shape == (4, 2)


def test_lloyd_best_so_far_is_monotone():
    ds = datasets.blobs(n=300, k=3, d=2, sep=6.0, seed=4)
    result = LloydKMeans(3, niter=8, seed=0).fit(ds.x, niter=8)
    ys = np.array([h["objective"] for h in result.history])
    assert np.all(np.diff(ys) <= 1e-9)


def test_ema_state_does_not_depend_on_hash_ordering():
    """Same seed, same data, same run: the stream must be reproducible."""
    ds = datasets.blobs(n=200, k=3, d=2, sep=6.0, seed=5)
    a = EMeans(3, nepochs=3, seed=7).fit(ds.x, nepochs=3, seed=7)
    b = EMeans(3, nepochs=3, seed=7).fit(ds.x, nepochs=3, seed=7)
    assert np.allclose(a.params["centroids"], b.params["centroids"])


def test_quantizer_handles_more_codewords_than_points():
    x = np.random.default_rng(6).normal(size=(5, 4))
    pq = ProductQuantizer(d=4, m=2, ksub=8, nepochs=2, seed=0).fit(x)
    assert pq.quantization_error(x) >= 0.0
    assert np.isfinite(pq.quantization_error(x))


def test_sparse_decomposition_with_one_component():
    ds = datasets.blobs(n=150, k=1, d=3, sep=0.0, seed=7)
    model = SparseDiagTied(1, sparsity=1, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=2, seed=0)
    assert np.isfinite(result.objective)


def test_nmf_with_one_component_is_finite():
    ds = datasets.nonneg_low_rank(n=100, d=4, rank=1, seed=8)
    model = NMF(1, updater=EMA(alpha=0.1), seed=0)
    result = model.fit(ds.x, batch_size=32, nepochs=5, seed=0)
    assert np.isfinite(result.objective)
    assert np.all(result.params["basis"] >= 0.0)


def test_empty_batch_is_impossible_by_construction():
    """The sampler never emits an empty batch, even when drop_last is off."""
    from emaopt.core import FitConfig

    ds = datasets.blobs(n=17, k=2, d=2, sep=5.0, seed=9)
    model = EMeans(2, nepochs=1, seed=0)
    result = model.fit(ds.x, batch_size=5, nepochs=1, seed=0)
    assert result.updates == 3  # floor(17 / 5)
    cfg = FitConfig(batch_size=5, nepochs=1, drop_last=False, seed=0)
    from emaopt.core import fit

    result = fit(model, ds.x, cfg)
    assert result.updates == 4  # ceil(17 / 5)
