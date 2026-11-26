"""Sparse diagonal+tied decomposition and the quantizer family."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import (
    HierarchicalCodebook,
    ProductQuantizer,
    ResidualQuantizer,
    SparseDiagTied,
    VectorQuantizer,
)
from emaopt.algorithms.kmeans import KMeans
from emaopt.benchmarks import datasets, metrics
from emaopt.state import EMA, Replace


def test_sparse_decomposition_recovers_the_atom_subspace():
    ds = datasets.sparse_decomposition(n=1200, d=16, k=8, sparsity=2, noise=0.15, seed=0)
    model = SparseDiagTied(8, sparsity=2, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=15, seed=0)
    alignment = metrics.subspace_alignment(result.params["atoms"].T,
                                           ds.meta["atoms"].T)
    assert alignment > 0.9, alignment
    # reconstruction must explain most of the energy (the data has no zero-mean
    # cluster structure, so the absolute error is large: compare to energy)
    energy = float(np.mean(np.sum(ds.x**2, axis=1)))
    assert result.objective < 0.35 * energy, (result.objective, energy)


def test_sparse_codes_are_actually_sparse():
    ds = datasets.sparse_decomposition(n=600, d=12, k=6, sparsity=1, noise=0.1, seed=1)
    model = SparseDiagTied(6, sparsity=1, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=10, seed=0)
    assert model.sparsity_level(ds.x, result.params) >= 1.0 - 1.0 / 6.0 - 1e-9
    codes = model.encode(ds.x, result.params)
    assert np.allclose(codes.sum(axis=1), 1.0)


def test_ema_sparse_decomposition_improves_over_iterations():
    ds = datasets.sparse_decomposition(n=800, d=12, k=6, sparsity=2, noise=0.2, seed=2)
    model = SparseDiagTied(6, sparsity=2, updater=EMA(alpha=0.05), seed=0)
    result = model.fit(ds.x, batch_size=128, nepochs=15, seed=0)
    assert result.objective < result.history[0]["objective"]
    assert np.all(np.asarray(result.params["sigma2"]) > 0.0)


def test_vector_quantizer_error_decreases_with_codebook_size():
    ds = datasets.blobs(n=800, k=4, d=2, sep=6.0, seed=3)
    small_model = VectorQuantizer(2, updater=EMA(alpha=0.2), seed=0)
    small = small_model.fit(ds.x, batch_size=128, nepochs=10, seed=0)
    large_model = VectorQuantizer(4, updater=EMA(alpha=0.2), seed=0)
    large = large_model.fit(ds.x, batch_size=128, nepochs=10, seed=0)
    assert large.objective < small.objective
    assert large.params["centroids"].shape == (4, 2)
    assert small.params["centroids"].shape == (2, 2)


def test_vector_quantizer_usage_reporting():
    ds = datasets.blobs(n=400, k=3, d=2, sep=6.0, seed=4)
    model = VectorQuantizer(3, updater=EMA(alpha=0.2), seed=0)
    result = model.fit(ds.x, batch_size=128, nepochs=8, seed=0)
    usage = model.code_usage(ds.x, result.params)
    assert set(usage) == {"dead", "imbalance", "perplexity"}
    assert usage["dead"] >= 0.0
    assert usage["imbalance"] >= 1.0
    assert 1.0 <= usage["perplexity"] <= 3.0 + 1e-9
    codes = model.encode(ds.x, result.params)
    recon = model.decode_codes(codes, result.params)
    assert recon.shape == ds.x.shape
    assert np.isclose(model.quantization_error(ds.x, result.params), result.objective)


def test_hierarchical_codebook_beats_its_coarse_stage():
    ds = datasets.blobs(n=1200, k=16, d=2, sep=6.0, seed=5)
    model = HierarchicalCodebook(n_coarse=4, n_fine=8, nepochs=6, seed=0).fit(ds.x)
    fine = model.quantization_error(ds.x)
    coarse = KMeans(4, seed=0).fit(ds.x, batch_size=256, nepochs=6, seed=0).objective
    assert fine < coarse
    assert model.bits_per_vector() == 5.0  # ceil(log2 4) + ceil(log2 8)
    codes = model.encode(ds.x)
    assert codes.shape == (ds.n, 2)
    assert model.decode(codes).shape == ds.x.shape


def test_product_quantizer_recovers_block_structure():
    rng = np.random.default_rng(6)
    c1 = rng.normal(scale=5.0, size=(4, 2))
    c2 = rng.normal(scale=5.0, size=(4, 2))
    n = 800
    a = rng.integers(0, 4, n)
    b = rng.integers(0, 4, n)
    x = np.hstack([c1[a], c2[b]]) + rng.normal(scale=0.01, size=(n, 4))
    pq = ProductQuantizer(d=4, m=2, ksub=4, nepochs=8, seed=0).fit(x)
    assert pq.quantization_error(x) < 0.01
    assert pq.bits_per_vector() == 4.0
    codes = pq.encode(x)
    assert codes.shape == (n, 2)
    assert pq.decode(codes).shape == x.shape


def test_residual_quantizer_improves_with_more_stages():
    ds = datasets.blobs(n=800, k=8, d=3, sep=5.0, seed=7)
    one = ResidualQuantizer(n_stages=1, ksub=8, nepochs=6, seed=0).fit(ds.x)
    three = ResidualQuantizer(n_stages=3, ksub=8, nepochs=6, seed=0).fit(ds.x)
    assert three.quantization_error(ds.x) < one.quantization_error(ds.x)
    assert three.bits_per_vector() > one.bits_per_vector()


def test_product_quantizer_requires_divisible_dimension():
    with pytest.raises(ValueError):
        ProductQuantizer(d=5, m=2, ksub=4)
