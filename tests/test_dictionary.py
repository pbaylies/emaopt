"""Dictionary learning: sparse codes, atom recovery, batch and streamed arms."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.dictionary import DictionaryLearner, ista, soft_threshold
from emaopt.benchmarks import datasets, metrics
from emaopt.state import EMA, Replace


def test_soft_threshold_and_ista_shrink_to_zero():
    assert np.allclose(soft_threshold(np.array([0.4, -0.4, 1.0]), 0.5), [0.0, 0.0, 0.5])
    atoms = np.eye(3)
    x = np.array([[2.0, 0.0, 0.0]])
    z = ista(x, atoms, lam=0.5, niter=50)
    assert np.allclose(z, [[1.5, 0.0, 0.0]], atol=1e-6)


def test_batch_dictionary_learning_recovers_atoms():
    ds = datasets.sparse_codes(n=1200, d=24, m=12, sparsity=2, noise=0.01, seed=0)
    model = DictionaryLearner(12, lam=0.02, updater=Replace(), seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=12, seed=0)
    alignment = metrics.subspace_alignment(result.params["atoms"],
                                           ds.meta["dictionary"])
    assert alignment > 0.7, alignment
    assert result.objective < 0.05, result.objective


def test_streamed_dictionary_learning_improves():
    ds = datasets.sparse_codes(n=900, d=20, m=10, sparsity=2, noise=0.02, seed=1)
    model = DictionaryLearner(10, lam=0.05, updater=EMA(alpha=0.1), seed=0)
    result = model.fit(ds.x, batch_size=128, nepochs=15, seed=0)
    assert result.objective < result.history[0]["objective"]
    assert result.objective < 0.2


def test_codes_are_sparse_and_reconstruct():
    ds = datasets.sparse_codes(n=400, d=16, m=8, sparsity=1, noise=0.0, seed=2)
    model = DictionaryLearner(8, lam=0.2, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=8, seed=0)
    codes = model.codes(ds.x, result.params)
    nnz = np.mean(codes != 0.0)
    assert nnz < 0.5, nnz
    recon = codes @ result.params["atoms"].T
    assert metrics.relative_error(ds.x, recon) < 0.2


def test_dictionary_atoms_stay_normalised():
    ds = datasets.sparse_codes(n=300, d=12, m=6, sparsity=2, noise=0.05, seed=3)
    model = DictionaryLearner(6, lam=0.05, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=6, seed=0)
    norms = np.linalg.norm(result.params["atoms"], axis=0)
    assert np.allclose(norms, 1.0, atol=1e-8)


def test_more_sparsity_penalty_gives_sparser_codes():
    ds = datasets.sparse_codes(n=300, d=12, m=6, sparsity=3, noise=0.02, seed=4)
    weak = DictionaryLearner(6, lam=0.001, seed=0).fit(ds.x, batch_size=None,
                                                       nepochs=5, seed=0)
    strong = DictionaryLearner(6, lam=0.3, seed=0).fit(ds.x, batch_size=None,
                                                       nepochs=5, seed=0)
    weak_model = DictionaryLearner(6, lam=0.001, seed=0)
    strong_model = DictionaryLearner(6, lam=0.3, seed=0)
    weak_nnz = np.mean(weak_model.codes(ds.x, weak.params) != 0.0)
    strong_nnz = np.mean(strong_model.codes(ds.x, strong.params) != 0.0)
    assert strong_nnz <= weak_nnz
