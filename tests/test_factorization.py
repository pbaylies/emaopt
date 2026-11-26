"""NMF and matrix factorization: recovery, monotonicity, non-negativity."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import ALSFactorization, NMF, SGDFactorization
from emaopt.algorithms.nmf import mu_nmf
from emaopt.benchmarks import datasets
from emaopt.state import EMA, Replace


def test_multiplicative_updates_decrease_the_objective():
    ds = datasets.nonneg_low_rank(n=300, d=10, rank=2, seed=0)
    _, _, losses = mu_nmf(ds.x, 2, niter=100, seed=0)
    assert np.all(np.diff(losses) <= 1e-12), losses[:5]
    assert losses[-1] < losses[0]


def test_reference_nmf_recovers_a_nonnegative_rank_two_matrix():
    ds = datasets.nonneg_low_rank(n=300, d=8, rank=2, noise=0.0, seed=1)
    u, v, losses = mu_nmf(ds.x, 2, niter=400, seed=0)
    rel = np.mean((ds.x - u @ v.T) ** 2) / np.mean(ds.x**2)
    assert rel < 0.01
    assert np.all(u >= 0.0) and np.all(v >= 0.0)


def test_persistent_nmf_recovers_the_same_matrix():
    ds = datasets.nonneg_low_rank(n=300, d=8, rank=2, noise=0.0, seed=2)
    model = NMF(2, updater=Replace(), seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=30, seed=0)
    rel = result.objective / np.mean(ds.x**2)
    assert rel < 0.05
    assert np.all(result.params["basis"] >= 0.0)


def test_ema_nmf_converges_on_minibatches():
    ds = datasets.nonneg_low_rank(n=600, d=8, rank=2, noise=0.0, seed=3)
    model = NMF(2, updater=EMA(alpha=0.05), seed=0)
    result = model.fit(ds.x, batch_size=128, nepochs=30, seed=0)
    rel = result.objective / np.mean(ds.x**2)
    assert rel < 0.1, rel
    assert result.objective < result.history[0]["objective"]


def test_nmf_rejects_negative_input_in_the_reference_path():
    x = np.array([[1.0, -1.0], [2.0, 3.0]])
    with pytest.raises(ValueError):
        mu_nmf(x, 1, niter=5)


def test_als_factorization_recovers_exact_low_rank():
    ds = datasets.low_rank(n=400, d=10, rank=3, noise=0.0, seed=4)
    model = ALSFactorization(3, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=15, seed=0)
    assert result.objective < 1e-3, result.objective


def test_ema_als_factorization_converges():
    ds = datasets.low_rank(n=800, d=10, rank=3, noise=0.01, seed=5)
    model = ALSFactorization(3, updater=EMA(alpha=0.1), seed=0)
    result = model.fit(ds.x, batch_size=256, nepochs=20, seed=0)
    assert result.objective < 0.05, result.objective


def test_sgd_factorization_is_finite_and_reasonable():
    ds = datasets.low_rank(n=200, d=6, rank=2, noise=0.0, seed=6)
    out = SGDFactorization(2, niter=30, seed=0).fit(ds.x)
    assert np.isfinite(out["rmse"])
    assert out["rmse"] < 0.2, out["rmse"]


def test_factorization_objective_is_scale_consistent():
    ds = datasets.low_rank(n=200, d=5, rank=2, noise=0.0, seed=7)
    model = ALSFactorization(2, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=10, seed=0)
    recon = model._coef(ds.x, result.params["basis"]) @ result.params["basis"].T
    assert np.isclose(result.objective,
                      float(np.sqrt(np.mean((ds.x - recon) ** 2))), rtol=1e-9)
