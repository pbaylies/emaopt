"""Gaussian mixtures: EM monotonicity, parameter recovery, degenerate covariances."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.gmm import GaussianMixture, em_gmm
from emaopt.benchmarks import datasets, metrics
from emaopt.state import EMA, Replace

from conftest import greedy_match_error


def test_batch_em_does_not_increase_nll():
    ds = datasets.blobs(n=600, k=3, d=2, sep=6.0, seed=0)
    model = GaussianMixture(3, covariance="diag", seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=12, seed=0)
    nlls = np.array([h["objective"] for h in result.history])
    assert np.all(np.diff(nlls) <= 1e-6), nlls


@pytest.mark.parametrize("covariance", ["spherical", "diag", "tied", "full"])
def test_parameter_recovery_by_covariance_family(covariance):
    ds = datasets.gaussian_mixture(n=2500, k=3, d=3, covariance=covariance,
                                   separation=8.0, seed=1)
    result = em_gmm(ds.x, 3, covariance=covariance, niter=25, seed=0)
    error = greedy_match_error(ds.meta["means"], result.params["means"])
    assert error < 1.2, f"{covariance} mean error {error}"
    assert np.isclose(result.params["weights"].sum(), 1.0)
    assert np.all(result.params["weights"] > 0.0)


def test_gmm_kmeans_seed_is_configurable(monkeypatch):
    from emaopt.algorithms import kmeans

    rng = np.random.default_rng(9)
    x = rng.normal(size=(5000, 4))
    seen: dict[str, int] = {}
    original = kmeans.LloydKMeans.fit

    def spy(self, data, **kwargs):
        seen["rows"] = data.shape[0]
        return original(self, data, **kwargs)

    monkeypatch.setattr(kmeans.LloydKMeans, "fit", spy)
    model = GaussianMixture(3, covariance="diag", init_sample=3000,
                            n_init=2, seed=0)
    state = model.initialize(x)
    assert seen["rows"] == 3000
    assert np.all(np.isfinite(model.decode(state)["means"]))


def test_responsibilities_are_a_distribution():
    ds = datasets.gaussian_mixture(n=300, k=3, d=2, separation=6.0, seed=2)
    model = GaussianMixture(3, covariance="diag", seed=0)
    model.fit(ds.x, batch_size=None, nepochs=3, seed=0)
    params = model.decode(model.initialize(ds.x))
    resp = model.responsibilities(ds.x, params)
    assert np.allclose(resp.sum(axis=1), 1.0)
    assert np.all(resp >= 0.0)


def test_singular_covariance_stays_finite():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(400, 2))
    x = np.column_stack([base, np.zeros(400)])  # rank-2 data in 3-D
    model = GaussianMixture(3, covariance="full", reg_covar=1e-6, seed=0)
    result = model.fit(x, batch_size=None, nepochs=5, seed=0)
    assert np.isfinite(result.objective)
    assert np.all(np.isfinite(result.params["means"]))
    assert np.all(np.isfinite(result.params["covariances"]))


def test_zero_variance_dimension_is_regularised():
    x = np.column_stack([np.random.default_rng(4).normal(size=200), np.full(200, 3.0)])
    model = GaussianMixture(2, covariance="diag", reg_covar=1e-6, seed=0)
    result = model.fit(x, batch_size=None, nepochs=5, seed=0)
    assert np.isfinite(result.objective)
    assert np.all(np.asarray(result.params["covariances"]) > 0.0)


def test_ema_gmm_converges_on_minibatches():
    ds = datasets.gaussian_mixture(n=2000, k=3, d=2, separation=9.0, seed=5)
    model = GaussianMixture(3, covariance="diag", updater=EMA(alpha=0.05), seed=0)
    result = model.fit(ds.x, batch_size=128, nepochs=25, seed=0)
    labels = np.argmax(model.responsibilities(ds.x, result.params), axis=1)
    assert metrics.adjusted_rand_index(ds.meta["labels"], labels) > 0.85
    # the streaming arm converges to a plateau: it must not diverge, and its best
    # point must improve on the first epoch (a per-step monotone claim would be
    # false for a noisy mini-batch estimator)
    objectives = np.array([h["objective"] for h in result.history])
    assert objectives.min() <= objectives[0]
    assert objectives[-1] <= objectives[0] * 1.01


def test_batch_size_one_still_produces_a_usable_model():
    ds = datasets.gaussian_mixture(n=150, k=2, d=2, separation=8.0, seed=6)
    model = GaussianMixture(2, covariance="spherical", updater=EMA(alpha=0.2), seed=0)
    result = model.fit(ds.x, batch_size=1, nepochs=5, seed=0)
    assert np.isfinite(result.objective)
    assert np.all(np.asarray(result.params["covariances"]) > 0.0)


def test_single_component_reduces_to_one_gaussian():
    rng = np.random.default_rng(7)
    x = rng.normal(loc=2.0, scale=1.5, size=(500, 2))
    result = em_gmm(x, 1, covariance="full", niter=5, seed=0)
    assert np.allclose(result.params["means"][0], x.mean(axis=0), atol=0.2)
    assert result.params["weights"][0] == pytest.approx(1.0)


def test_empty_component_falls_back_to_a_finite_mean():
    x = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0]]), 10, axis=0)
    model = GaussianMixture(4, covariance="diag", seed=0)
    result = model.fit(x, batch_size=None, nepochs=2, seed=0)
    assert np.all(np.isfinite(result.params["means"]))
    assert np.isfinite(result.objective)
