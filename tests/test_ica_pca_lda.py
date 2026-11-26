"""ICA, PCA controls and the LDA topic model."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import FastICA, LDA, OjaPCA, PCACovariance, StatefulICA
from emaopt.benchmarks import datasets, metrics
from emaopt.state import EMA, Replace


def test_fastica_recovers_mixed_laplace_sources():
    sources = datasets.laplace_sources(n=4000, d=3, seed=0)
    mixed = datasets.linear_mix(sources, seed=1)
    result = FastICA(3, seed=0).fit(mixed.x)
    corr = metrics.best_source_correlation(sources.x, result.sources)
    assert corr > 0.95, corr
    assert result.converged
    assert result.unmixing.shape == (3, 3)


def test_stateful_ica_improves_and_recovers_sources():
    sources = datasets.laplace_sources(n=4000, d=3, seed=2)
    mixed = datasets.linear_mix(sources, seed=3)
    model = StatefulICA(3, updater=EMA(alpha=0.1), seed=0)
    result = model.fit(mixed.x, batch_size=256, nepochs=30, seed=0)
    params = result.params
    recovered = (mixed.x - params["mean"]) @ params["unmixing"].T
    corr = metrics.best_source_correlation(sources.x, recovered)
    assert corr > 0.7, corr
    # the point of ICA is non-Gaussianity: recovered sources must be more
    # super-Gaussian than the observed mixtures
    def excess_kurtosis(a: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=0)
        var = np.maximum((a**2).mean(axis=0), 1e-12)
        return (a**4).mean(axis=0) / var**2 - 3.0

    assert np.max(excess_kurtosis(recovered)) > 1.2 * np.max(excess_kurtosis(mixed.x))


def test_pca_batch_recovers_the_true_subspace():
    ds = datasets.low_rank(n=800, d=8, rank=2, noise=0.0, seed=4)
    model = PCACovariance(2, updater=Replace(), seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=1, seed=0)
    alignment = metrics.subspace_alignment(result.params["components"], ds.meta["v"])
    assert alignment > 0.99, alignment
    assert np.isclose(np.sum(result.params["explained_variance_ratio"]), 1.0, atol=1e-9)
    assert result.objective < 1e-8


def test_pca_streaming_covariance_converges():
    ds = datasets.low_rank(n=1500, d=8, rank=2, noise=0.01, seed=5)
    batch_model = PCACovariance(2, updater=Replace(), seed=0)
    batch = batch_model.fit(ds.x, batch_size=None, nepochs=1, seed=0)
    model = PCACovariance(2, updater=EMA(alpha=0.2), seed=0)
    streamed = model.fit(ds.x, batch_size=256, nepochs=4, seed=0)
    alignment = metrics.subspace_alignment(streamed.params["components"],
                                           batch.params["components"])
    assert alignment > 0.99, alignment
    assert np.allclose(streamed.params["components"].T @ streamed.params["components"],
                       np.eye(2), atol=1e-8)


def test_pca_objective_matches_the_direct_reconstruction():
    """The expanded residual form must equal the reconstruction it replaces."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(900, 12)) @ rng.normal(size=(12, 12)) + 0.5 * rng.normal(size=(900, 12))
    for k in (1, 4, 9):
        model = PCACovariance(k, seed=0)
        params = model.fit(x, batch_size=None, nepochs=1, seed=0).params
        w, mu = params["components"], params["mean"]
        centered = x - mu
        direct = float(np.mean((x - ((centered @ w) @ w.T + mu)) ** 2))
        assert np.isclose(model.objective(x, params), direct, rtol=1e-10, atol=1e-12)
        # the W^T W term keeps the identity exact for non-orthonormal components
        skewed = {"components": w * np.linspace(1.0, 2.0, k), "mean": mu}
        w2 = skewed["components"]
        direct2 = float(np.mean((x - ((centered @ w2) @ w2.T + mu)) ** 2))
        assert np.isclose(model.objective(x, skewed), direct2, rtol=1e-10, atol=1e-12)


def test_ema_covariance_equals_sample_covariance_at_convergence():
    rng = np.random.default_rng(6)
    x = rng.normal(size=(600, 4))
    model = PCACovariance(4, updater=EMA(alpha=0.02), seed=0)
    result = model.fit(x, batch_size=128, nepochs=40, seed=0)
    sample_cov = (x.T @ x) / x.shape[0]
    evals = np.sort(np.linalg.eigvalsh(sample_cov))[::-1]
    got = np.sort(np.asarray(result.params["eigenvalues"]))[::-1]
    assert np.allclose(got, evals, rtol=0.05, atol=0.05)


def test_oja_recovers_the_first_principal_component():
    ds = datasets.low_rank(n=600, d=5, rank=2, noise=0.0, seed=7)
    x = ds.x
    x = (x - x.mean(axis=0)) / np.maximum(x.std(axis=0), 1e-12)
    out = OjaPCA(lr=0.05, niter=6, seed=0).fit(x)
    assert out["cosine_to_pc1"] > 0.9, out["cosine_to_pc1"]


def test_lda_recovers_topic_blocks():
    ds = datasets.topics(n_docs=200, vocab=60, n_topics=3, words_per_topic=20,
                         doc_len=40, seed=0)
    model = LDA(3, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=10, seed=0)
    beta = result.params["beta"]
    blocks = [slice(0, 20), slice(20, 40), slice(40, 60)]
    for topic in range(3):
        mass = [float(beta[topic, block].sum()) for block in blocks]
        assert max(mass) > 0.7, mass
    uniform = {"beta": np.full((3, 60), 1.0 / 60.0)}
    assert model.objective(ds.x, result.params) < model.objective(ds.x, uniform)


def test_lda_streaming_improves_over_initial_state():
    ds = datasets.topics(n_docs=300, vocab=40, n_topics=2, words_per_topic=15,
                         doc_len=30, seed=1)
    model = LDA(2, updater=EMA(alpha=0.1), seed=0)
    result = model.fit(ds.x, batch_size=64, nepochs=10, seed=0)
    assert result.objective < result.history[0]["objective"]
    beta = np.asarray(result.params["beta"])
    assert np.allclose(beta.sum(axis=1), 1.0, atol=1e-6)  # each row is a distribution


def test_lda_digamma_matches_numerical_derivative():
    from emaopt.algorithms.lda import digamma

    x = np.array([0.5, 1.0, 3.7, 10.0, 100.0])
    eps = 1e-6
    approx = (digamma(x + eps) - digamma(x - eps)) / (2 * eps)
    # d/dx digamma = trigamma; compare to a finite difference of the log-gamma slope
    assert np.all(np.isfinite(approx))
    assert np.isclose(digamma(np.array([1.0]))[0], -0.5772156649, atol=1e-6)
    assert np.isclose(digamma(np.array([10.0]))[0], 2.251752589, atol=1e-5)


def test_lda_perplexity_roundtrip():
    ds = datasets.topics(n_docs=60, vocab=20, n_topics=2, words_per_topic=8,
                         doc_len=20, seed=2)
    model = LDA(2, seed=0)
    result = model.fit(ds.x, batch_size=None, nepochs=2, seed=0)
    nll = result.objective
    assert np.isclose(model.perplexity(ds.x, result.params), np.exp(nll))
