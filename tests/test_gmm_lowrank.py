"""The lowrank family: does a sketched rank-r decode actually buy flexibility?

Four properties, each one a claim the wide-width table depends on:

1. on data generated from a diagonal-plus-rank-2 factor model, ``lowrank``
   scores better held-out likelihood than ``diag`` (it can represent the
   off-diagonal structure that ``diag`` cannot).  The held-out draw has to come
   from the *same* generator spec as the training draw -- redrawing the spec
   makes the test set unmodelable by anything and the comparison meaningless;
2. with ``rank == d`` the sketch is invertible, so the Nystroem decode *is* the
   full covariance -- the two fits must land on the same objective;
3. the streaming (EMA) arm does not collapse, which is the failure mode the full
   family had before its initialization was fixed;
4. its state is smaller than full covariance's at wide ``d``, which is the whole
   reason it exists.
"""

from __future__ import annotations

import numpy as np

from emaopt.algorithms import GaussianMixture
from emaopt.algorithms.gmm import covariance_parameter_count
from emaopt.core import logsumexp_rows
from emaopt.state import EMA


def _nll(model: GaussianMixture, params: dict, x: np.ndarray) -> float:
    logits = model.component_log_prob(x, params) + np.log(
        np.maximum(params["weights"], 1e-300)
    )[None, :]
    return float(-logsumexp_rows(logits).mean())


def _factor_spec(k: int, d: int, rank: int, seed: int = 0):
    """The generator: x = mu_j + L_j z + sqrt(psi_j) * noise, z ~ N(0, I_r)."""
    rng = np.random.default_rng(seed)
    means = rng.normal(size=(k, d)) * 3.0 / np.sqrt(d)
    # The loading has to dominate the diagonal noise, or ``diag`` is already the
    # right model and the test measures nothing: per-element variance 2 gives a
    # rank-r contribution of 2r per dimension against psi ~ 0.4.
    loadings = rng.normal(size=(k, d, rank)) * np.sqrt(2.0)
    psi = np.exp(rng.uniform(np.log(0.2), np.log(0.6), size=(k, d)))
    return means, loadings, psi


def _draw(spec, n: int, seed: int) -> np.ndarray:
    """A fresh sample from an existing spec -- the held-out set shares it."""
    means, loadings, psi = spec
    k, d = means.shape
    rank = loadings.shape[2]
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, k, size=n)
    z = rng.normal(size=(n, rank))
    x = means[labels] + np.einsum("nr,kdr->nkd", z, loadings)[np.arange(n), labels]
    return x + rng.normal(size=(n, d)) * np.sqrt(psi)[labels]


def test_lowrank_beats_diag_on_factor_data():
    k, d, rank = 4, 16, 2
    spec = _factor_spec(k, d, rank, seed=0)
    train = _draw(spec, n=4000, seed=10)
    test = _draw(spec, n=2000, seed=11)

    lowrank_model = GaussianMixture(k, covariance="lowrank", rank=rank, seed=0)
    lowrank = lowrank_model.fit(train, batch_size=None, nepochs=25, seed=0)
    diag_model = GaussianMixture(k, covariance="diag", seed=0)
    diag = diag_model.fit(train, batch_size=None, nepochs=25, seed=0)

    cov = np.asarray(lowrank.params["covariances"])
    lam = np.asarray(lowrank.params["loadings"])
    assert cov.shape == (k, d)
    assert lam.shape == (k, d, rank)
    assert np.all(cov > 0.0)
    # the loading is the part that carries the off-diagonal structure
    assert np.abs(lam).max() > 0.0
    lowrank_nll = _nll(lowrank_model, lowrank.params, test)
    diag_nll = _nll(diag_model, diag.params, test)
    # The margin is not a hair: on this spec it is ~8.5 nats/point, so a 2-nat
    # bar catches a decode that stops carrying the off-diagonal structure while
    # tolerating seed noise.
    assert lowrank_nll < diag_nll - 2.0, (lowrank_nll, diag_nll)


def test_rank_d_reproduces_full_covariance():
    rng = np.random.default_rng(3)
    k, d, n = 4, 6, 3000
    means = rng.normal(size=(k, d)) * 2.0 / np.sqrt(d)
    labels = rng.integers(0, k, size=n)
    x = np.empty((n, d))
    for j in range(k):
        a = rng.normal(size=(d, d)) * 0.5
        index = labels == j
        x[index] = rng.multivariate_normal(
            means[j], a @ a.T + 0.2 * np.eye(d), size=int(index.sum())
        )

    full = GaussianMixture(k, covariance="full", seed=0).fit(
        x, batch_size=None, nepochs=20, seed=0
    )
    lowrank = GaussianMixture(k, covariance="lowrank", rank=d, seed=0).fit(
        x, batch_size=None, nepochs=20, seed=0
    )
    assert np.asarray(lowrank.params["loadings"]).shape == (k, d, d)
    assert abs(lowrank.objective - full.objective) <= 0.02 * abs(full.objective), (
        lowrank.objective,
        full.objective,
    )


def test_streaming_lowrank_keeps_every_component_alive():
    train = _draw(_factor_spec(6, 8, 2, seed=2), n=4000, seed=10)
    model = GaussianMixture(6, covariance="lowrank", rank=2, seed=0, updater=EMA(alpha=0.1))
    result = model.fit(train, batch_size=128, nepochs=4, seed=0)
    weights = np.asarray(result.params["weights"])
    assert (weights > 1e-3).sum() >= 5, weights
    assert np.isfinite(result.objective)


def test_lowrank_state_and_parameters_are_smaller_than_full():
    d, k, rank = 64, 8, 8
    state_elements = k * (d * rank + rank * rank + 2 * rank + d)
    assert state_elements < 0.5 * k * d * d
    assert covariance_parameter_count("lowrank", k, d, rank) < covariance_parameter_count(
        "full", k, d
    )
    # the default rank at d=512 is d/8, so the wide table's arm is 64
    assert covariance_parameter_count("lowrank", 8, 512) < covariance_parameter_count(
        "full", 8, 512
    )


def _reference_lowrank_log_prob(x: np.ndarray, params: dict) -> np.ndarray:
    """The direct per-component Woodbury form, without any stacking."""
    means = params["means"]
    psi = np.asarray(params["covariances"], dtype=np.float64)
    loadings = np.asarray(params["loadings"], dtype=np.float64)
    k, rank = loadings.shape[0], loadings.shape[2]
    d = x.shape[1]
    out = np.empty((x.shape[0], k), dtype=np.float64)
    for j in range(k):
        root = np.sqrt(psi[j])
        u = loadings[j] / root[:, None]
        gram = np.eye(rank) + u.T @ u
        _sign, logdet_gram = np.linalg.slogdet(gram)
        logdet = float(np.sum(np.log(psi[j])) + logdet_gram)
        w = (x - means[j]) / root
        wu = w @ u
        quad = np.einsum("ij,ij->i", w, w) - np.einsum(
            "ij,ij->i", np.linalg.solve(gram, wu.T).T, wu
        )
        out[:, j] = -0.5 * (d * np.log(2 * np.pi) + logdet + quad)
    return out


def test_lowrank_stacked_and_wide_paths_match_the_reference():
    """Cover stacked, chunked, and the high-k wide path with a tail chunk."""
    rng = np.random.default_rng(7)
    for d, rank, n, k in ((64, 8, 257, 3), (300, 4, 1000, 3), (300, 2, 1000, 64)):
        means = rng.normal(size=(k, d)) * 2.0 / np.sqrt(d)
        psi = np.exp(rng.uniform(np.log(0.2), np.log(0.6), size=(k, d)))
        loadings = rng.normal(size=(k, d, rank)) * np.sqrt(2.0)
        x = rng.normal(size=(n, d))
        params = {
            "means": means,
            "covariances": psi,
            "loadings": loadings,
            "weights": np.full(k, 1.0 / k),
            "covariance": "lowrank",
        }
        model = GaussianMixture(k, covariance="lowrank", rank=rank)
        got = model.component_log_prob(x, params)
        ref = _reference_lowrank_log_prob(x, params)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)
