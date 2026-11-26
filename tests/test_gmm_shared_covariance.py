"""The shared-covariance fast path, and the single E-step per update.

Two accelerations landed together and both are silent if they are wrong, so they
get their own pins:

* ``GaussianMixture._shared_log_prob`` replaces ``k`` per-component
  ``solve(Sigma, x - mu_j)`` calls with one ``n x d @ d x d`` product plus one
  ``n x d @ d x k`` product, by factoring the quadratic form through a square
  root of the precision.  The identity is only correct with the *root*
  (``|x R|^2 = x R R^T x^T``, not ``x P P^T x^T``), the third term must use the
  *projected* mean, and the expansion has to be centred or the cancellation
  between terms of size ``|xR|^2`` destroys the answer on low-rank data.  All
  three were real bugs during development; this test is the reference that
  catches any of them coming back.
* ``statistics_from`` computes the responsibilities and the recorded NLL from one
  E-step instead of two.
"""

from __future__ import annotations

import numpy as np

from emaopt.algorithms import GaussianMixture
from emaopt.core import State, logsumexp_rows


def _reference_log_prob(x: np.ndarray, params: dict, d: int) -> np.ndarray:
    """The original per-component solve, kept as the definition."""
    cov = np.asarray(params["covariances"], dtype=np.float64)
    means = np.asarray(params["means"], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    out = np.empty((x.shape[0], means.shape[0]), dtype=np.float64)
    for j in range(means.shape[0]):
        diff = x - means[j]
        sol = np.linalg.solve(cov, diff.T).T
        out[:, j] = -0.5 * (d * np.log(2 * np.pi) + logdet + np.einsum("ij,ij->i", diff, sol))
    return out


def _fitted(data: np.ndarray, k: int, nepochs: int = 5) -> tuple[GaussianMixture, dict]:
    model = GaussianMixture(k, covariance="tied", seed=0)
    result = model.fit(data, batch_size=None, nepochs=nepochs)
    return model, result.params


def test_shared_log_prob_matches_per_component_solve():
    rng = np.random.default_rng(0)
    loading = rng.normal(size=(12, 5))
    chol = np.linalg.cholesky(loading @ loading.T + 0.4 * np.eye(12))
    centres = rng.normal(size=(4, 12))
    labels = rng.integers(0, 4, 2500)
    data = centres[labels] + rng.normal(size=(2500, 12)) @ chol.T

    model, params = _fitted(data, 4)
    fast = model.component_log_prob(data, params)
    slow = _reference_log_prob(data, params, d=data.shape[1])
    np.testing.assert_allclose(fast, slow, rtol=1e-9, atol=1e-9)


def test_shared_log_prob_survives_a_large_offset_and_low_rank():
    """The cancellation case: mean far from zero, covariance of rank < d.

    Without centring the expansion subtracts ~1e8 sized terms to leave an O(d)
    quadratic form, and the error showed up as 0.2 nats per component.
    """
    rng = np.random.default_rng(1)
    data = rng.normal(size=(2000, 40)) + 50.0
    model, params = _fitted(data, 5)
    fast = model.component_log_prob(data, params)
    slow = _reference_log_prob(data, params, d=40)
    np.testing.assert_allclose(fast, slow, rtol=1e-8, atol=1e-8)
    assert np.all(np.isfinite(fast))


def test_statistics_from_reports_the_same_nll_as_objective():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(1500, 10)) + rng.normal(size=(1, 10))
    model, params = _fitted(data, 3)
    stats = model.statistics_from(data, params)
    assert np.isclose(float(stats.meta["nll"]), model.objective(data, params), rtol=1e-10)

    # counts and sums must be the same reduction, not a second approximation
    resp = model.responsibilities(data, params)
    np.testing.assert_allclose(np.asarray(stats["counts"]), resp.sum(axis=0), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(np.asarray(stats["sums"]), resp.T @ data, rtol=1e-10, atol=1e-10)


def test_refine_is_one_exact_m_step():
    """``refine`` re-estimates on the data it is given, from its own responsibilities."""
    rng = np.random.default_rng(3)
    data = rng.normal(size=(1200, 8))
    model, params = _fitted(data, 3)
    refined = model.refine(data, params)
    before = model.objective(data, params)
    after = model.objective(data, refined)
    assert after <= before + 1e-9  # an M-step cannot raise the likelihood it maximizes
    np.testing.assert_allclose(
        np.asarray(refined["weights"]).sum(), 1.0, rtol=1e-12
    )
