"""The optional compiled host E-step (T21) against the NumPy reference.

The artifact is optional, so every test here is skipped when it is not built;
what must hold either way is the last test -- the wiring is a no-op when the
kernel is absent or switched off.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt import GaussianMixture, hostkernels


def _available() -> bool:
    return hostkernels.available()


needs_kernel = pytest.mark.skipif(
    not _available(),
    reason="compiled host kernel not built",
)


def _reference_logits(x, means, covs, weights=None):
    """The broadcast form, written independently of the shipped BLAS path."""
    inv = 1.0 / covs
    logdet = np.log(covs).sum(axis=1)
    diff = x[:, None, :] - means[None, :, :]
    quad = np.einsum("nkd,kd->nk", diff * diff, inv)
    logits = -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + logdet[None, :] + quad)
    if weights is not None:
        logits = logits + np.log(np.maximum(weights, 1e-300))[None, :]
    return logits


def _reference_step(x, means, covs, weights):
    logits = _reference_logits(x, means, covs, weights)
    top = logits.max(axis=1, keepdims=True)
    top = np.where(np.isfinite(top), top, 0.0)
    e = np.exp(logits - top)
    s = e.sum(axis=1, keepdims=True)
    resp = e / np.maximum(s, 1e-300)
    resp[~np.isfinite(resp)] = 0.0
    resp = resp / np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)
    log_norm = top[:, 0] + np.log(s[:, 0])
    return {
        "counts": resp.sum(axis=0),
        "sums": resp.T @ x,
        "sq": resp.T @ (x * x),
        "nll": float(-log_norm.mean()),
    }


def _problem(n=512, k=6, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = rng.uniform(0.5, 1.5, size=(k, d))
    weights = rng.uniform(0.2, 2.0, size=k)
    weights /= weights.sum()
    return x, means, covs, weights


@needs_kernel
def test_logits_match_the_reference():
    x, means, covs, _ = _problem()
    got = hostkernels.diag_logits(x, means, covs)
    ref = _reference_logits(x, means, covs)
    assert np.max(np.abs(got - ref)) <= 1e-9
    assert got.shape == (x.shape[0], means.shape[0])


@needs_kernel
def test_c_logits_accept_non_float64_parameters():
    """ctypes requires float64, but callers may pass float32 or integer arrays.

    ``_prepare`` normalises its own copies; the direct path also transposes
    ``means`` and must not reuse the caller's dtype.
    """
    x, means, covs, weights = _problem(n=64, k=4, d=8)
    x32 = x.astype(np.float32)
    means32 = means.astype(np.float32)
    covs32 = covs.astype(np.float32)
    weights32 = weights.astype(np.float32)
    lib = hostkernels._load()
    assert lib is not None
    got = hostkernels._c_logits(lib, x32, means32, covs32, weights32)
    ref = _reference_logits(x32.astype(np.float64),
                            means32.astype(np.float64),
                            covs32.astype(np.float64),
                            weights32.astype(np.float64))
    assert np.max(np.abs(got - ref)) <= 1e-9

    means_i = np.rint(means).astype(np.int64)
    got_i = hostkernels._c_logits(lib, x, means_i, covs, weights)
    ref_i = _reference_logits(x, means_i.astype(np.float64), covs, weights)
    assert np.max(np.abs(got_i - ref_i)) <= 1e-9


@needs_kernel
def test_step_matches_the_reference():
    x, means, covs, weights = _problem()
    got = hostkernels.diag_estep(x, means, covs, weights)
    ref = _reference_step(x, means, covs, weights)
    for key in ("counts", "sums", "sq"):
        scale = max(float(np.max(np.abs(ref[key]))), 1.0)
        assert np.max(np.abs(got[key] - ref[key])) <= 1e-9 * scale, key
    assert abs(got["nll"] - ref["nll"]) <= 1e-9
    assert got["counts"].shape == (means.shape[0],)
    assert got["sums"].shape == means.shape


@needs_kernel
def test_returns_are_deterministic():
    x, means, covs, weights = _problem()
    a = hostkernels.diag_estep(x, means, covs, weights)
    b = hostkernels.diag_estep(x, means, covs, weights)
    for key in ("counts", "sums", "sq"):
        assert np.array_equal(a[key], b[key]), key
    assert a["nll"] == b["nll"]


@needs_kernel
@pytest.mark.parametrize("shape", [(1, 1, 1), (7, 1, 40), (300, 3, 1), (2, 16, 512)])
def test_shapes(shape):
    n, k, d = shape
    x, means, covs, weights = _problem(n=n, k=k, d=d)
    got = hostkernels.diag_estep(x, means, covs, weights)
    ref = _reference_step(x, means, covs, weights)
    assert np.max(np.abs(got["sums"] - ref["sums"])) <= 1e-9 * max(
        float(np.max(np.abs(ref["sums"]))), 1.0)
    assert abs(got["nll"] - ref["nll"]) <= 1e-9


@needs_kernel
def test_gmm_statistics_use_the_kernel_and_match_it():
    """The wiring reaches the same numbers the kernel produces directly."""
    x, means, covs, weights = _problem(n=900, k=5, d=8)
    params = {"means": means, "covariances": covs, "weights": weights,
              "covariance": "diag"}
    model = GaussianMixture(5, covariance="diag")
    state = model.statistics_from(x, params)
    direct = hostkernels.diag_estep(x, means, covs, weights)
    assert np.array_equal(np.asarray(state["counts"]), direct["counts"])
    assert np.array_equal(np.asarray(state["sums"]), direct["sums"])
    assert np.array_equal(np.asarray(state["sq"]), direct["sq"])

    logits_kernel = hostkernels.diag_logits(x, means, covs)
    logits_model = model.component_log_prob(x, params)
    assert np.array_equal(logits_kernel, logits_model)


@needs_kernel
def test_component_log_prob_does_not_add_log_weights_twice():
    """The log-density entry point must return log N, not log N + log w.

    ``statistics_from`` *does* need the weighted form, so the kernel can add
    log w; ``component_log_prob``'s callers add the prior themselves.  With
    non-uniform weights a double count would show up as exactly log w.
    """
    x, means, covs, weights = _problem(n=300, k=4, d=8)
    params = {"means": means, "covariances": covs, "weights": weights,
              "covariance": "diag"}
    model = GaussianMixture(4, covariance="diag")
    got = model.component_log_prob(x, params)
    assert np.max(np.abs(got - _reference_logits(x, means, covs))) <= 1e-9
    weighted = hostkernels.diag_logits(x, means, covs, weights)
    assert np.allclose(weighted - got, np.log(weights)[None, :], atol=1e-12)


def test_disabling_the_kernel_is_a_no_op(monkeypatch):
    """EMAOPT_HOST_KERNEL=numpy restores the shipped NumPy path exactly."""
    x, means, covs, weights = _problem(n=400, k=4, d=8)
    params = {"means": means, "covariances": covs, "weights": weights,
              "covariance": "diag"}
    model = GaussianMixture(4, covariance="diag")
    monkeypatch.setenv("EMAOPT_HOST_KERNEL", "numpy")
    assert hostkernels.available() is False
    state = model.statistics_from(x, params)
    ref = _reference_step(x, means, covs, weights)
    assert abs(float(state.meta["nll"]) - ref["nll"]) <= 1e-9
    assert np.max(np.abs(np.asarray(state["sums"]) - ref["sums"])) <= 1e-9 * max(
        float(np.max(np.abs(ref["sums"]))), 1.0)


def test_status_reports_why_the_kernel_is_unavailable(monkeypatch):
    monkeypatch.setenv("EMAOPT_HOST_KERNEL", "numpy")
    status = hostkernels.status()
    assert status["available"] is False
    assert "EMAOPT_HOST_KERNEL" in status["error"]


@needs_kernel
def test_log_weights_are_counted_exactly_once():
    """``component_log_prob`` is unweighted; ``statistics_from`` adds log w.

    The compiled kernel can add log-weights itself, so the risk is a caller that
    passes weights to both the kernel and its own softmax and counts them twice.
    This pins the contract on the path that matters: log-density without
    weights, statistics with them.
    """
    x, means, covs, weights = _problem(n=600, k=5, d=8)
    params = {"means": means, "covariances": covs, "weights": weights,
              "covariance": "diag"}
    model = GaussianMixture(5, covariance="diag")

    logits = model.component_log_prob(x, params)
    assert np.array_equal(logits, hostkernels.diag_logits(x, means, covs))
    weighted = hostkernels.diag_logits(x, means, covs, weights)
    # non-uniform weights must actually change the log-density by log w
    assert np.allclose(weighted - logits,
                       np.log(weights)[None, :], atol=1e-12)

    state = model.statistics_from(x, params)
    direct = hostkernels.diag_estep(x, means, covs, weights)
    assert np.array_equal(np.asarray(state["counts"]), direct["counts"])
    ref = _reference_step(x, means, covs, weights)
    assert np.max(np.abs(np.asarray(state["sums"]) - ref["sums"])) <= 1e-9 * max(
        float(np.max(np.abs(ref["sums"]))), 1.0)
