"""The optional numba backend against the NumPy expressions it replaces.

Skipped when numba is not installed: the backend is an extra, and a checkout
without it must behave exactly as before.
"""

from __future__ import annotations

import numpy as np
import pytest

numba = pytest.importorskip("numba")

from emaopt.kernels import numba as backend  # noqa: E402


@pytest.fixture
def warm(monkeypatch):
    monkeypatch.setenv("EMAOPT_NUMBA", "1")
    monkeypatch.setenv("EMAOPT_HOST_KERNEL_THREADS", "2")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    monkeypatch.setitem(backend._state, "configured", False)
    backend.warm()
    return backend


def _problem(n=8192, d=16, k=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = rng.uniform(0.5, 1.5, size=(k, d))
    return x, means, covs


def _numpy_logits(x, means, covs, weights=None):
    inv = 1.0 / covs
    quad = (x * x) @ inv.T - 2.0 * (x @ (means * inv).T)
    quad += (means * (means * inv)).sum(axis=1)[None, :]
    out = -0.5 * quad - 0.5 * (np.log(covs).sum(axis=1)[None, :]
                               + x.shape[1] * np.log(2.0 * np.pi))
    if weights is not None:
        out += np.log(weights)[None, :]
    return out


def _numpy_estep(x, means, covs, weights):
    logits = _numpy_logits(x, means, covs, weights)
    top = logits.max(axis=1)
    top = np.where(np.isfinite(top), top, 0.0)
    e = np.exp(logits - top[:, None])
    s = e.sum(axis=1)
    resp = e / np.maximum(s[:, None], 1e-300)
    resp[~np.isfinite(resp)] = 0.0
    resp /= np.maximum(resp.sum(axis=1)[:, None], 1e-300)
    return {"counts": resp.sum(axis=0), "sums": resp.T @ x,
            "sq": resp.T @ (x * x), "nll": float(-(top + np.log(s)).mean())}


def test_logits_match_numpy(warm):
    x, means, covs = _problem()
    got = warm.logits(x, means, covs, None,
                      x.shape[0] * x.shape[1] * means.shape[0], force=True)
    assert got is not None and got.shape == (x.shape[0], means.shape[0])
    assert np.max(np.abs(got - _numpy_logits(x, means, covs))) <= 1e-9


def test_logits_declines_below_the_work_gate(warm):
    x, means, covs = _problem(n=16, d=4, k=2)
    assert warm.logits(x, means, covs, None, x.size) is None


def test_wide_logits_matches_numpy(warm):
    x, means, covs = _problem(n=512, d=320, k=8)
    work = x.shape[0] * means.shape[0] * means.shape[1]
    got = warm.logits(x, means, covs, None, work)
    assert got is not None
    assert np.max(np.abs(got - _numpy_logits(x, means, covs))) <= 1e-9


def test_logits_declines_at_narrow_width(warm):
    x, means, covs = _problem(n=8192, d=16, k=16)
    work = x.shape[0] * means.shape[0] * means.shape[1]
    assert work >= warm._WORK_GATE
    assert warm.logits(x, means, covs, None, work) is None


def test_fused_estep_matches_numpy(warm):
    x, means, covs = _problem(n=512, d=16, k=8)
    weights = np.full(means.shape[0], 1.0 / means.shape[0])
    got = warm.estep(x, means, covs, weights, force=True)
    ref = _numpy_estep(x, means, covs, weights)
    assert got is not None
    for key in ("counts", "sums", "sq"):
        assert np.max(np.abs(got[key] - ref[key])) <= 1e-9, key
    assert abs(got["nll"] - ref["nll"]) <= 1e-9


def test_wide_fused_estep_matches_numpy(warm):
    """The wide path is two GEMMs plus a compiled softmax, and stays exact."""
    x, means, covs = _problem(n=512, d=320, k=8)
    weights = np.full(means.shape[0], 1.0 / means.shape[0])
    got = warm.estep(x, means, covs, weights)
    ref = _numpy_estep(x, means, covs, weights)
    assert got is not None
    for key in ("counts", "sums", "sq"):
        assert np.max(np.abs(got[key] - ref[key])) <= 1e-9, key
    assert abs(got["nll"] - ref["nll"]) <= 1e-9


def test_epoch_runner_matches_numpy(warm, monkeypatch):
    from emaopt import emeans
    from emaopt.algorithms import kmeans as kmeans_mod

    x = np.random.default_rng(4).normal(size=(1000, 8))
    monkeypatch.delenv("EMAOPT_NUMBA", raising=False)
    monkeypatch.delenv("EMAOPT_INTERNAL_KERNELS", raising=False)
    kmeans_mod._INTERNAL_RUNNER = False
    ref = emeans(x, k=4, nepochs=2, seed=0)

    monkeypatch.setenv("EMAOPT_NUMBA", "1")
    kmeans_mod._INTERNAL_RUNNER = False
    got = emeans(x, k=4, nepochs=2, seed=0)
    assert got.updates == ref.updates
    assert got.objective == pytest.approx(ref.objective, rel=1e-9, abs=1e-12)
    kmeans_mod._INTERNAL_RUNNER = False


def test_epoch_runner_declines_wide_updates(warm):
    """The gate keeps the runner where the loop's threaded GEMM is faster.

    Measured with five epochs at eight threads: 11.52 against the loop's 17.19
    ms at 8192 x 64, but 190.01 against 143.29 at 8192 x 320 k=64.  The gate
    is per-update work, so both inputs are (n, d, k, batch, per_epoch, total).
    """
    runner = backend.epoch_runner()
    assert runner is not None
    assert runner.ready(8192, 64, 16, 512, 16, 80) is True
    assert runner.ready(8192, 320, 64, 512, 16, 80) is False


def test_logsumexp_matches_numpy(warm):
    logits = np.random.default_rng(1).normal(-3.0, 1.5, size=(4096, 16))
    got = warm.logsumexp(logits)
    top = logits.max(axis=1)
    ref = top + np.log(np.exp(logits - top[:, None]).sum(axis=1))
    assert got is not None
    assert np.max(np.abs(got - ref)) <= 1e-12


def test_logsumexp_keeps_non_finite_rows(warm):
    edge = np.array([[-np.inf, -np.inf], [np.nan, 1.0], [0.0, 0.0]])
    got = warm.logsumexp(edge)
    with np.errstate(divide="ignore", invalid="ignore"):
        top = edge.max(axis=1)
        top = np.where(np.isfinite(top), top, 0.0)
        ref = top + np.log(np.exp(edge - top[:, None]).sum(axis=1))
    assert got is not None
    assert np.array_equal(np.isnan(got), np.isnan(ref))
    assert np.allclose(got[~np.isnan(ref)], ref[~np.isnan(ref)])


def test_unset_env_means_the_arm_is_off(monkeypatch):
    monkeypatch.delenv("EMAOPT_NUMBA", raising=False)
    monkeypatch.setitem(backend._state, "configured", False)
    assert backend.available() is False
    x, means, covs = _problem()
    assert backend.logits(x, means, covs, None, 10**9) is None
