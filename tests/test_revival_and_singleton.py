"""T31 (singleton variance) and T32 (revival detection) in the diag family."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.gmm import GaussianMixture
from emaopt.core import State, UpdateContext


def diag_state(counts, sums, sq, meta=None):
    return State({"counts": np.asarray(counts, dtype=float),
                  "sums": np.asarray(sums, dtype=float),
                  "sq": np.asarray(sq, dtype=float)},
                 0, dict(meta or {}))


def test_singleton_is_shrunk_toward_the_pooled_variance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 6)) * 1.5 + 2.0
    n, d = x.shape
    sums = x.sum(axis=0)
    sq = (x**2).sum(axis=0)
    model = GaussianMixture(2, covariance="diag", seed=0, reg_covar=1e-6)
    state = diag_state(
        [float(n), 1.0],
        np.vstack([sums, x[0]]),
        np.vstack([sq, x[0] ** 2]),
    )

    params = model.decode(state)

    # the decoder pools over the state's total mass, which includes the singleton
    mass = float(n) + 1.0
    mean_all = (sums + x[0]) / mass
    pooled = np.maximum((sq + x[0] ** 2) / mass - mean_all**2, 0.0)
    # one floor for the state: reg_covar times the pooled mean variance
    floor = 1e-6 * pooled.mean()
    want = (1.0 * 0.0 + d * np.maximum(pooled, floor)) / (1.0 + d) + floor
    assert np.allclose(params["covariances"][1], want, rtol=1e-10)
    assert np.all(params["covariances"][1] > 1e-4)      # not a reg_covar spike


def test_healthy_components_are_untouched():
    """N >= d must decode bit-identically to the unshrunk formula."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(800, 4))
    n = x.shape[0]
    sums = x.sum(axis=0)
    sq = (x**2).sum(axis=0)
    model = GaussianMixture(2, covariance="diag", seed=0, reg_covar=1e-6)
    state = diag_state([float(n), float(n)], np.vstack([sums, sums]),
                       np.vstack([sq, sq]))

    params = model.decode(state)

    pooled = np.maximum(sq / n - (sums / n) ** 2, 0.0)
    plain = sq / n - (sums / n) ** 2 + 1e-6 * pooled.mean()
    assert np.array_equal(params["covariances"][0], plain)
    assert np.array_equal(params["covariances"][1], plain)


def _pinned_state(var_second, masses=(1_000_000.0, 4.0), d=4):
    """A two-component state whose second component sits exactly at the floor."""
    reg = 1e-6
    mu0 = np.full(d, 3.0)
    mu1 = np.zeros(d)
    counts = np.array(masses)
    sums = np.vstack([mu0 * masses[0], mu1 * masses[1]])
    var0 = np.arange(1.0, d + 1.0)          # widest axis is the last one
    sq = np.vstack([masses[0] * (var0 + mu0**2),
                    masses[1] * (var_second + mu1**2)])
    return diag_state(counts, sums, sq)


def test_a_component_pinned_at_the_variance_floor_is_revived():
    model = GaussianMixture(2, covariance="diag", seed=0, reg_covar=1e-6,
                            revival=True, revival_threshold=1e-3)
    state = _pinned_state(np.full(4, 1e-6))

    out = model._revive(state, UpdateContext(step=0, batch_size=1000,
                                             dataset_size=1000))

    counts = out["counts"]
    assert counts[0] == pytest.approx(counts[1]) == pytest.approx(500_000.0)
    assert out.meta["_revivals"] == 1
    means = out["sums"] / counts[:, None]
    # separated along the donor's widest axis (index 3), symmetrically
    assert np.allclose(means[0] + means[1], 2 * 3.0)
    assert means[0][3] - means[1][3] > 0.1
    assert np.allclose(means[0][:3], means[1][:3])


def test_a_healthy_component_is_not_revived():
    model = GaussianMixture(2, covariance="diag", seed=0, reg_covar=1e-6,
                            revival=True, revival_threshold=1e-3)
    state = _pinned_state(np.ones(4))

    out = model._revive(state, UpdateContext(step=0, batch_size=1000,
                                             dataset_size=1000))

    assert out is state
