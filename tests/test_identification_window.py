"""T30: the full-family identification gate can read window evidence.

``decode`` gated the full family on ``counts < 4*d``, but under an EMA fold
``counts`` is batch-scale while the covariance is an average over ~(2-a)/a
batches.  The gate now reads ``counts * (2-a)/a`` when the caller passes the
step size, and stays on raw counts by default so every recorded result keeps its
meaning.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.gmm import GaussianMixture
from emaopt.core import State


def _correlated(n: int = 120, d: int = 32, seed: int = 0):
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(d, d)) / np.sqrt(d)
    x = rng.normal(size=(n, d)) @ mixing
    # three separated clusters, so the decode has real structure to find
    x[: n // 3] += 2.0
    x[n // 3: 2 * n // 3] -= 2.0
    return x


def _decoded(window):
    x = _correlated()
    model = GaussianMixture(3, covariance="full", identification_window=window)
    init = model.initialize(x)
    params = model.decode(init)
    stats = model.statistics_from(x, params)
    return model, model.decode(State(stats.values, 0, dict(init.meta)))


def test_raw_counts_stay_the_default_and_regularise():
    x = _correlated()
    model = GaussianMixture(3, covariance="full")
    assert model.identification_window is None
    _, params = _decoded(None)
    # 120 rows over 3 components is ~40 counts each against a floor of 4*32=128
    assert params["regularized_components"] == 3
    assert params["identification_factor"] == 1.0
    for cov in params["covariances"]:
        assert np.allclose(cov, np.diag(np.diag(cov)))


def test_window_evidence_identifies_the_same_state():
    _, params = _decoded(0.1)
    # 40 counts * (2-0.1)/0.1 = 760 > 128, so the off-diagonals are kept
    assert params["identification_factor"] == pytest.approx(19.0)
    assert params["regularized_components"] == 0
    cov = params["covariances"]
    assert all(c.shape == (32, 32) for c in cov)
    off = cov - np.stack([np.diag(np.diag(c)) for c in cov])
    assert np.abs(off).max() > 0.0
    assert np.allclose(cov, np.swapaxes(cov, 1, 2))


def test_window_one_is_the_raw_rule():
    _, raw = _decoded(None)
    _, one = _decoded(1.0)
    assert one["identification_factor"] == 1.0
    assert one["regularized_components"] == raw["regularized_components"]
    assert np.array_equal(one["covariances"], raw["covariances"])


def test_invalid_windows_are_refused():
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            GaussianMixture(2, covariance="full", identification_window=bad)


def test_diagonal_family_is_unaffected():
    x = _correlated()
    model = GaussianMixture(3, covariance="diag", identification_window=0.1)
    init = model.initialize(x)
    params = model.decode(init)
    assert params["covariance"] == "diag"
    assert "identification_factor" not in params
