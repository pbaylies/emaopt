"""The whitening cache is content-keyed, not identity-keyed (T60).

The bug: ``FastICAStat.whitening_matrix`` cached on ``self._wz_source is not
cov``, and ``_fold`` assigns a *new* array on every block, so the check never
hit and a d x d eigen-solve ran per block even when the covariance had not moved.

What the tests pin, in the order the measurements established it:

* an unchanged covariance is not re-solved;
* ``whitening="fixed"`` solves exactly once and uses the supplied basis;
* a **cadence** (``whiten_every``) is the safe speed setting: a few blocks of
  reuse cost <1% of the unmixing;
* a **large movement tolerance** is not safe: at 5% the whitening goes stale
  enough to lose the solution (|cos| ~0.45).  That is why ``whiten_tol`` ships
  at 0.0 and the cadence is the knob with a measured trajectory cost.

Sizes: d=128, 1024-row blocks, 64 blocks -- a stream long enough for the
covariance to settle and for a solve to be a real 128x128 eigendecomposition.
The file ran in ~4-6 s locally.
"""
from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.ica import FastICAStat, fastica_td, whitening_matrix

D, K = 128, 64
ROWS, BLOCK = 32_768, 1024
BLOCKS = ROWS // BLOCK


def _stationary(n=ROWS, d=D, seed=0):
    """iid Gaussian under a random mix: a covariance that *settles*.

    A stream whose covariance keeps moving cannot demonstrate a cadence, and one
    whose covariance is exactly constant cannot demonstrate anything else.
    """
    rng = np.random.default_rng(seed)
    mix = rng.normal(size=(d, d)) / np.sqrt(d)
    return rng.normal(size=(n, d)) @ mix


def _matched(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float(np.abs(a @ b.T).max(axis=1).mean())


def test_unchanged_covariance_is_not_resolved():
    stat = FastICAStat(8, mode="ema", alpha=0.02, whitening="running",
                       whiten_tol=0.0)
    stat.prepare(d=D)
    stat.update(_stationary(1024, seed=1))
    # ``update`` folds, so the next call legitimately pays for the new C;
    # what must not happen is a second solve with no fold in between.
    stat.whitening_matrix()
    solves_after_first = stat.whitening_solves
    assert solves_after_first >= 2
    for _ in range(5):
        stat.whitening_matrix()            # no fold in between
    assert stat.whitening_solves == solves_after_first


def test_cadence_is_the_safe_speed_setting():
    cosines, saved = [], []
    for seed in range(1):
        data = _stationary(seed=seed)
        # this file pins the *running* path; auto is the fixed-whitening
        # default now, and it has its own gate in test_streaming_ica.
        common = dict(batch_size=BLOCK, nepochs=1, decay=0.02, seed=seed,
                      shuffle=False, record_history=False, n_components=K,
                      whitening="running")
        ref = fastica_td(data, **common)
        cadence = fastica_td(data, whiten_every=4, **common)
        cosines.append(_matched(ref.unmixing, cadence.unmixing))
        saved.append((ref.config["whitening_solves"],
                      cadence.config["whitening_solves"]))
    assert all(got < ref for ref, got in saved), saved
    assert max(got for _ref, got in saved) <= 0.5 * min(ref for ref, _g in saved)
    assert min(cosines) >= 0.99, cosines


def test_large_tolerance_is_the_wrong_knob():
    """5% of relative movement is far too coarse -- documented, not shipped."""
    data = _stationary(seed=0)
    common = dict(batch_size=BLOCK, nepochs=1, decay=0.02, seed=0,
                  shuffle=False, record_history=False, n_components=K,
                  whitening="running")
    ref = fastica_td(data, **common)
    stale = fastica_td(data, whiten_tol=0.05, **common)
    assert stale.config["whitening_solves"] < ref.config["whitening_solves"]
    assert _matched(ref.unmixing, stale.unmixing) < 0.95


def test_zero_tolerance_keeps_the_every_fold_behaviour():
    data = _stationary(16_384, seed=4)
    result = fastica_td(data, 16, batch_size=1024, nepochs=1, seed=0,
                        shuffle=False, whiten_tol=0.0, record_history=False,
                        whitening="running")
    # 16 blocks; C moves every fold, so the every-fold arm keeps solving
    assert result.config["whitening_solves"] >= 8


def test_fixed_whitening_solves_once_and_fits():
    data = _stationary(16_384, seed=5)
    cov = data.T @ data / data.shape[0]
    wz = whitening_matrix(cov, K)
    fixed = fastica_td(data, K, batch_size=BLOCK, seed=0, shuffle=False,
                       whitening="fixed", covariance=cov, record_history=False)
    assert fixed.config["whitening_solves"] == 1
    assert fixed.unmixing.shape[1] == data.shape[1]
    # the fixed arm whitens with the supplied basis, not its own first block
    assert np.allclose(fixed.whitening, wz)
    assert np.isfinite(fixed.objective)


def test_fixed_whitening_requires_a_covariance():
    with pytest.raises(ValueError, match="covariance"):
        fastica_td(_stationary(512, d=16, seed=6), 4, whitening="fixed",
                   record_history=False)
