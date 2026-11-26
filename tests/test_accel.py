"""The libmvec shim is lazily built, so its first use must be thread-safe.

A 16-worker ICA fold deadlocked in ``ctypes.CDLL`` when several workers raced
through ``_lib``/``_build``.  These tests pin both halves of the fix: only one
build runs, and every concurrent caller gets numpy-identical tanh output.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from emaopt import accel


def test_concurrent_first_use_builds_once(monkeypatch):
    calls: list[int] = []
    real_build = accel._build

    def slow_build():
        calls.append(1)
        time.sleep(0.05)  # widen the window a race would need
        return real_build()

    monkeypatch.setattr(accel, "_state", {"lib": None, "tried": False})
    monkeypatch.setattr(accel, "_build", slow_build)

    with ThreadPoolExecutor(max_workers=8) as pool:
        libs = list(pool.map(lambda _: accel._lib(), range(8)))

    assert len(calls) == 1
    assert all(lib is libs[0] for lib in libs)


def test_concurrent_tanh_matches_numpy(monkeypatch):
    monkeypatch.setattr(accel, "_state", {"lib": None, "tried": False})
    x = np.random.default_rng(0).standard_normal((2048, 64))

    with ThreadPoolExecutor(max_workers=16) as pool:
        outs = list(pool.map(lambda _: accel.tanh(x), range(32)))

    assert all(np.array_equal(out, outs[0]) for out in outs)
    assert np.allclose(outs[0], np.tanh(x), rtol=0, atol=1e-12)
