"""GPU differential arms for streaming PCA and TD-FastICA: unit gates.

Everything skips cleanly without CUDA. The semantic gates are the load-bearing
assertions: the device transcription must equal the numpy function it replaces
before any parity claim downstream means anything (the Phase 7b discipline).
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)

from emaopt.algorithms.ica import fastica_td
from emaopt.gpu_pca_ica import (
    ica_gpu_stat_swap,
    pca_gpu_stat_swap,
    semantic_gate_ica,
    semantic_gate_pca,
)
from emaopt.state.sufficient import GramianStat


def _gauss(n=4096, d=64, seed=3):
    """``(n, d)`` samples with correlated columns.

    These are rows of data, not a Gramian: every stat under test forms
    ``X_b^T X_b`` itself.  This helper used to return ``a @ a.T / n``, which
    fed an ``(n, n)`` matrix in as data -- so the "d=32 ICA gate" was really
    whitening 2048 dimensions (45 s) and the PCA gate 4096 (5 s), while looking
    like the small gate it was named for.
    """
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, d))
    a[:, 1:] += 0.6 * a[:, :-1]  # keep the columns correlated
    return a


def _worst(d: dict) -> float:
    # the documented fields only: bool flags would masquerade as gaps
    assert d["passed_1e_minus_12"] is True, d
    return float(d["worst_rel_gap"])


def test_pca_semantic_gate():
    data = _gauss()
    out = semantic_gate_pca([data[:1024], data[1024:2048]], dtype="float64")
    assert _worst(out) <= 1e-12


def test_ica_semantic_gate():
    data = _gauss(2048, 32)
    out = semantic_gate_ica([data[:1024], data[1024:]], d=32,
                            n_components=8, dtype="float64")
    assert _worst(out) <= 1e-12


def test_pca_f64_trajectory_parity():
    data = _gauss(4096, 48)
    cpu = GramianStat(center=True, mode="sum")
    dev = GramianStat(center=True, mode="sum")
    pca_gpu_stat_swap(dev, dtype="float64", device="cuda")
    for start in range(0, data.shape[0], 1024):
        blk = data[start:start + 1024]
        cpu.update(blk)
        dev.update(blk)
    gap = float(np.abs(cpu.covariance() - dev.covariance()).max())
    assert gap <= 1e-12 * max(float(np.abs(cpu.covariance()).max()), 1e-30)


def test_ica_f64_trajectory_parity():
    data = _gauss(2048, 32)
    # the swap arm patches the streaming path; auto would precompute a
    # covariance and never call it
    kw = dict(n_components=8, batch_size=1024, seed=0, whitening="running")
    ref = fastica_td(data, **kw)
    with ica_gpu_stat_swap("float64", "cuda"):
        got = fastica_td(data, **kw)
    ref_w = np.asarray(ref.unmixing, dtype=np.float64)
    got_w = np.asarray(got.unmixing, dtype=np.float64)
    assert np.abs(ref_w - got_w).max() <= 1e-10 * max(np.abs(ref_w).max(), 1e-30)


def test_ladder_dtype_resolution_is_explicit():
    from emaopt.gpu_pca_ica import resolve_dtype
    assert resolve_dtype("float64") == torch.float64
    assert resolve_dtype("float32") == torch.float32
    with pytest.raises((TypeError, ValueError)):
        resolve_dtype("bfloat99")
