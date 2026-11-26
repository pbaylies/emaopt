"""Resident Gramian / TD-FastICA arms (T24): do they change the answer?

The resident arms exist to remove a per-block upload, not to change arithmetic,
so the load-bearing assertions are equality against the arms that already exist:

* float64 PCA resident == the swap arm == the numpy ``GramianStat`` (1e-12);
* float64 ICA resident == ``fastica_td`` unmixing (1e-10, the swap arm's bar);
* an EMA Gramian fold is refused instead of silently re-implemented.

Sizes: d=128 with 16 blocks for the Gramian, d=64 with 32 blocks for ICA.  That
is a real stream (a 16.8 MB upload, a 128x128 Gramian, 32 whitening solves on
the numpy side) rather than the smallest input that exercises the path, and the
file runs in ~6-10 s.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA", allow_module_level=True)

from emaopt.algorithms.ica import fastica_td
from emaopt.gpu_pca_ica import (
    ica_gpu_resident_fit,
    pca_gpu_resident_stat,
    pca_gpu_stat_swap,
)
from emaopt.state.sufficient import GramianStat

D, BLOCK, BLOCKS = 128, 8192, 16
ROWS = BLOCK * BLOCKS


def _data(n=ROWS, d=D, seed=5):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d))


def _rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.abs(a - b).max() / max(float(np.abs(a).max()), 1e-300))


def test_pca_resident_matches_numpy_and_swap():
    data = _data()
    numpy_stat = GramianStat(center=True, mode="sum")
    swap_stat = GramianStat(center=True, mode="sum")
    pca_gpu_stat_swap(swap_stat, dtype="float64", device="cuda")
    for start in range(0, data.shape[0], BLOCK):
        blk = data[start:start + BLOCK]
        numpy_stat.update(blk)
        swap_stat.update(blk)

    resident, info = pca_gpu_resident_stat(data, batch_size=BLOCK,
                                           dtype="float64")
    assert info["blocks"] == BLOCKS
    assert info["rows"] == ROWS

    assert _rel(numpy_stat.covariance(), resident.covariance()) <= 1e-12
    # the resident fold is the same elementwise addition the swap arm did
    assert _rel(swap_stat.covariance(), resident.covariance()) <= 1e-12
    assert _rel(numpy_stat.values["gram"], resident.values["gram"]) <= 1e-12


def test_pca_resident_f32_stays_within_the_chunk_gate():
    data = _data()
    ref = GramianStat(center=True, mode="sum")
    ref.update(data)
    resident, info = pca_gpu_resident_stat(data, batch_size=BLOCK,
                                           dtype="float32")
    assert info["dtype"] == "float32"
    # the precision choice's cost at this blocking; the shard-shape gate is
    # T22's R1 (5e-7 for 2048-row chunks at d=512)
    assert _rel(ref.values["gram"], resident.values["gram"]) <= 1e-5


def test_pca_resident_refuses_ema():
    with pytest.raises(ValueError, match="sum"):
        pca_gpu_resident_stat(_data(2048), stat=GramianStat(center=True,
                                                            mode="ema"))


def test_ica_resident_matches_numpy_trajectory():
    data = _data(16_384, 64)
    # resident arm is the single-pass form, so compare like with like
    kw = dict(n_components=16, batch_size=512, seed=0, shuffle=False,
              whitening="running")
    ref = fastica_td(data, **kw)
    got, info = ica_gpu_resident_fit(data, 16, batch_size=512, seed=0,
                                     shuffle=False, whitening="running")
    assert info["updates"] == 32
    assert _rel(ref.unmixing, got.unmixing) <= 1e-10


def test_ica_resident_keeps_the_block_order():
    """A permuted stream must land somewhere else -- otherwise the index path is
    silently reading the wrong rows and the equality test above proves nothing."""
    data = _data(16_384, 64)
    straight, _ = ica_gpu_resident_fit(data, 16, batch_size=512, seed=0,
                                       shuffle=False)
    shuffled, _ = ica_gpu_resident_fit(data, 16, batch_size=512, seed=0,
                                       shuffle=True)
    assert _rel(straight.unmixing, shuffled.unmixing) > 1e-12
