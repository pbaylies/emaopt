"""The blocked Gramian (T22) used by the PCA and TD-ICA device arms.

The default float64 product must stay bit-identical to the shipped path; a
reduced dtype must use the blocked form, and the blocked form must be closer to
the float64 answer than the unblocked one -- that ordering is the whole reason
it exists.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emaopt.gpu_pca_ica import _GRAM_CHUNK_ROWS, _device_gram  # noqa: E402

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _block(n=5000, d=17, seed=0):
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(n, d)))


def _rel(got, ref):
    scale = max(float(np.abs(ref).max()), 1e-300)
    return float(np.abs(got - ref).max() / scale)


@CUDA
def test_f64_gramian_is_the_plain_product():
    """f64 takes one product -- the same op, bit for bit.

    "Unchanged" is stated against the operation itself, not against NumPy: a
    device GEMM does not reproduce the host BLAS reduction order bit for bit at
    any precision, and never did.
    """
    x = _block()
    xt = torch.as_tensor(x, dtype=torch.float64, device="cuda")
    got = _device_gram(xt).cpu().numpy()
    assert np.array_equal(got, (xt.t() @ xt).double().cpu().numpy())
    assert _rel(got, x.T @ x) <= 1e-14


@CUDA
def test_f32_gramian_is_blocked_and_closer_than_unblocked():
    x = _block(n=5000, d=33)
    ref = x.T @ x
    xt = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    blocked = _device_gram(xt).cpu().numpy()
    unblocked = (xt.t() @ xt).double().cpu().numpy()
    assert blocked.shape == (33, 33)
    assert _rel(blocked, ref) <= 5e-7
    assert _rel(blocked, ref) <= _rel(unblocked, ref)


@CUDA
def test_blocked_gramian_is_symmetric_and_deterministic():
    x = _block(n=4097, d=9)          # not a multiple of the chunk
    xt = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    first = _device_gram(xt).cpu().numpy()
    second = _device_gram(xt).cpu().numpy()
    assert np.array_equal(first, second)
    assert np.allclose(first, first.T, rtol=0.0, atol=0.0)


@CUDA
def test_small_blocks_skip_the_loop():
    """Fewer rows than one chunk: a single product, no accumulator."""
    x = _block(n=_GRAM_CHUNK_ROWS, d=5)
    xt = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    got = _device_gram(xt).cpu().numpy()
    assert np.array_equal(got, (xt.t() @ xt).double().cpu().numpy())
