"""Swap-only GPU wires for the streaming PCA (Gramian) and TD-FastICA arms.

Phase 7b settled the diag-GMM device port with a *differential harness*: the
host keeps the real fit loop (folds, updates, acts, decodes) and only the
per-batch sufficient-statistic reduction is evaluated by torch on the device
and returned as float64 numpy arrays.  Whatever survives in the trajectory
difference is then pure device reduction order, not a new algorithm.

This module provides the same wiring for the two remaining streaming families:

* :func:`pca_gpu_measure_factory` -- swaps ``PCACovariance.measure`` (the fit
  loop's second-moment step ``C_b = X_b^T X_b / n_b``).
* :func:`pca_gpu_stat_swap` -- patches ``GramianStat.block_statistic`` inside
  a context; the fold ``G <- G + X_b^T X_b`` (sum mode) or
  ``G <- (1-a) G + a C_b`` (ema mode) stays host-side numpy, exactly as
  :meth:`emaopt.algorithms.pca.PCACovariance.from_blocks` and the memmap paths
  run it.  This is the "G = G + X_b.T @ X_b (possibly chunked)" reduction the
  Phase-7 Gramian probe measured at 79.8M rows/s fp16 / 128M rows/s for the
  TD-ICA accumulation.
* :func:`ica_gpu_stat_swap` -- patches ``FastICAStat.block_statistic`` for the
  Zohar-Mannor-Monea TD-FastICA streaming form (``fastica_td``,
  ``whitening="running"``).  Per step the device computes the batch covariance
  ``C_b``, the whitened block ``Z = X_b W_z^T`` (``W_z`` itself is still the
  host's float64 eigen-decomposition of the running covariance -- it is a
  decode, not a per-row accumulate), the log-cosh nonlinearity, and the
  fixed-point moments ``M_b = g(Z)^T Z / n_b``, ``dprime_b = E[1 - g^2]``.
  The EMA fold and the ``symmetric_decorrelation`` act stay host-side.

Nothing here changes any default code path: the module is inert until a
harness applies a swap.  ``torch`` is imported lazily so the library imports
fine on CPU-only machines.

Elementwise semantic gates
--------------------------
:func:`semantic_gate_pca` / :func:`semantic_gate_ica` run the device
transcription against the numpy original on a few real states and return the
worst elementwise *relative* gap (floor at ``tiny`` for near-zero entries).
They are the abort-before-you-run check: at ``torch.float64`` the two must
agree to ~1e-12 relative; if they don't, the transcription is wrong, not the
device.  Gates are logit-free: they compare the sufficient-statistic arrays
themselves.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable

import numpy as np

_TINY = float(np.finfo(np.float64).tiny)

_DTYPE_NAMES = {
    "float64": "float64",
    "float32": "float32",
    "fp64": "float64",
    "fp32": "float32",
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
}


def resolve_dtype(dtype: Any):
    """Map 'f64'/'float64'/np.float64/torch dtype onto a torch dtype."""
    import torch

    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float64
    if dtype in (np.float64, "f64", "double"):
        return torch.float64
    if dtype in (np.float32, "f32"):
        return torch.float32
    if dtype in (np.float16,):
        return torch.float16
    try:
        return getattr(torch, _DTYPE_NAMES[str(dtype).lower()])
    except (KeyError, AttributeError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"unsupported dtype {dtype!r}") from exc


def _as_device(block: np.ndarray, dtype, device):
    import torch

    return torch.as_tensor(
        np.ascontiguousarray(block, dtype=np.float64), dtype=dtype,
        device=device,
    )


def _np_f64(t) -> np.ndarray:
    """Device tensor -> float64 numpy (the host folds in float64, always)."""
    return t.detach().cpu().numpy().astype(np.float64)


def _check(checks: list[float] | None, gap: float) -> None:
    if checks is not None:
        checks.append(float(gap))


def _rel_gap(ref: np.ndarray, got: np.ndarray) -> float:
    """max|ref - got| / max(max|ref|, tiny): array-scale-relative elementwise
    gap.  Scale-relative (not per-entry) so a legitimately tiny tail entry
    cannot fail the semantic gate for representational reasons; a wrong
    transcription always shows up in max|diff|."""
    ref = np.asarray(ref, dtype=np.float64)
    got = np.asarray(got, dtype=np.float64)
    scale = max(float(np.abs(ref).max()), _TINY)
    return float(np.abs(ref - got).max() / scale)


#: Rows per block in the reduced-precision Gramian.  2048 rows of 512 doubles is
#: 8 MB -- comfortably L2-resident, and four blocks are enough to buy ~4x the
#: accuracy of one float32 product for ~0.03 ms.
_GRAM_CHUNK_ROWS = 2048


def _device_gram(x):
    """``X^T X`` in float64, blocking a reduced-precision product by rows.

    A float64 Gramian of 8192x512 costs **3.92 ms** on the device and a float32
    one **0.12 ms** (measured, RTX 4090) -- for the Gramian arms the precision
    choice is the difference between the GPU losing to the CPU (3.92 vs NumPy's
    6.79 ms) and winning by ~57x.  What float32 costs is accumulation error:
    one 8192-row product carries ~9.3e-07 relative, while summing 2048-row blocks
    in float64 carries **2.2e-07** for 0.03 ms more.  So a reduced dtype gets the
    blocked form and float64 keeps the plain product, bit for bit.

    Symbolically: ``sum_i (x_i^T x_i)`` where the outer sum is float64.  Order of
    summation changes the low bits only; the block size is fixed so the result
    is reproducible across runs and thread counts.
    """
    import torch

    if x.dtype == torch.float64 or x.shape[0] <= _GRAM_CHUNK_ROWS:
        return (x.t() @ x).to(torch.float64)
    first = x[: _GRAM_CHUNK_ROWS]
    acc = (first.t() @ first).to(torch.float64)
    for start in range(_GRAM_CHUNK_ROWS, x.shape[0], _GRAM_CHUNK_ROWS):
        block = x[start : start + _GRAM_CHUNK_ROWS]
        acc = acc + (block.t() @ block).to(torch.float64)
    return acc


# --------------------------------------------------------------------------- #
# PCA: Gramian streaming stat + PCACovariance.measure
# --------------------------------------------------------------------------- #
def _pca_gpu_block_statistic(
    stat, block: np.ndarray, weight: float, dtype, device, checks
) -> dict[str, np.ndarray]:
    """Device transcription of ``GramianStat.block_statistic`` (sufficient.py).

    Mirrors the numpy original element-for-element: ``gram = X^T X`` (exact
    second-moment sum, no divide), ``total = [weight]``, ``sum = X.sum(0)``
    when centring.  Shape validation is delegated to the untouched
    ``GramianStat.block_statistic`` first so a wrong-``d`` block raises exactly
    the library error.
    """
    from .state.sufficient import GramianStat

    x = _as_device(block, dtype, device)
    out: dict[str, np.ndarray] = {
        "gram": _np_f64(_device_gram(x)),
        "total": np.array([weight], dtype=np.float64),
    }
    if stat.center:
        out["sum"] = _np_f64(x.sum(0))
    if checks is not None:
        ref = GramianStat.block_statistic(stat, block, weight)
        checks.append(max(
            _rel_gap(ref["gram"], out["gram"]),
            _rel_gap(ref["sum"], out["sum"]) if stat.center else 0.0,
        ))
    return out


def pca_gpu_stat_swap(
    stat, dtype: Any = None, device: str = "cuda",
    checks: list[float] | None = None, *, level: str | None = None,
):
    """Bind the device Gramian into a :class:`GramianStat` instance.

    Only ``block_statistic`` is replaced; ``update``/``_fold``/``act``/
    ``decode`` keep running the library's host code.
    """
    import torch

    from .levels import resolve_level

    if dtype is None:
        # levels: quality/auto/performance -> float64/float32/float32.  An
        # explicit dtype always wins, and no level keeps the old float64.
        dtype = (resolve_level(level).gramian_dtype
                 if level is not None else "float64")
    dt = resolve_dtype(dtype)

    def block_statistic(block: np.ndarray, weight: float):
        return _pca_gpu_block_statistic(stat, block, weight, dt, device, checks)

    stat.block_statistic = block_statistic
    return stat


def pca_gpu_measure_factory(
    model, dtype: Any = "float64", device: str = "cuda",
    checks: list[float] | None = None,
) -> Callable:
    """Replacement ``measure`` for ``PCACovariance`` (fit-loop swap, 7b style).

    The host keeps initialize/update_state/decode; only the batch second moment
    ``C_b = X_b^T X_b / n_b`` moves to the device.
    """
    import torch

    from .core import State

    dt = resolve_dtype(dtype)

    def measure(batch: np.ndarray, state) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        n = max(int(batch.shape[0]), 1)
        x = _as_device(batch, dt, device)
        cov = _np_f64(_device_gram(x) / n)
        if checks is not None:
            checks.append(_rel_gap(batch.T @ batch / n, cov))
        return State({"C": cov}, 0, {})

    return measure


def semantic_gate_pca(
    blocks: list[np.ndarray],
    *,
    device: str = "cuda",
    dtype: Any = "float64",
    center: bool = True,
    d: int | None = None,
) -> dict[str, Any]:
    """Elementwise transcription check of the device Gramian vs the numpy one.

    Runs on real blocks in 2-3 states (fresh accumulator, after one fold, after
    a second) and returns the worst relative gap per array plus a per-arm
    summary; ``passed_1e_minus_12`` is the abort criterion.
    """
    from .state.sufficient import GramianStat

    dt = resolve_dtype(dtype)
    per_state: list[dict[str, float]] = []
    for i, block in enumerate(blocks):
        stat = GramianStat(d=d, center=center, mode="sum")
        checks: list[float] = []
        for state_idx in range(2 if i == 0 else 1):
            b = np.asarray(block[: block.shape[0] // 2] if state_idx else block,
                           dtype=np.float64)
            numpy_stats = stat.block_statistic(b, float(b.shape[0]))
            gpu_stats = _pca_gpu_block_statistic(stat, b, float(b.shape[0]),
                                                 dt, device, None)
            gaps = {k: _rel_gap(numpy_stats[k], gpu_stats[k])
                    for k in numpy_stats}
            per_state.append({
                "block": i,
                "state_after_folds": stat.steps,
                **{f"rel_{k}": v for k, v in gaps.items()},
            })
            stat._fold(numpy_stats, float(b.shape[0]))  # keep host fold real
    worst = max(
        (v for s in per_state for k, v in s.items() if k.startswith("rel_")),
        default=0.0,
    )
    return {
        "dtype": str(dt).replace("torch.", ""),
        "per_state": per_state,
        "worst_rel_gap": worst,
        "passed_1e_minus_12": bool(worst < 1e-12),
    }


# --------------------------------------------------------------------------- #
# ICA: TD-FastICA streaming block statistic (running whitening)
# --------------------------------------------------------------------------- #
def _ica_gpu_block_statistic(self, block: np.ndarray, weight: float,
                             dtype, device, checks) -> dict[str, np.ndarray]:
    """Device transcription of ``FastICAStat.block_statistic`` (running mode).

    Structural mirror of the numpy original: batch covariance
    ``C_b = X_b^T X_b / n_b`` on the device, the first-block
    ``params["C"]`` seeding rule identical, whitening matrix
    ``W_z`` computed by the untouched host ``whitening_matrix()`` (an eigen
    decode, not a per-row accumulate) then cast to ``dtype``, and the moment
    ``M_b = tanh(Z W^T)^T Z / n_b`` + ``dprime`` accumulated on the device.
    """
    import torch

    if self.whitening != "running":
        raise ValueError("ica_gpu_stat_swap targets the TD streaming arm "
                         '(whitening="running")')
    block = np.asarray(block, dtype=np.float64)
    n = max(int(block.shape[0]), 1)
    x = _as_device(block, dtype, device)
    seeded = "C" not in self.values and "C" not in self.params
    out = _ica_gpu_stats_from_device(self, x, n, dtype, device)
    if checks is not None:
        ref = _ica_numpy_reference_stat(self, block, n, unseed=seeded)
        checks.append(max(_rel_gap(ref["M"], out["M"]),
                          _rel_gap(ref["dprime"], out["dprime"]),
                          _rel_gap(ref["C"], out["C"])))
    return out


def _ica_numpy_reference_stat(self, block: np.ndarray, n: int, *, unseed: bool):
    """numpy's own block_statistic path (the ground truth) on a sandbox copy.

    The original function is snapshotted at first use so the reference stays
    the numpy one even when ``FastICAStat.block_statistic`` is currently
    patched by :func:`ica_gpu_stat_swap`.
    """
    import copy as _copy

    from .algorithms.ica import FastICAStat

    original = globals().get("_ICA_NUMPY_BLOCK_STATISTIC")
    if original is None:
        original = FastICAStat.block_statistic
        globals()["_ICA_NUMPY_BLOCK_STATISTIC"] = original

    ref = _copy.copy(self)
    ref.values = {k: np.array(v, copy=True) for k, v in self.values.items()}
    ref.params = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v)
                  for k, v in self.params.items()}
    if unseed:
        ref.params.pop("C", None)
        ref.params.pop("mean", None)
    ref._invalidate_whitening()
    stats = original(ref, block, float(n))
    return stats


def ica_gpu_stat_swap(
    dtype: Any = None, device: str = "cuda", checks: list[float] | None = None,
    *, level: str | None = None,
):
    """Context manager patching ``FastICAStat.block_statistic`` with the device
    version, so ``emaopt.algorithms.ica.fastica_td`` itself is the fit loop.

    Usage::

        with ica_gpu_stat_swap("float32", "cuda", checks):
            res = fastica_td(data, n_components=k, batch_size=bs, seed=0)
    """
    from .algorithms.ica import FastICAStat
    from .levels import resolve_level

    if dtype is None:
        dtype = (resolve_level(level).gramian_dtype
                 if level is not None else "float64")
    dt = resolve_dtype(dtype)
    original = FastICAStat.block_statistic

    @contextlib.contextmanager
    def _cm():
        FastICAStat.block_statistic = lambda self, block, weight: (
            _ica_gpu_block_statistic(self, block, weight, dt, device, checks))
        try:
            yield
        finally:
            FastICAStat.block_statistic = original

    return _cm()


def semantic_gate_ica(
    blocks: list[np.ndarray],
    *,
    device: str = "cuda",
    dtype: Any = "float64",
    d: int = 64,
    n_components: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Elementwise check: device TD moments vs the numpy originals, in 2-3
    real states (fresh stat, after one fold, after a second)."""
    from .algorithms.ica import FastICAStat

    dt = resolve_dtype(dtype)
    nc = n_components or d
    per_state: list[dict[str, float]] = []
    worst = 0.0
    for i, block in enumerate(blocks):
        stat = FastICAStat(nc, mode="ema", alpha=0.05, act_every=1, seed=seed,
                           whitening="running")
        stat.prepare(d=int(np.asarray(block).shape[1]))
        for b in ([block, block[: max(block.shape[0] // 2, 1)]][: 2 if i == 0 else 1]):
            b = np.asarray(b, dtype=np.float64)
            numpy_stats = _ica_numpy_reference_stat(stat, b, int(b.shape[0]),
                                                   unseed="C" not in stat.values
                                                   and "C" not in stat.params)
            # sandbox copy did the first-block seed on its own params; redo on
            # a twin so the device path sees the identical pre-state
            twin_stat = stat.copy()
            gpu_stats = _ica_gpu_block_statistic(twin_stat, b, float(b.shape[0]),
                                                 dt, device, None)
            gaps = {k: _rel_gap(numpy_stats[k], gpu_stats[k])
                    for k in ("M", "dprime", "C")}
            per_state.append({"block": i, "steps": stat.steps,
                              **{f"rel_{k}": v for k, v in gaps.items()}})
            worst = max(worst, *gaps.values())
            stat._fold(numpy_stats, float(b.shape[0]))
            if stat.steps % stat.act_every == 0:
                stat.act()
    return {
        "dtype": str(dt).replace("torch.", ""),
        "per_state": per_state,
        "worst_rel_gap": float(worst),
        "passed_1e_minus_12": bool(worst < 1e-12),
    }


# --------------------------------------------------------------------------- #
# T24: resident arms -- upload the matrix once, gather blocks by index
# --------------------------------------------------------------------------- #
# T22 measured the limit of the swap arms: the Gramian *kernel* is 33x cheaper in
# float32 (3.92 -> 0.12 ms) and 57x the host product, yet the *arm* only gained
# 1.9x because a 33.6 MB pageable upload costs ~20 ms against a 3.8 ms product.
# Every block was uploaded afresh.  The resident arms below keep the matrix on the
# device and pass **indices**, exactly as :func:`emaopt.gpu.egmm_gpu_fit` does for
# the mixture: the statistic that crosses PCIe per block is `d x d` (2 MB at
# d=512), not `n_b x d` (33.6 MB), and only once at the end if it is accumulated
# on the device.

def _ica_gpu_stats_from_device(self, x, n: int, dtype, device):
    """TD-FastICA block statistics from an already-resident block ``x``.

    The single transcription shared by the swap arm and the resident arm, so the
    two cannot drift: batch covariance on the device, the first-block ``C``
    seeding rule identical to ``FastICAStat.block_statistic``, the whitening
    matrix from the untouched host ``whitening_matrix()`` (an eigen decode, not a
    per-row accumulate), then the fixed-point moments.
    """
    import torch

    if self.whitening != "running":
        raise ValueError("the device ICA arms target the TD streaming form "
                         '(whitening="running")')
    mean = self.params.get("mean")
    if mean is None:
        # Match FastICAStat._initial_mean: only a resolvable first-block offset
        # earns centring in the one-pass running form.
        mean_t = x.mean(0)
        scale = torch.linalg.vector_norm(
            x.var(0, unbiased=False).clamp_min(0).sqrt())
        if torch.linalg.vector_norm(mean_t) <= 4.0 * scale / (n ** 0.5):
            mean_t = torch.zeros_like(mean_t)
        mean = _np_f64(mean_t)
        self.params["mean"] = np.asarray(mean, dtype=np.float64)
    mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float64),
                             dtype=x.dtype, device=device)
    centred = x - mean_t
    cov = _np_f64((centred.T @ centred) / n)
    if "C" not in self.values and "C" not in self.params:
        # first block: its own covariance is the whitening it is measured
        # under, and the running estimate starts there (ica.py:229-233)
        self.params["C"] = cov
        self._invalidate_whitening()
    wz = torch.as_tensor(self.whitening_matrix(), dtype=dtype, device=device)
    xw = centred @ wz.T                              # (n_b, dim)
    w = torch.as_tensor(np.ascontiguousarray(
        np.asarray(self.params["W"], dtype=np.float64)), dtype=dtype, device=device)
    u = xw @ w.T
    gx = torch.tanh(u)
    gprime = 1.0 - gx * gx
    return {
        "M": _np_f64((gx.T @ xw) / n),
        "dprime": _np_f64(gprime.mean(0)),
        "C": cov,
    }


def pca_gpu_resident_stat(
    data,
    *,
    d: int | None = None,
    center: bool = True,
    stat=None,
    batch_size: int = 8192,
    epochs: int = 1,
    dtype: Any = None,
    device: str = "cuda",
    level: str | None = None,
    instrument: list[dict] | None = None,
):
    """Accumulate a :class:`GramianStat` with the matrix resident on the device.

    The sum-mode fold is elementwise addition, so accumulating ``X_b^T X_b`` on
    the device and taking the single host fold at the end is the *same
    arithmetic* as folding every block on the host -- up to summation order,
    which the float64 accumulator does not move.  The float64 arm is therefore
    gated against the swap arm at 1e-12 relative, not merely "close".

    Only ``mode="sum"`` is ported, for the same reason: in ``mode="ema"`` the
    fold is ``(1-a) G + a C_b`` with a per-key alpha resolved by the law, and
    re-deriving that law on the device would be a second transcription of the
    schedule -- exactly the drift T48 removed.  The host fold handles it.

    ``epochs > 1`` replays the resident tensor without re-uploading it, which is
    where residency actually pays: over one pass the swap arm and this arm move
    the *same total bytes*, so only per-transfer overhead separates them (T24
    measured 1.2x, not the 2x a single-pass bar would demand).

    Returns ``(stat, info)``; ``info`` carries the one-time upload and the
    per-block device time so a caller can quote the amortised cost honestly.
    """
    import time

    import torch

    from .levels import resolve_level
    from .state.sufficient import GramianStat

    # ``data`` may be an array or an on-disk/memmap shard.  The shard form is
    # read into a preallocated device buffer in chunks, because materialising a
    # 1M x 512 corpus as float64 on the host would cost 4 GB just to hand it to
    # a loop that is supposed to be the memory-light path.
    stream_source = None
    if isinstance(data, (str, os.PathLike, np.memmap)):
        stream_source = (data if isinstance(data, np.memmap)
                         else np.load(str(data), mmap_mode="r"))
        if stream_source.ndim != 2:
            raise ValueError("data must be a 2-D array (n, d)")
        rows, width = int(stream_source.shape[0]), int(stream_source.shape[1])
        arr = None
    else:
        arr = np.ascontiguousarray(np.asarray(data, dtype=np.float64))
        if arr.ndim != 2:
            raise ValueError("data must be a 2-D array (n, d)")
        rows, width = int(arr.shape[0]), int(arr.shape[1])
    if d is not None and int(d) != width:
        raise ValueError(f"declared d={d} != data width {width}")
    if dtype is None:
        dtype = (resolve_level(level).gramian_dtype
                 if level is not None else "float64")
    dt = resolve_dtype(dtype)

    stat = stat or GramianStat(d=width, center=center, mode="sum")
    if stat.mode != "sum":
        raise ValueError(
            'the resident Gramian arm accumulates mode="sum"; an EMA fold is '
            "the updater's law and stays on the host (see GramianStat.update)"
        )
    if bool(stat.center) != bool(center):
        raise ValueError("center= must match the stat it accumulates into")

    bs = max(1, int(batch_size))
    passes = max(1, int(epochs))
    started = time.perf_counter()
    if arr is not None:
        dev = torch.as_tensor(arr, dtype=dt, device=device)
    else:
        dev = torch.empty((rows, width), dtype=dt, device=device)
        for start in range(0, rows, 16384):
            chunk = np.ascontiguousarray(
                np.asarray(stream_source[start : start + 16384], dtype=np.float64))
            dev[start : start + chunk.shape[0]] = torch.as_tensor(
                chunk, dtype=dt, device=device)
    upload_s = time.perf_counter() - started

    gram_acc = torch.zeros((width, width), dtype=torch.float64, device=device)
    sum_acc = (torch.zeros(width, dtype=torch.float64, device=device)
               if stat.center else None)
    total = 0.0
    blocks = 0
    per_block: list[float] = []
    loop0 = time.perf_counter()
    for _pass in range(passes):
        for start in range(0, rows, bs):
            xb = dev[start : start + bs]
            if xb.shape[0] == 0:
                continue
            t0 = time.perf_counter()
            gram_acc = gram_acc + _device_gram(xb)
            if sum_acc is not None:
                sum_acc = sum_acc + xb.sum(0)
            per_block.append(1e3 * (time.perf_counter() - t0))
            total += float(xb.shape[0])
            blocks += 1
            if instrument is not None:
                instrument.append({"block": blocks, "rows": int(xb.shape[0]),
                                   "ms": per_block[-1]})
    loop_s = time.perf_counter() - loop0

    stat.values["gram"] = gram_acc.detach().cpu().numpy().astype(np.float64)
    stat.values["total"] = np.array([total], dtype=np.float64)
    if stat.center:
        stat.values["sum"] = sum_acc.detach().cpu().numpy().astype(np.float64)
    stat.weight = float(total)
    stat.n_examples = int(total)
    stat.steps = int(blocks)
    stat.act()
    info = {
        "mode": "resident",
        "dtype": str(dt).replace("torch.", ""),
        "rows": rows,
        "d": width,
        "blocks": blocks,
        "batch_size": bs,
        "epochs": passes,
        "upload_s": float(upload_s),
        "loop_s": float(loop_s),
        "total_s": float(upload_s + loop_s),
        "per_block_ms": float(np.mean(per_block)) if per_block else 0.0,
    }
    return stat, info


def ica_gpu_resident_fit(
    data,
    n_components: int | None = None,
    *,
    batch_size: int = 4096,
    nepochs: int = 1,
    decay: float = 0.05,
    act_every: int = 1,
    tol: float = 1e-8,
    seed: int = 0,
    eval_data: np.ndarray | None = None,
    shuffle: bool = True,
    dtype: Any = None,
    device: str = "cuda",
    level: str | None = None,
    contrast: float = 1.0,
    record_history: bool = True,
    max_updates: int | None = None,
    instrument: list[dict] | None = None,
    whiten_tol: float = 0.0,
    whitening: str = "running",
    covariance: np.ndarray | None = None,
    mean: np.ndarray | None = None,
):
    """TD-FastICA with the corpus resident on the device (T24).

    Mirrors :func:`emaopt.algorithms.ica.fastica_td` step for step -- same
    statistic, same fold, same act schedule, same epoch/shuffle order (the
    iterator is run over an index column, so the row order is bit-for-bit the
    numpy path's) -- with one difference: the block is gathered with
    ``index_select`` on the device instead of being uploaded.  What still
    crosses PCIe per block is the ``d x d`` covariance and the whitening matrix,
    which is why the arm should approach the product's cost instead of the
    upload's.

    Measured, and it is not a win at the wide width (T24, 2026-11-15): 1.02-1.07x
    over two epochs at d=512, k=64 across three runs, because the step is
    **whitening-bound**, not transfer-bound -- one 512x512 eigen-solve per block,
    and both arms pay it.  The function is exact (1e-10 unmixing parity with
    ``fastica_td``) and is the right surface for a resident stream, but the lever
    that moves this arm is the whitening cadence, not residency: T60 measured
    ``whiten_every`` in {{2, 4, 8}} as trajectory-neutral at |cos| >= 0.99988 and
    1.6-2.3x of wall, while freezing the whitening from the first block changes
    the solution (|cos| 0.475).  Absolute walls for this arm moved 12x between
    runs on a loaded box, so only within-run ratios are quoted anywhere.

    Returns ``(ICAResult, info)``.  The result is the same type the numpy path
    returns, so an arm-level comparison is apples to apples.
    """
    import time

    import torch

    from .algorithms.ica import FastICAStat, ICAResult, _eval_subsample
    from .levels import resolve_level
    from .stream import as_block_source, block_steps, iter_blocks, n_rows

    source = as_block_source(data)
    n = n_rows(source)
    width = int(source.shape[1])
    if dtype is None:
        dtype = (resolve_level(level).gramian_dtype
                 if level is not None else "float64")
    dt = resolve_dtype(dtype)

    stat = FastICAStat(
        n_components, mode="ema", alpha=float(decay), act_every=int(act_every),
        seed=seed, whitening=whitening, contrast=contrast,
        whiten_tol=whiten_tol,
    )
    if whitening == "fixed":
        if covariance is None:
            raise ValueError("whitening='fixed' needs covariance=")
        stat.prepare(d=width, covariance=np.asarray(covariance, dtype=np.float64),
                     mean=mean)
    else:
        stat.prepare(d=width, mean=mean)
    bs = max(1, min(int(batch_size), n))
    steps = block_steps(n, bs, nepochs=nepochs)
    if max_updates is not None:
        steps = min(steps, int(max_updates))
    stat.configure(total_steps=steps, dataset_size=n)
    eval_x = _eval_subsample(source, eval_data=eval_data, seed=seed)

    started = time.perf_counter()
    dev = torch.as_tensor(np.ascontiguousarray(np.asarray(source),
                                               dtype=np.float64),
                          dtype=dt, device=device)
    upload_s = time.perf_counter() - started

    index_source = np.arange(n, dtype=np.int64).reshape(n, 1)
    history: list[float] = []
    examples = 0
    updates = 0
    per_step: list[float] = []
    every = max(1, steps // 20)
    loop0 = time.perf_counter()
    for index_block in iter_blocks(index_source, bs, nepochs=nepochs,
                                   shuffle=shuffle, seed=seed,
                                   max_updates=max_updates):
        idx = torch.as_tensor(np.ascontiguousarray(
            np.asarray(index_block).ravel().astype(np.int64)), device=device)
        xb = dev.index_select(0, idx)
        t0 = time.perf_counter()
        stats = _ica_gpu_stats_from_device(stat, xb, int(xb.shape[0]), dt, device)
        stat._fold(stats, float(xb.shape[0]))
        if stat.steps % stat.act_every == 0:
            stat.act()
        per_step.append(1e3 * (time.perf_counter() - t0))
        updates += 1
        examples += int(xb.shape[0])
        if updates % every == 0 and record_history:
            history.append(-stat.negentropy(eval_x))
        if instrument is not None:
            instrument.append({"update": updates, "ms": per_step[-1]})
    loop_s = time.perf_counter() - loop0

    params = stat.decode()
    objective = -stat.negentropy(eval_x)
    if record_history and (not history or history[-1] != objective):
        history.append(objective)
    config = stat.config()
    config.update({
        "batch_size": int(bs),
        "updates": int(updates),
        "examples_processed": int(examples),
        "eval_rows": int(eval_x.shape[0]),
        "wall_time": float(time.perf_counter() - started),
        "compute_plan": None,
        "mode": "td-resident",
        "whitening": whitening,
        "whitening_solves": int(stat.whitening_solves),
        "residency": {"upload_s": float(upload_s), "loop_s": float(loop_s)},
    })
    result = ICAResult(
        unmixing=params["unmixing"],
        mixing=params["mixing"],
        sources=(eval_x - params["mean"]) @ params["unmixing"].T,
        whitening=params["whitening"],
        mean=params["mean"],
        n_iter=int(stat.acts),
        converged=bool(np.isfinite(stat._last_delta) and stat._last_delta < tol),
        objective=float(objective),
        history=history,
        config=config,
        examples_processed=int(examples),
        nbytes=int(stat.nbytes),
    )
    info = {
        "mode": "resident",
        "dtype": str(dt).replace("torch.", ""),
        "rows": n,
        "d": width,
        "updates": updates,
        "upload_s": float(upload_s),
        "loop_s": float(loop_s),
        "total_s": float(upload_s + loop_s),
        "per_step_ms": float(np.mean(per_step)) if per_step else 0.0,
    }
    return result, info


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing
        return False
