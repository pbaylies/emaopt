"""Optional numba backend for the host E-step and e-means epoch kernels.

This is a third implementation of the row-shaped kernels that already have a C
artifact -- the diagonal log-density, the row log-sum-exp, and the fused
diagonal E-step -- plus an epoch runner for e-means.  It is opt-in twice over:
``pip install "emaopt[numba]"`` supplies the dependency, and ``EMAOPT_NUMBA=1``
asks for it at run time.  A fit never silently pays for a JIT compile.

Why a JIT is reasonable here
----------------------------
The row kernels are parallel maps: one output per row, no cross-row
reduction, so ``prange`` cannot change the answer.  The fused E-step writes the
responsibility matrix in the same pass as the softmax and then uses BLAS for the
three sufficient-statistic reductions; the missing piece in the old backend was
the absence of that path, not a different arithmetic contract.  The e-means
runner is serial by construction, because its updates have a carried dependency
through ``sums``/``counts``.

Accuracy: no ``fastmath``.  NumPy's pairwise summation differs from a serial
sum, so agreement with the shipped path is to rounding, not bit for bit.  The
record in ``evidence/host_kernels.json`` is the authority on both accuracy and
which arm wins a shape.

Costs, stated rather than hidden: numba and llvmlite are a large optional
install, the first call in a process pays compilation unless ``cache=True``
writes a disk cache, and the compiled functions live in numba's cache directory
rather than beside the package.  Normal dispatch sends wide rows through the
BLAS-backed path and keeps the scalar narrow kernels off shapes where the
measured C/internal arms win; ``force=True`` is for the parity bench and tests,
not for a production default.
"""

from __future__ import annotations

import os
import threading

import numpy as np
from numba import get_num_threads, njit, prange, set_num_threads

_LOG_2PI = float(np.log(2.0 * np.pi))

#: Whether the arm was asked for.  ``EMAOPT_NUMBA=1`` enables it; unset leaves
#: it off, because a first-call compile is not something a fit should discover.
_ENABLE = {"1", "on", "true", "yes", "numba"}

#: ``n * k * d`` below which the NumPy or C arm is left in place.  The kernels
#: are threaded, and a pool launch costs more than it saves on a small batch.
_WORK_GATE = 1_000_000

#: Narrowest row the normal logits/fused dispatch takes.  The scalar narrow
#: kernel loses to NumPy/C at ``d=64``; the wide path is two BLAS GEMMs and wins
#: at ``d=320``/``512``, so the gate sends only wide rows here unless a caller
#: explicitly forces the parity path.  ``_MAX_D`` remains an alias for the
#: maintainer probes that already read it.
_WIDE_MIN_D = 128
_MAX_D = _WIDE_MIN_D

#: Per-update distance work (``batch * k * d``) above which the epoch runner
#: declines and the fit keeps its NumPy loop.  The runner is serial by
#: construction, while an update at this size is a GEMM the loop's threaded
#: BLAS wins: five epochs at eight threads, seven alternating samples --
#: 4.19M per update (8192 x 512, k=16) is 63.09 against the loop's 61.67 ms
#: (0.98x), 5.24M (8192 x 320, k=32) is 115.65 against 93.29 (0.81x), and
#: 10.49M (8192 x 320, k=64) is 190.01 against 143.29 (0.75x).  The internal
#: runner's crossover is the same place (see ``internal_kernels``), so both
#: gates use 4.5M.
_WIDE_STEP_WORK = 4_500_000

_lock = threading.Lock()
_state: dict = {"configured": False, "nodes": 1, "error": ""}


def enabled() -> bool:
    return os.environ.get("EMAOPT_NUMBA", "").strip().lower() in _ENABLE


def _configure() -> None:
    """Set numba's worker count once, from the same environment the C arm uses."""
    global _state
    if _state["configured"]:
        return
    with _lock:
        if _state["configured"]:
            return
        try:
            from .hostkernels import _threads

            nodes = _threads()
        except Exception:            # pragma: no cover - import cycle guard
            nodes = max(1, os.cpu_count() or 1)
        # ``NUMBA_NUM_THREADS`` fixes the pool at import time; asking for more
        # than that cap raises, and catching the raise in ``available()`` used to
        # disable the whole arm silently when a caller set both.  Clamp to what
        # numba will accept, and skip the call when it is already the right size.
        try:
            current = int(get_num_threads())
        except Exception:            # pragma: no cover - threading layer
            current = None
        if current is not None:
            nodes = min(int(nodes), current)
            if nodes != current:
                try:
                    set_num_threads(int(nodes))
                except ValueError:   # pragma: no cover - environment
                    nodes = current
        _state["nodes"] = int(nodes)
        _state["configured"] = True


def available() -> bool:
    """Whether the arm will run: requested, importable, and configured."""
    if not enabled():
        return False
    try:
        _configure()
    except Exception as exc:         # pragma: no cover - threading layer
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return False
    return True


def status() -> dict:
    return {"enabled": enabled(), "available": available(),
            "nodes": _state["nodes"], "error": _state["error"],
            "gate": _WORK_GATE, "wide_min_d": _WIDE_MIN_D,
            "max_d": _MAX_D, "fused_estep": True, "wide_blas": True,
            "epoch_runner": True}


@njit(cache=True, parallel=True, nogil=True)
def _logits(x, mean_t, inv_t, off, out, n, d, k):
    """``(n, k)`` diagonal-Gaussian log-density, one pass, rows in parallel.

    ``mean_t``/``inv_t`` are ``(d, k)`` so the inner ``j`` loop is contiguous;
    each row's ``s`` is an independent reduction, so threading cannot reorder
    anything across rows.
    """
    for i in prange(n):
        for j in range(k):
            s = 0.0
            for c in range(d):
                diff = x[i * d + c] - mean_t[c * k + j]
                s += diff * diff * inv_t[c * k + j]
            out[i * k + j] = off[j] - 0.5 * s


@njit(cache=True, nogil=True)
def _logits_blas(x, mean_inv_t, inv_t, quad_bias, off, out, n, d, k):
    """Wide-row log-density as two BLAS GEMMs plus an elementwise finish.

    NumPy's reference path wins at wide rows because it does not reduce over
    ``d`` once per ``(row, component)``: it rewrites the quadratic as
    ``x @ (mean/cov).T`` and ``(x*x) @ (1/cov).T``, so OpenBLAS does the
    blocked, vectorised work.  The scalar wide kernel here could not match
    that (measured 15.5 ms against 5.7 ms at 8192x320, k=64), so the wide arm
    uses the same two products and spends its compiled time on the finish.

    ``mean_inv_t`` and ``inv_t`` are ``(d, k)``; ``quad_bias`` folds the
    component-only term ``mean**2 / cov``.  No ``fastmath``: the result differs
    from NumPy only by BLAS reduction order.
    """
    a = x @ mean_inv_t
    b = (x * x) @ inv_t
    for i in range(n):
        for j in range(k):
            out[i, j] = off[j] - 0.5 * (
                b[i, j] - 2.0 * a[i, j] + quad_bias[j])


@njit(cache=True, parallel=True, nogil=True)
def _logsumexp(logits, out, n, k):
    """``out[i] = log(sum_j exp(logits[i, j]))``, row-stable and row-parallel."""
    for i in prange(n):
        mx = logits[i * k]
        for j in range(1, k):
            v = logits[i * k + j]
            if v > mx:
                mx = v
        if not np.isfinite(mx):
            mx = 0.0
        s = 0.0
        for j in range(k):
            s += np.exp(logits[i * k + j] - mx)
        out[i] = mx + np.log(s)


@njit(cache=True, parallel=True, nogil=True)
def _estep_rows(x, mean_t, inv_t, off, resp, log_norm, n, d, k):
    """Fused row pass: logits -> shifted softmax -> responsibilities -> NLL.

    The responsibility matrix doubles as the logits scratch.  The caller then
    performs the three sufficient-statistic reductions with BLAS; keeping that
    boundary means the compiled path has the same arithmetic and the same
    return contract as the NumPy and C arms without a second hand-written
    reduction tree.

    This is the narrow path.  At wide rows the long ``d`` reduction loses to
    two GEMMs, so :func:`_estep_wide` is used instead.
    """
    for i in prange(n):
        base = i * k
        for j in range(k):
            s = 0.0
            for c in range(d):
                diff = x[i * d + c] - mean_t[c * k + j]
                s += diff * diff * inv_t[c * k + j]
            resp[base + j] = off[j] - 0.5 * s

        mx = resp[base]
        for j in range(1, k):
            if resp[base + j] > mx:
                mx = resp[base + j]
        if not np.isfinite(mx):
            mx = 0.0

        total = 0.0
        for j in range(k):
            e = np.exp(resp[base + j] - mx)
            resp[base + j] = e
            total += e
        inv_total = 1.0 / max(total, 1e-300)
        row_sum = 0.0
        for j in range(k):
            value = resp[base + j] * inv_total
            if not np.isfinite(value):
                value = 0.0
            resp[base + j] = value
            row_sum += value
        inv_row = 1.0 / max(row_sum, 1e-300)
        for j in range(k):
            resp[base + j] *= inv_row
        log_norm[i] = mx + np.log(total)


@njit(cache=True, parallel=True, nogil=True)
def _estep_wide(x, mean_inv_t, inv_t, quad_bias, off, resp, log_norm,
                n, d, k):
    """Wide fused E-step: two GEMMs, then a parallel softmax pass.

    The quadratic is computed the same way as :func:`_logits_blas`; only the
    softmax/normalisation stays in the compiled row loop.  This keeps the
    BLAS-backed log-density and removes NumPy's intermediate ``e``/``resp``
    temporaries while preserving the same responsibility contract.
    """
    a = x @ mean_inv_t
    b = (x * x) @ inv_t
    for i in prange(n):
        base = i * k
        mx = -1e300
        for j in range(k):
            value = off[j] - 0.5 * (
                b[i, j] - 2.0 * a[i, j] + quad_bias[j])
            resp[base + j] = value
            if value > mx:
                mx = value
        if not np.isfinite(mx):
            mx = 0.0

        total = 0.0
        for j in range(k):
            e = np.exp(resp[base + j] - mx)
            resp[base + j] = e
            total += e
        inv_total = 1.0 / max(total, 1e-300)
        row_sum = 0.0
        for j in range(k):
            value = resp[base + j] * inv_total
            if not np.isfinite(value):
                value = 0.0
            resp[base + j] = value
            row_sum += value
        inv_row = 1.0 / max(row_sum, 1e-300)
        for j in range(k):
            resp[base + j] *= inv_row
        log_norm[i] = mx + np.log(total)


@njit(cache=True, nogil=True)
def _epoch_tiled(data, order, alphas, sums, counts, cen_t, fallback, acc,
                 final_sums, final_counts, d, k, batch, steps, step0,
                 do_final, revival_threshold, step_inertia):
    """One e-means epoch with internal's flat-array, transposed-centroid contract.

    decode, fold, assign, accumulate, optional finalize pass, then stop at
    the first dead cluster so the caller can revive it.  It is serial because
    every update depends on the previous ``sums``/``counts``.
    """
    for s in range(steps):
        a = alphas[step0 + s]
        for j in range(k):
            cnt = counts[j]
            if cnt > 1e-12:
                inv = 1.0 / cnt
                for c in range(d):
                    cen_t[c * k + j] = sums[j * d + c] * inv
            else:
                for c in range(d):
                    cen_t[c * k + j] = fallback[j * d + c]

        for j in range(k):
            counts[j] = (1.0 - a) * counts[j]
            for c in range(d):
                sums[j * d + c] = (1.0 - a) * sums[j * d + c]

        inertia = 0.0
        for i in range(batch):
            row = order[s * batch + i]
            for j in range(k):
                acc[j] = 0.0
            for c in range(d):
                xv = data[row * d + c]
                for j in range(k):
                    diff = xv - cen_t[c * k + j]
                    acc[j] += diff * diff
            best = 0
            bestd = acc[0]
            for j in range(k):
                if acc[j] < bestd:
                    bestd = acc[j]
                    best = j
            inertia += bestd
            counts[best] += a
            for c in range(d):
                sums[best * d + c] += a * data[row * d + c]
        step_inertia[0] = inertia

        if do_final == 1:
            for j in range(k):
                cnt = counts[j]
                if cnt > 1e-12:
                    inv = 1.0 / cnt
                    for c in range(d):
                        cen_t[c * k + j] = sums[j * d + c] * inv
                else:
                    for c in range(d):
                        cen_t[c * k + j] = fallback[j * d + c]
            for i in range(batch):
                row = order[s * batch + i]
                for j in range(k):
                    acc[j] = 0.0
                for c in range(d):
                    xv = data[row * d + c]
                    for j in range(k):
                        diff = xv - cen_t[c * k + j]
                        acc[j] += diff * diff
                best = 0
                bestd = acc[0]
                for j in range(k):
                    if acc[j] < bestd:
                        bestd = acc[j]
                        best = j
                final_counts[best] += 1.0
                for c in range(d):
                    final_sums[best * d + c] += data[row * d + c]

        if revival_threshold >= 0.0:
            for j in range(k):
                if counts[j] < revival_threshold:
                    return s + 1
    return steps


class _EpochRunner:
    """Adapter for ``KMeans``'s compiled-epoch protocol."""

    tiled = True

    def ready(self, n, d, k, batch_size, per_epoch, total_steps) -> bool:
        if batch_size * k * d > _WIDE_STEP_WORK:
            return False
        return per_epoch >= 1 and batch_size >= 1

    def call(self, args, *, tiled: bool = True) -> int:
        if not tiled:
            raise RuntimeError("numba epoch runner only provides the tiled form")
        return int(_epoch_tiled(*args))


_runner: _EpochRunner | None = None


def epoch_runner() -> _EpochRunner | None:
    """The opt-in epoch runner, or ``None`` when the arm is off."""
    global _runner
    if not available():
        return None
    if _runner is None:
        _runner = _EpochRunner()
    return _runner


def warm() -> None:
    """Compile all kernels on a tiny problem (timings must not include it)."""
    _configure()
    _logits(np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64),
            np.ones(4, dtype=np.float64), np.zeros(2),
            np.empty(2), 2, 2, 1)
    _logits_blas(np.zeros((2, 2), dtype=np.float64),
                 np.zeros((2, 2), dtype=np.float64),
                 np.ones((2, 2), dtype=np.float64), np.zeros(2), np.zeros(2),
                 np.empty((2, 2)), 2, 2, 2)
    _logsumexp(np.zeros(4), np.empty(2), 2, 2)
    _estep_rows(np.zeros(4, dtype=np.float64), np.zeros(4, dtype=np.float64),
                np.ones(4, dtype=np.float64), np.zeros(2),
                np.empty(2), np.empty(2), 2, 2, 1)
    _estep_wide(np.zeros((2, 2), dtype=np.float64),
                np.zeros((2, 2), dtype=np.float64),
                np.ones((2, 2), dtype=np.float64), np.zeros(2), np.zeros(2),
                np.empty(4), np.empty(2), 2, 2, 2)
    data = np.zeros(2, dtype=np.float64)
    order = np.zeros(2, dtype=np.int64)
    _epoch_tiled(data, order, np.ones(1), np.zeros(1), np.ones(1),
                 np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1),
                 np.zeros(1), 1, 1, 2, 1, 0, 0, -1.0, np.zeros(1))


def logits(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
           weights: np.ndarray | None, work: int, *,
           force: bool = False) -> np.ndarray | None:
    """``(n, k)`` log-densities, or ``None`` when this arm should not run.

    ``force`` bypasses the work/width dispatch gates for the parity bench.  It
    does not bypass ``EMAOPT_NUMBA``; an unset environment still means no arm.
    """
    if not available():
        return None
    if not force and (work < _WORK_GATE or np.shape(x)[-1] <= _WIDE_MIN_D):
        return None
    try:
        x = np.ascontiguousarray(x, dtype=np.float64)
        means = np.ascontiguousarray(means, dtype=np.float64)
        covs = np.ascontiguousarray(covs, dtype=np.float64)
        n, d = x.shape
        k = means.shape[0]
        off = -0.5 * (d * _LOG_2PI + np.log(covs).sum(axis=1))
        if weights is not None:
            off = off + np.log(np.maximum(
                np.asarray(weights, dtype=np.float64), 1e-300))
        out = np.empty((n, k), dtype=np.float64)
        if d > _WIDE_MIN_D:
            mean_inv_t = np.ascontiguousarray((means / covs).T)
            inv_t = np.ascontiguousarray((1.0 / covs).T)
            quad_bias = np.ascontiguousarray(
                (means * (means / covs)).sum(axis=1))
            _logits_blas(x, mean_inv_t, inv_t, quad_bias,
                         np.ascontiguousarray(off), out, n, d, k)
        else:
            mean_t = np.ascontiguousarray(means.T).reshape(-1)
            inv_t = np.ascontiguousarray((1.0 / covs).T).reshape(-1)
            _logits(x.reshape(-1), mean_t, inv_t, np.ascontiguousarray(off),
                    out.reshape(-1), n, d, k)
    except Exception as exc:         # pragma: no cover - compile/threading
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return None
    return out


def estep(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
          weights: np.ndarray | None = None, *, want_resp: bool = False,
          force: bool = False) -> dict | None:
    """One fused E-step, or ``None`` when this arm should not run.

    The row pass is compiled; the three statistic reductions are BLAS over the
    responsibility matrix.  ``force=True`` is for the parity bench and tests.
    """
    if not available():
        return None
    work = int(np.shape(x)[0]) * int(np.shape(means)[0]) * int(np.shape(means)[1])
    if not force and (work < _WORK_GATE or np.shape(x)[-1] <= _WIDE_MIN_D):
        return None
    try:
        x = np.ascontiguousarray(x, dtype=np.float64)
        means = np.ascontiguousarray(means, dtype=np.float64)
        covs = np.ascontiguousarray(covs, dtype=np.float64)
        n, d = x.shape
        k = means.shape[0]
        if n == 0:
            return {"counts": np.zeros(k), "sums": np.zeros((k, d)),
                    "sq": np.zeros((k, d)), "nll": 0.0}
        off = -0.5 * (d * _LOG_2PI + np.log(covs).sum(axis=1))
        if weights is not None:
            off = off + np.log(np.maximum(
                np.asarray(weights, dtype=np.float64), 1e-300))
        resp = np.empty((n, k), dtype=np.float64)
        log_norm = np.empty(n, dtype=np.float64)
        if d > _WIDE_MIN_D:
            _estep_wide(x,
                        np.ascontiguousarray((means / covs).T),
                        np.ascontiguousarray((1.0 / covs).T),
                        np.ascontiguousarray((means * (means / covs)).sum(axis=1)),
                        np.ascontiguousarray(off), resp.reshape(-1), log_norm,
                        n, d, k)
        else:
            _estep_rows(x.reshape(-1),
                        np.ascontiguousarray(means.T).reshape(-1),
                        np.ascontiguousarray((1.0 / covs).T).reshape(-1),
                        np.ascontiguousarray(off), resp.reshape(-1), log_norm,
                        n, d, k)
        counts = resp.sum(axis=0)
        sums = resp.T @ x
        sq = resp.T @ (x * x)
        out = {"counts": counts, "sums": sums, "sq": sq,
               "nll": float(-log_norm.mean())}
        if want_resp:
            out["resp"] = resp
        return out
    except Exception as exc:         # pragma: no cover - compile/threading
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return None


def logsumexp(logits_matrix: np.ndarray) -> np.ndarray | None:
    """Row log-sum-exp, or ``None`` when this arm should not run."""
    if not available():
        return None
    try:
        rows, cols = logits_matrix.shape
        out = np.empty(rows, dtype=np.float64)
        _logsumexp(np.ascontiguousarray(logits_matrix).reshape(-1), out,
                   rows, cols)
    except Exception as exc:         # pragma: no cover - compile/threading
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return None
    return out
