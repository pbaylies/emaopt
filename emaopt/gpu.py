"""GPU device arm for the streaming diagonal-GMM E-step (eGMM stage).

A supported promotion of the Phase 7b differential harness
(a phase-7b study), which proved that swapping *only*
:meth:`GaussianMixture.measure <emaopt.algorithms.gmm.GaussianMixture.measure>`
for a ``torch.float64`` GPU E-step leaves the real fit loop untouched: the
elementwise transcription gate against :meth:`component_log_prob
<emaopt.algorithms.gmm.GaussianMixture.component_log_prob>` sits at ~2.7e-12
nats, and the 120-step LAION-shape trajectory drift stayed flat at <=1.6e-13
with a bit-identical final NLL (recorded).

Usage::

    from emaopt import GaussianMixture
    from emaopt.gpu import egmm_gpu_measure          # or emaopt.egmm_gpu_measure
    model = GaussianMixture(16, covariance="diag", seed=0)
    model.measure = egmm_gpu_measure(model, device="cuda")
    result = model.fit(x, batch_size=8192, nepochs=5)

When the *whole loop* should stay on the device -- fold, decode and objective
included -- use :func:`egmm_gpu_fit`.  That is a different contract: it is
faster and it is not bit-identical to the host arm (see its docstring).

Design contract:

* Nothing in this module is reachable from the library default path.  The
  package ``__init__`` exports it lazily (PEP 562) so torch is never imported
  unless a caller asks for the GPU arm, and ``GaussianMixture`` is untouched:
  the wiring is per-instance assignment of ``measure``.
* Only the E-step runs on-device at ``dtype``; the fold/update math stays on
  host in float64 (the updater is shared code), and the sufficient statistics
  are returned as float64 numpy arrays, exactly the shape the CPU
  ``statistics_from`` produces.  That is the honest "what does a
  low-precision E-step cost" question the Phase 7b ladder asks.
* The transcription mirrors the harness: broadcast-form quadratic
  ``sum_d (x_d - mu_jd)^2 / var_jd`` (an exact rearrangement of the CPU BLAS
  form, gate gap ~2.7e-12 nats at f64), the exp-shift softmax of
  :func:`emaopt.core.softmax_rows`, and the non-finite hygiene of
  ``statistics_from`` (gmm.py:832-833).  ``log`` of the weights is evaluated
  in float64 before the dtype cast -- a k-vector, negligible bandwidth, and
  clamping at 1e-300 in an 11-bit format would fabricate a failure mode the
  CPU arm does not have.  ``logdet`` deliberately *is* evaluated at ``dtype``:
  that is a real E-step quantity, and its behaviour on tiny variances is part
  of what the ladder measures (see the ladder study).
"""

from __future__ import annotations

import dataclasses
import math
import time
from typing import Any, Callable

import numpy as np

import torch

from .algorithms.gmm import GaussianMixture
from .core import FitConfig, FitResult, State, UpdateContext
from .levels import resolve_level
from .state import AdaptiveEMA, AffineUpdater, EMA, Replace, RunningMean

__all__ = ["egmm_gpu_measure", "egmm_gpu_measure_resident", "egmm_gpu_fit",
         "transcribe_diag_logprob"]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _diag_logits(x: "torch.Tensor", means: "torch.Tensor",
                 covs: "torch.Tensor",
                 x2: "torch.Tensor | None" = None) -> "torch.Tensor":
    """``(n, k)`` ``log N(x; mu_j, diag cov_j)`` in the tensors' dtype, BLAS form.

    The same rearrangement ``component_log_prob`` uses on the host: expanding
    ``sum_d (x_d - mu_jd)^2 / var_jd`` into two GEMMs plus a k-vector removes the
    ``n*k*d`` difference tensor the broadcast form materialises (537 MB at
    8192x16x512 in float64), and is what makes the fused loop memory-light
    instead of bandwidth-bound.  The two forms differ by round-off only
    (~1.4e-12 nats on the host, measured); the ladder's gates are set against
    the host arm, so the device loop carries that difference explicitly rather
    than claiming bit-identity to it.
    """
    d = x.shape[1]
    inv_var = 1.0 / covs
    mean_inv = means * inv_var
    # ``x * x`` is needed twice per step (this quadratic and the ``sq``
    # statistic); at float64 the caller hands the same array in and saves one
    # n x d temporary.  Computed here when not supplied, so other callers are
    # unaffected.
    quad = (x * x if x2 is None else x2) @ inv_var.T - 2.0 * (x @ mean_inv.T)
    quad = quad + (means * mean_inv).sum(1)[None, :]
    quad = quad.clamp(min=0.0)
    logdet = torch.log(covs).sum(1)[None, :]
    return -0.5 * quad - 0.5 * (d * _LOG_2PI + logdet)


def _decode_diag(counts, sums, sq, *, family: str, reg_covar: float,
                 reg_covar_relative: bool, fallback):
    """The ``diag``/``spherical`` decode of ``(counts, sums, sq)`` on device.

    Mirrors :meth:`GaussianMixture.decode` term for term (same 1e-12 count
    floor, same ``max(second - mean^2, 0)``, same floor, same ``counts/total``
    weights).  The floor is one number for the state: with
    ``reg_covar_relative`` it is ``reg_covar`` times the pooled mean variance,
    which is what the host decode computes.  Empty components take the stored
    fallback mean through an unconditional ``where`` so no step needs a
    device-to-host sync to decide whether any component is empty.
    """
    d = sums.shape[1]
    safe = counts.clamp(min=1e-12)
    means = sums / safe[:, None]
    means = torch.where((counts <= 1e-9)[:, None], fallback, means)
    second = (sq / safe[:, None] - means * means).clamp(min=0.0)
    if reg_covar_relative:
        mass = counts.sum().clamp(min=1e-12)
        mean_all = sums.sum(dim=0) / mass
        pooled = ((sq.sum(dim=0) / mass) - mean_all * mean_all).clamp(min=0.0)
        scale = pooled.mean()
        # Arithmetic only: a ``.as_tensor(..., device=...)`` here would be a
        # host-to-device copy, which is illegal inside a CUDA graph capture.
        # Degenerate (all-zero) statistics fall back to the absolute floor, the
        # same answer ``GaussianMixture.reg_covar_floor`` gives.
        positive = (scale > 0).to(scale.dtype)
        floor = reg_covar * (scale * positive + (1.0 - positive))
    else:
        floor = reg_covar
    if family == "spherical":
        cov = (second.mean(dim=1, keepdim=True) + floor).expand(-1, d)
    else:
        cov = second + floor
    weights = counts / counts.sum().clamp(min=1e-12)
    return means, cov, weights


def _fold_alphas(model: GaussianMixture, ctx: UpdateContext,
                 keys: tuple[str, ...], aux: dict) -> dict[str, float]:
    """Per-key ``a`` such that the fold is ``S <- (1 - a) S + a s``.

    The device loop folds on the GPU, so it cannot hand tensors to the host
    updaters (they are NumPy arrays by contract).  Instead it reproduces the
    *algebra* of the laws that are affine blends, reading every configuration
    value from the updater object itself -- schedules, ``per_key`` overrides,
    ``amin``/``amax``/``prior_*`` -- so no constant is re-derived here.

    Supported: ``Replace`` (``a=1``), ``EMA`` (any schedule, including
    ``per_key``), ``RunningMean`` (``a=1/(n+1)``) and ``AdaptiveEMA``
    (``a=b/(b+n_eff)`` with its own ``n_eff`` recursion).  Everything else
    raises: a law that is not an affine blend of the new statistic (momentum,
    plateau halving, the rate portfolio, the debiased and Robbins-Monro forms)
    has to stay on the host path, and saying so is cheaper than approximating
    it.  Equivalence is measured, not assumed: ``tests/test_gpu_resident_fit.py``
    checks this dispatch against the implied blend of the host updater.
    """
    updater = model.updater
    if isinstance(updater, Replace):
        return {key: 1.0 for key in keys}
    if isinstance(updater, AffineUpdater):
        # One fold for every affine law (T46).  The host fold and this one used to
        # be separate transcriptions, which is why three affine laws --
        # RobbinsMonro, WeightedMean, BiasCorrectedEMA -- were rejected here
        # without there being anything non-affine about them.
        alphas, _ = updater.blend(ctx, keys, aux)
        return {key: float(alphas[key]) for key in keys}
    raise NotImplementedError(
        "the device-resident fold reproduces the affine blend laws "
        f"(Replace, EMA, RunningMean, AdaptiveEMA); got {type(updater).__name__}. "
        "Use the host fit() for this updater."
    )


def _batch_slices(n: int, batch_size: int, nepochs: int, seed: int,
                  shuffle: bool, drop_last: bool, max_updates: int | None):
    """``(epoch, i, order, start, size)`` -- the library's batch stream.

    ``core._iter_batches`` slices ``data[idx]`` on the host; a device-resident
    loop must not (that gather is ~10 ms/step at the LAION shape, measured in the
    Phase-10 probe).  So this exposes the *order* instead of the rows: ``order``
    is the epoch's permutation, ``start``/``size`` locate the batch inside it,
    and the loop uploads ``order`` once per epoch and slices it on the device.
    Same stream, no host gather, and no per-step pageable copy (which would also
    stall the stream at every step).  ``tests/test_gpu_resident_fit.py`` pins the
    stream against ``core._iter_batches`` row for row.
    """
    from .plan import plan_batches

    per_epoch = plan_batches(n, batch_size, nepochs, drop_last=drop_last).per_epoch
    rng = np.random.default_rng(seed)
    emitted = 0
    for epoch in range(nepochs):
        order = rng.permutation(n) if shuffle else np.arange(n)
        for i in range(per_epoch):
            start = i * batch_size
            size = int(order[start : start + batch_size].size)
            if size == 0:
                continue
            yield epoch, i, order, start, size
            emitted += 1
            if max_updates is not None and emitted >= max_updates:
                return


def transcribe_diag_logprob(x, means, covs):
    """``(n, k)`` ``log N(x; mu_j, diag cov_j)`` in the tensors' dtype.

    Same expression the harness gated elementwise against
    ``GaussianMixture.component_log_prob`` (f64 gap ~1.6e-12 nats); the
    broadcast form is chosen over the CPU's BLAS rearrangement because it
    degrades honestly at reduced precision (no cancellation between the
    three expanded products).
    """
    d = x.shape[1]
    logdet = torch.log(covs).sum(1)                       # (k,)
    diff = x[:, None, :] - means[None, :, :]             # (n, k, d)
    quad = ((diff * diff) / covs[None, :, :]).sum(2)     # (n, k)
    return -0.5 * (d * _LOG_2PI + logdet[None, :] + quad)


def egmm_gpu_measure(
    model: GaussianMixture,
    device: Any = None,
    *,
    dtype: torch.dtype = torch.float64,
    instrument: list[dict] | None = None,
) -> Callable[[np.ndarray, State], State]:
    """Build a GPU ``measure`` for the diagonal eGMM arm of ``model``.

    Assign it as ``model.measure``; the fit loop, updater and decode stay on
    the shared host code path.  ``dtype`` selects the device precision of the
    E-step (float64 reproduces the Phase 7b differential result; float32 /
    float16 are the ladder arms).  ``instrument``, when given a list, gets one
    dict per call with transfer/compute/download timings and non-finite
    diagnostics -- used by the ladder study to answer *how* a
    reduced-precision arm breaks, not just *whether*.
    """
    if not isinstance(model, GaussianMixture):
        raise TypeError("egmm_gpu_measure takes a GaussianMixture")
    if model.covariance not in ("diag", "spherical"):
        raise ValueError(
            "the GPU arm is only wired for the diagonal families "
            f"('diag'/'spherical'); got covariance={model.covariance!r}"
        )
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"dtype must be a floating torch dtype, got {dtype!r}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        count = torch.cuda.device_count()
        if (not torch.cuda.is_available() or count == 0
                or (device.index is not None and device.index >= count)):
            raise RuntimeError(
                f"CUDA device {device!r} requested but not available "
                f"({count} visible CUDA device(s))"
            )
    tiny = torch.finfo(dtype).tiny
    # float64: keep the harness constants bit-exactly (2.2e-308 / 1e-300);
    # reduced dtypes: 1e-300 is a literal 0 there, so clamp at the format tiny.
    rowsum_floor = max(1e-300, tiny)

    def measure(batch: np.ndarray, state: State) -> State:
        params = model.decode(state)  # host, float64, shared code
        x_np = np.ascontiguousarray(batch, dtype=np.float64)
        use_events = device.type == "cuda"
        if use_events:
            t0 = (torch.cuda.Event(True), torch.cuda.Event(True))
            t0[0].record()
        x = torch.as_tensor(x_np, dtype=dtype, device=device)
        mu = torch.as_tensor(params["means"], dtype=dtype, device=device)
        cov = torch.as_tensor(params["covariances"], dtype=dtype, device=device)
        if use_events:
            t0[1].record()
            ev_compute = torch.cuda.Event(True)
            ev_compute.record()

        logw = torch.log(
            torch.clamp(
                torch.as_tensor(params["weights"], dtype=torch.float64,
                                device=device),
                min=1e-300,
            )
        ).to(dtype)
        logits = transcribe_diag_logprob(x, mu, cov) + logw[None, :]

        # softmax_rows with exp-shift, then statistics_from's hygiene
        m = logits.max(1, keepdim=True).values
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        e = torch.exp(logits - m)
        s = e.sum(1, keepdim=True)
        r = e / torch.clamp(s, min=tiny)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        r = r / torch.clamp(r.sum(1, keepdim=True), min=rowsum_floor)
        log_norm = m + torch.log(s)                      # (n, 1)

        counts = r.sum(0)
        sums = r.T @ x
        sq = r.T @ (x * x)
        if use_events:
            ev_compute_done = torch.cuda.Event(True)
            ev_compute_done.record()
            # CUDA is async: the three statistics reductions were only
            # *launched*.  Without this wait the host timer below bills their
            # execution to the download (T41), and the largest segment of the
            # step is reported as transfer.
            ev_compute_done.synchronize()
        payload_mb = x.numel() * x.element_size() / 1e6
        t_dl0 = time.perf_counter()
        stats = {
            "counts": counts.cpu().numpy().astype(np.float64),
            "sums": sums.cpu().numpy().astype(np.float64),
            "sq": sq.cpu().numpy().astype(np.float64),
        }
        download_s = time.perf_counter() - t_dl0
        if use_events:
            ev_done = torch.cuda.Event(True)
            ev_done.record()
            ev_done.synchronize()
        nll = -float(log_norm.to(torch.float64).mean().item())
        if instrument is not None:
            bad_stats = int(
                sum(
                    int((~torch.isfinite(v)).sum().item())
                    for v in (counts, sums, sq)
                )
            )
            entry: dict[str, Any] = {
                "step": int(state.step),
                "payload_mb": payload_mb,
                "nll": nll,
                "stats_nonfinite": bad_stats,
                "resp_nonfinite": int((~torch.isfinite(r)).sum().item()),
                "logdet_nonfinite_components": int(
                    (~torch.isfinite(torch.log(cov).sum(1))).sum().item()
                ),
                "cov_min": float(cov.to(torch.float64).min().item()),
            }
            if use_events:
                entry["transfer_ms"] = t0[0].elapsed_time(t0[1])
                entry["compute_ms"] = ev_compute.elapsed_time(ev_compute_done)
            entry["download_ms"] = 1e3 * download_s
            instrument.append(entry)
        return State(stats, 0, {"nll": nll})

    return measure


def egmm_gpu_measure_resident(
    model: GaussianMixture,
    x_full: np.ndarray,
    device: Any = None,
    *,
    dtype: torch.dtype = torch.float64,
    instrument: list[dict] | None = None,
) -> Callable[[np.ndarray, State], State]:
    """Device-resident variant of :func:`egmm_gpu_measure` (Phase 10 spike).

    ``x_full`` -- the exact matrix ``fit`` will be given -- is uploaded ONCE;
    each batch is then recognized by row identity (first-column locator,
    sampled exact verification) and gathered on device, so the per-step H2D
    payload -- ~70 % of the transfer arm's step time (§11.2 instruments) --
    disappears.  Everything else is identical to ``egmm_gpu_measure``, and
    because the gathered rows are bit-identical to the uploaded ones, the
    resulting trajectory must equal the transfer arm's step for step; that
    equality is the promotion gate (tests/test_gpu_arm.py).

    Requires row-unique data (CLIP embeddings qualify); a locator collision
    raises, never silently mis-gathers.
    """
    if not isinstance(model, GaussianMixture):
        raise TypeError("egmm_gpu_measure_resident takes a GaussianMixture")
    if model.covariance not in ("diag", "spherical"):
        raise ValueError("only the diagonal families are wired")
    x_np = np.ascontiguousarray(np.asarray(x_full, dtype=np.float64))
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # structured KEY_COLS-column key: f16-quantized embeddings collide in a
    # few columns (measured: 2 columns -> ~5 duplicate pairs per 200k rows);
    # 16 quantized coordinates identify the shipped rows, and any two rows
    # equal on the prefix are value-identical for the E-step anyway -- the
    # per-step verification compares full rows and would catch the rest.
    KEY_COLS = 16
    key = np.empty(x_np.shape[0],
                   dtype=[(str(i), "<f8") for i in range(KEY_COLS)])
    for i in range(KEY_COLS):
        key[str(i)] = x_np[:, i]
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    dup_groups = int(x_np.shape[0] - np.unique(key).shape[0])
    if dup_groups:
        # f16-quantized embeddings share 16-column prefixes (measured ~455 per
        # 200k); resolution below compares FULL rows within a group, so the
        # gather stays exact. A row that is unique on its prefix resolves in
        # the fast path; only prefix-colliding batch rows pay the scan.
        pass
    x_gpu = torch.as_tensor(x_np, dtype=dtype, device=device)
    _check = {"done": False}

    def measure(batch: np.ndarray, state: State) -> State:
        b = np.ascontiguousarray(batch, dtype=np.float64)
        bk = np.empty(b.shape[0],
                      dtype=[(str(i), "<f8") for i in range(KEY_COLS)])
        for i in range(KEY_COLS):
            bk[str(i)] = b[:, i]
        pos = np.searchsorted(sorted_key, bk)
        ids = order[pos]
        # resolve prefix collisions: searchsorted returns the group's left
        # end, correct unless this row has same-prefix neighbours.
        M = sorted_key.shape[0]
        amb = np.zeros(b.shape[0], dtype=bool)
        amb[1:] = sorted_key[pos[1:] - 1] == bk[1:]          # prev same key
        nxt = pos[:-1] + 1
        nb = nxt < M
        amb[:-1][nb] |= (sorted_key[nxt[nb]] == bk[:-1][nb])  # next same key
        if amb.any():
            for r in np.nonzero(amb)[0]:
                found = -1
                p_ = pos[r]
                while p_ < x_np.shape[0] and sorted_key[p_] == bk[r]:
                    if np.array_equal(x_np[order[p_]], b[r]):
                        found = int(order[p_])
                        break
                    p_ += 1
                if found < 0:
                    raise ValueError(
                        "resident locator: batch row not found in x_full "
                        "(prefix collision exhausted)")
                ids[r] = found
        if not _check["done"]:
            if not np.array_equal(b, x_np[ids]):
                raise ValueError(
                    "resident locator verification failed: batch rows are not "
                    "rows of x_full (collisions or out-of-source data)"
                )
            _check["done"] = True
        else:  # sampled check each step, ~microseconds
            sel = np.arange(0, b.shape[0], max(b.shape[0] // 64, 1))
            if not np.array_equal(b[sel], x_np[ids[sel]]):
                raise ValueError("resident locator sampled check failed")
        t_loc0 = time.perf_counter()
        params = model.decode(state)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)
        ev0 = torch.cuda.Event(True) if device.type == "cuda" else None
        if ev0 is not None:
            ev0.record()
        x = x_gpu[ids_t]
        mu = torch.as_tensor(params["means"], dtype=dtype, device=device)
        cov = torch.as_tensor(params["covariances"], dtype=dtype, device=device)
        locator_s = time.perf_counter() - t_loc0
        logw = torch.log(
            torch.clamp(torch.as_tensor(params["weights"],
                                         dtype=torch.float64, device=device),
                        min=1e-300)
        ).to(dtype)
        logits = transcribe_diag_logprob(x, mu, cov) + logw[None, :]
        m = logits.max(1, keepdim=True).values
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        e = torch.exp(logits - m)
        s = e.sum(1, keepdim=True)
        r = e / torch.clamp(s, min=torch.finfo(dtype).tiny)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        r = r / torch.clamp(r.sum(1, keepdim=True),
                            min=max(1e-300, torch.finfo(dtype).tiny))
        log_norm = m + torch.log(s)
        # Three stages, three timers (T41).  A host timer wrapped around the
        # ``.cpu()`` calls used to include whatever was still queued on the
        # device -- and the two statistics GEMMs are the step's largest segment
        # -- so "download" was really GEMM+transfer and "compute" was only the
        # responsibilities.  Events separate them.
        if device.type == "cuda":
            ev_stats = torch.cuda.Event(True)
            ev_stats.record()
        counts_t = r.sum(0)
        sums_t = r.T @ x
        sq_t = r.T @ (x * x)
        if device.type == "cuda":
            ev_xfer = torch.cuda.Event(True)
            ev_xfer.record()
            ev_xfer.synchronize()          # idle device => honest transfer timer
        t_dl0 = time.perf_counter()
        stats = {
            "counts": counts_t.cpu().numpy().astype(np.float64),
            "sums": sums_t.cpu().numpy().astype(np.float64),
            "sq": sq_t.cpu().numpy().astype(np.float64),
        }
        download_s = time.perf_counter() - t_dl0
        if device.type == "cuda":
            compute_ms = ev0.elapsed_time(ev_stats)
            stats_ms = ev_stats.elapsed_time(ev_xfer)
        else:
            compute_ms = 0.0
            stats_ms = 0.0
        nll = -float(log_norm.to(torch.float64).mean().item())
        if instrument is not None:
            instrument.append({"step": int(state.step), "nll": nll,
                               "locator_ms": 1e3 * locator_s,
                               "compute_ms": compute_ms,
                               "stats_ms": stats_ms,
                               "download_ms": 1e3 * download_s})
        return State(stats, 0, {"nll": nll})

    return measure


def egmm_gpu_fit(
    model: GaussianMixture,
    x: np.ndarray,
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
    device: Any = None,
    dtype: torch.dtype | None = None,
    eval_mode: str = "stream",
    final_score: str = "device",
    callback: Callable[[State, UpdateContext, float], None] | None = None,
    instrument: list[dict] | None = None,
    cuda_graph: bool = False,
    stat_dtype: torch.dtype | None = None,
    level: str | None = None,
    chunk_rows: int = 65536,
) -> FitResult:
    """Run the streaming diagonal-GMM loop entirely on the device (T19).

    :func:`egmm_gpu_measure` moves *only* the E-step to the GPU: every step
    still uploads a batch, downloads the sufficient statistics, folds them into
    the state and decodes the parameters on the host, and :func:`~emaopt.core.fit`
    scores the full evaluation set once per epoch.  The Phase-10 probe measured
    what that leaves: 4.4 ms/step of round trip, ~10 ms/step of host-side
    ``data[idx]`` gather and ~1.5 s/fit of full-data objective -- none of which
    delivery tricks can remove.  This function removes all of them:

    * ``x`` is uploaded once; a step is handed **indices** and the rows are
      gathered with ``index_select`` on the device;
    * the E-step, the fold ``S <- (1-a) S + a s`` and the ``diag``/``spherical``
      decode all run on the device, so no statistic or parameter crosses PCIe
      per step;
    * the objective is **streamed** from the E-step the step already computed
      (``eval_mode="stream"``), or scored chunk-by-chunk on the device
      (``eval_mode="exact"``).

    What it gives up: the trajectory is no longer bit-identical to the host
    arm.  ``egmm_gpu_measure`` exists to answer "what does a device E-step cost"
    and therefore reproduces the host arithmetic exactly; this loop is free to
    use the host's *BLAS* rearrangement of the same quadratic form on tensor
    cores, to fold in a different summation order, and to score a streamed
    objective.  Same algorithm, same statistics, same decode -- a different
    floating-point path.  The gates that replace bit-identity are endpoint
    parity with the host arm (measured in a resident-loop study)
    and, above all, that the sequence of updates is the library's: same batch
    order, same schedule arguments, same fold algebra.

    Specifics worth knowing:

    * the fold always accumulates in float64.  ``dtype`` selects the precision
      of the E-step only (as in the measure arm): the decoded means and
      covariances are cast to it per step and the log-weights are evaluated in
      float64 before that cast, so an f16 run degrades the logits, never the
      state.  float64 is the default because it is the only setting whose
      endpoint is indistinguishable from the host arm (<=1e-12 nats); f32 is the
      endpoint-grade alternative where the covariance floor clears f32's range,
      and f16 is measured to lose (see the T19 ladder).  On Ada, f64 arithmetic
      runs at 1/64 rate, so a GEMM-heavy E-step at large ``d`` is where a
      reduced dtype could buy wall time -- measure before choosing;
    * synchronization: an uninstrumented run touches the host once at the end of
      each epoch (to read the streamed objective) and once when it finishes.
      Batch indices are uploaded once per epoch, not per step, because a
      per-step pageable copy stalls the stream every step.  CUDA events are
      allocated only when ``instrument`` is given, recorded non-blocking, and
      resolved after the final sync -- so timing a run costs one sync, not one
      per step;
    * ``stat_dtype=torch.float32`` moves *only* the two statistics GEMMs to
      float32; responsibilities, log-normalisers and the fold stay float64, so
      the error it adds is the reduction's, not the E-step's.  Measured on the
      LAION shape: 1.39x cheaper per step (1.576 -> 1.138 ms) for a 2.8e-07-nat
      endpoint drift, which is 100x finer than the f16 source data.  Off by
      default: float64 remains the setting that is indistinguishable from the
      host arm (a step-profile study);
    * ``cuda_graph=True`` captures the step -- device gather, E-step,
      statistics -- into one CUDA graph per distinct batch size and replays it.
      Same kernels, same buffers, so the captured output is **bit-identical**
      to the eager loop (asserted in the tests); the state is folded in place
      so the graph's decode always reads the live parameters.  It removes the
      launch overhead of ~40 small ops per step, measured at 8% of an isolated
      step (1.392 vs 1.518 ms at the LAION shape) -- but capture itself costs a
      sync and three E-steps per distinct batch shape (~15 ms here), so at the
      T20 shape (120 updates, two batch shapes) that cost lands inside the
      training interval: per-step train time measured 7-8% higher, while total
      wall was a wash across runs (graph 0.91-1.08 s vs eager 0.97-1.06 s).
      Swept over fit length (T21: 120, 576 and 2304 updates), end-to-end wall is
      a wash at every length (0.95-1.01x, win 1.01x at 2304), so capture never
      pays for THIS step: 1.2 of its 1.29 ms is two dgemms, and the launch
      overhead it removes is not the bottleneck.  The flag stays off by default
      and is kept as the tested answer to "is launch overhead the problem" -- it
      is not.  A graph cannot attribute time to segments, so ``cuda_graph`` and
      ``instrument`` are mutually exclusive;
    * ``a`` per key comes from the model's own updater through
      :func:`_fold_alphas` -- ``Replace``, ``EMA`` (any schedule or ``per_key``),
      ``RunningMean`` and ``AdaptiveEMA``.  Other laws raise instead of being
      approximated; call :func:`~emaopt.core.fit` for those;
    * ``finalize=True`` is supported (the last epoch's exact statistics are
      re-measured under post-update parameters on the device, as the host hook
      does).  ``revival=True`` is not: it is a host RNG walk the report measured
      as never earning its place;
    * ``fit``'s memory budget is not consulted: this *is* the streaming form,
      and the batch is exactly ``config.batch_size`` (after the identifiability
      floor), so a comparison against a host arm is a comparison at one batch.
      ``config.drop_last`` decides whether the trailing partial batch is used; a
      config you pass is honoured exactly, and the default follows
      ``GaussianMixture.fit`` (keep it except for the ``full`` family), so an
      endpoint comparison is not silently off by one update per epoch;
    * ``peak_memory`` is 0: ``tracemalloc`` does not see torch's allocator.

    ``callback`` receives a host :class:`~emaopt.core.State` per step (a
    download + decode per step -- use it for drift tracing, not for timing).
    ``instrument`` gets one dict per step with device-event segment timings.
    """
    if not isinstance(model, GaussianMixture):
        raise TypeError("egmm_gpu_fit takes a GaussianMixture")
    if model.covariance not in ("diag", "spherical"):
        raise ValueError(
            "the device-resident loop is wired for the diagonal families "
            f"('diag'/'spherical'); got covariance={model.covariance!r}"
        )
    if model.revival:
        raise NotImplementedError(
            "revival is not ported to the device loop: splitting a donor is a "
            "per-component host RNG walk over decoded statistics.  Use the "
            "host fit() for revival=True."
        )
    # ``level`` is the spectrum (quality / auto / performance, see
    # emaopt.levels): it fills whichever dtype the caller did not pass, so an
    # explicit argument always wins.  No level behaves exactly like the old
    # defaults (float64 E-step, float64 statistics).
    spec = resolve_level(level) if level is not None else None
    if dtype is None:
        dtype = spec.egmm_torch_dtype() if spec is not None else torch.float64
    if stat_dtype is None:
        stat_dtype = spec.stat_torch_dtype() if spec is not None else torch.float64
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError(f"dtype must be a floating torch dtype, got {dtype!r}")
    if eval_mode not in ("stream", "exact"):
        raise ValueError(
            f"eval_mode must be 'stream' or 'exact', got {eval_mode!r}"
        )
    if final_score not in ("device", "host"):
        raise ValueError(
            f"final_score must be 'device' or 'host', got {final_score!r}"
        )
    if not isinstance(cuda_graph, bool):
        raise ValueError(f"cuda_graph must be bool, got {cuda_graph!r}")
    if stat_dtype not in (torch.float64, torch.float32):
        raise ValueError(
            "stat_dtype selects the precision of the two statistics GEMMs and "
            f"must be float64 or float32; got {stat_dtype!r}"
        )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda":
        count = torch.cuda.device_count()
        if (not torch.cuda.is_available() or count == 0
                or (device.index is not None and device.index >= count)):
            raise RuntimeError(
                f"CUDA device {device!r} requested but not available "
                f"({count} visible CUDA device(s))"
            )

    if cuda_graph and device.type != "cuda":
        raise ValueError(
            "cuda_graph=True needs a CUDA device (it captures a device graph); "
            "the eager loop is the CPU path"
        )
    if cuda_graph and instrument is not None:
        raise ValueError(
            "cuda_graph=True replays one captured step and cannot attribute "
            "time to gather/E-step/fold segments; pass instrument=None or "
            "cuda_graph=False"
        )

    # When the caller does not pass a config, follow the model's own convention
    # for the trailing partial batch.  ``GaussianMixture.fit`` keeps it for every
    # family but ``full``; ``FitConfig`` carries its own default, and a loop that
    # disagrees by one batch per epoch does not reproduce the host trajectory
    # (measured: 120 vs 125 updates at 200k rows, endpoints 1.40 nats apart while
    # the *scoring* of identical parameters agreed to 0.0).  An explicit config
    # is honoured as given, so a caller can still ask for either.
    if config is None:
        config = FitConfig()
        if hasattr(config, "drop_last"):
            config = dataclasses.replace(
                config, drop_last=model.covariance == "full")
    x_np = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x_np.ndim != 2:
        raise ValueError("data must be a 2-D array (n, d)")
    n, d = int(x_np.shape[0]), int(x_np.shape[1])
    batch_size = model.resolve_batch_size(config, n, d=d)
    # Before the identifiability floor: records carry the optimizer's own answer,
    # not the raw ``FitConfig.batch_size`` (which may be the "auto" sentinel).
    requested_batch = int(batch_size)
    min_batch = (int(model.identification_min_batch(d))
                 if config.enforce_identifiability else 0)
    if min_batch > 0 and batch_size < min_batch:
        batch_size = int(min(min_batch, n))
    from .plan import plan_batches

    batches = plan_batches(n, batch_size, config.nepochs,
                           drop_last=config.drop_last,
                           max_updates=config.max_updates)
    per_epoch = batches.per_epoch
    total_steps = batches.total_steps
    eval_every = config.eval_every_updates or per_epoch
    if eval_mode == "stream" and (eval_data is not None or eval_every != per_epoch):
        raise ValueError(
            "eval_mode='stream' scores each epoch's batches under the "
            "parameters in force when they were measured, so it needs the "
            "training data and one record per epoch; use eval_mode='exact' "
            "for held-out data or a custom eval_every_updates"
        )

    wall0, cpu0 = time.perf_counter(), time.process_time()
    t_up = time.perf_counter()
    x_gpu = torch.as_tensor(x_np, dtype=dtype, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    upload_s = time.perf_counter() - t_up
    # Measured on its own: charging the eval upload from ``t_up`` would count
    # the training upload twice.
    eval_upload_s = 0.0
    if eval_data is None:
        x_eval, eval_np = x_gpu, x_np
    else:
        eval_np = np.ascontiguousarray(np.asarray(eval_data, dtype=np.float64))
        t_eval = time.perf_counter()
        x_eval = torch.as_tensor(eval_np, dtype=dtype, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        eval_upload_s = time.perf_counter() - t_eval

    host_state = model.initialize(x_np, config)
    keys = tuple(model.state_keys)
    k = model.n_components
    fallback = torch.as_tensor(
        np.asarray(host_state.meta.get("fallback_means",
                                       np.zeros((k, d), dtype=np.float64)),
                   dtype=np.float64),
        dtype=torch.float64, device=device)
    # The state is float64 regardless of the E-step dtype (see docstring).
    values: dict[str, torch.Tensor] = {
        key: torch.as_tensor(np.asarray(host_state[key], dtype=np.float64),
                             dtype=torch.float64, device=device)
        for key in keys
    }
    tiny = torch.finfo(dtype).tiny
    rowsum_floor = max(1e-300, tiny)

    def _decode(vals):
        return _decode_diag(vals["counts"], vals["sums"], vals["sq"],
                            family=model.covariance, reg_covar=model.reg_covar,
                            # the host default is relative: a duck-typed model
                            # without the attribute must not silently get the
                            # absolute floor
                            reg_covar_relative=getattr(
                                model, "reg_covar_relative", True),
                            fallback=fallback)

    def _estep(xb):
        """One E-step at ``dtype``; statistics and log-normaliser stay float64."""
        means, cov, weights = _decode(values)
        # The state and decode are float64 by contract; the E-step runs at the
        # rows' dtype.  Cast both operands rather than relying on promotion: a
        # float64 log-weight added to float32 logits would silently promote the
        # whole softmax back to float64 and undo the point of the choice.
        dt = xb.dtype
        means, cov = means.to(dt), cov.to(dt)
        logw = torch.log(weights.clamp(min=1e-300)).to(dt)
        # One n x d product feeds both the quadratic term and the ``sq``
        # statistic.  The reuse is float64-only: at a reduced dtype the
        # statistic must be formed from the *float64* rows, and squaring the
        # rounded rows instead would be a different (worse) number.
        x2 = xb * xb if dt == torch.float64 else None
        logits = _diag_logits(xb, means, cov, x2=x2) + logw[None, :]
        m = logits.max(1, keepdim=True).values
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        e = torch.exp(logits - m)
        s = e.sum(1, keepdim=True)
        r = e / torch.clamp(s, min=tiny)
        r = torch.where(torch.isfinite(r), r, torch.zeros_like(r))
        r = r / torch.clamp(r.sum(1, keepdim=True), min=rowsum_floor)
        log_norm = m + torch.log(s)
        r64 = r.to(torch.float64)
        # The two statistics GEMMs are the step's largest segment and Ada runs
        # float64 at 1/64 the float32 rate.  ``stat_dtype=torch.float32`` moves
        # only THAT reduction to float32 (responsibilities and log-normalisers
        # stay float64) and the fold is always float64, so the error it adds is
        # the accumulation's (~1e-6 relative), not the E-step's.  Off by
        # default: see a step-profile study for the measured price and
        # drift.
        if stat_dtype is torch.float32:
            rs = r.to(torch.float32)
            xs = xb.to(torch.float32)
            stats = {
                "counts": r64.sum(0),               # k-vector: keep float64
                "sums": (rs.t() @ xs).to(torch.float64),
                "sq": (rs.t() @ (xs * xs)).to(torch.float64),
            }
        else:
            xb64 = xb.to(torch.float64)
            x2_64 = xb64 * xb64 if x2 is None else x2
            stats = {
                "counts": r64.sum(0),
                "sums": r64.t() @ xb64,
                "sq": r64.t() @ x2_64,
            }
        bad = ((~torch.isfinite(logits)).sum()
               if instrument is not None else None)
        return stats, log_norm, bad

    def _score(x_t, chunk, vals):
        """Mean NLL of ``vals`` over ``x_t`` in row chunks (one pass)."""
        means, cov, weights = _decode(vals)
        dt = x_t.dtype
        means, cov = means.to(dt), cov.to(dt)
        logw = torch.log(weights.clamp(min=1e-300)).to(dt)
        total, seen = 0.0, 0
        step = max(1, int(chunk))
        for start in range(0, x_t.shape[0], step):
            xb = x_t[start : start + step]
            logits = _diag_logits(xb, means, cov) + logw[None, :]
            m = logits.max(1, keepdim=True).values
            m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
            lse = m + torch.log(torch.exp(logits - m).sum(1, keepdim=True))
            total += float((-lse.to(torch.float64)).sum().item())
            seen += int(xb.shape[0])
        return total / max(seen, 1)

    captured_steps: dict[int, list] = {}
    graph_replays = 0

    def _replay_step(size, order_t, start):
        """Replay the captured step for ``size`` rows, building it on first use.

        One graph per distinct batch size: capture bakes the shapes, and the
        last batch of an epoch is smaller when ``drop_last=False``.  The graphs
        are built (and warmed up on a side stream) once, which syncs; every
        later step is a single ``replay()``.  The fold updates ``values`` in
        place so the pointers the captured decode reads are the live state.
        """
        nonlocal graph_replays
        entry = captured_steps.get(size)
        if entry is None:
            idx = torch.empty(size, dtype=torch.long, device=device)
            idx.copy_(order_t[start : start + size])
            side = torch.cuda.Stream(device=device)
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(2):
                    _estep(x_gpu.index_select(0, idx))
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = _estep(x_gpu.index_select(0, idx))
            entry = [graph, idx, out]
            captured_steps[size] = entry
            # Stream capture RECORDS the kernels; it does not run them.  Replay
            # once here, or the first step folds whatever the buffers held.
            graph.replay()
        else:
            entry[1].copy_(order_t[start : start + size])
            entry[0].replay()
        graph_replays += 1
        return entry[2]

    history: list[dict[str, Any]] = []
    updates = 0
    examples = 0
    aux: dict[str, float] = {}
    acc: dict[str, torch.Tensor] | None = None
    pending: list[tuple[int, Any]] = []
    bad_total = torch.zeros((), dtype=torch.long, device=device)
    stream_sum = torch.zeros((), dtype=torch.float64, device=device)
    stream_n = 0
    cur_epoch = -1
    last_epoch = config.nepochs - 1
    last_loss = float("nan")

    t_loop0 = time.perf_counter()
    cur_order = None
    order_t = None
    with torch.no_grad():
        for epoch, i, order, start, size in _batch_slices(
            n, batch_size, config.nepochs, config.seed, config.shuffle,
            config.drop_last, config.max_updates,
        ):
            if order is not cur_order:
                # one 8-byte-per-row upload per epoch, not one per step: a
                # per-step pageable copy is a stream stall 25 times an epoch.
                cur_order = order
                order_t = torch.as_tensor(order, dtype=torch.long,
                                          device=device)
            if epoch != cur_epoch:
                cur_epoch = epoch
                stream_sum.zero_()
                stream_n = 0
            ctx = UpdateContext(
                step=updates,
                batch_size=size,
                dataset_size=n,
                total_steps=total_steps,
                epoch=epoch,
                examples_processed=examples,
            )
            ev = None
            if instrument is not None and device.type == "cuda":
                ev = [torch.cuda.Event(True) for _ in range(4)]
                ev[0].record()
            if cuda_graph:
                stats, log_norm, bad = _replay_step(size, order_t, start)
                xb = None
            else:
                xb = x_gpu.index_select(0, order_t[start : start + size])
                if ev is not None:
                    ev[1].record()
                stats, log_norm, bad = _estep(xb)
            if bad is not None:
                bad_total = bad_total + bad
            if eval_mode == "stream":
                stream_sum = stream_sum + (-log_norm.to(torch.float64)).sum()
                stream_n += size
            alphas = _fold_alphas(model, ctx, keys, aux)
            for key in keys:
                a = alphas[key]
                if a >= 1.0:
                    if cuda_graph:
                        values[key].copy_(stats[key])
                    else:
                        values[key] = stats[key]
                elif cuda_graph:
                    # in place: the captured decode holds these pointers
                    values[key].mul_(1.0 - a).add_(stats[key] * a)
                else:
                    values[key] = values[key] * (1.0 - a) + stats[key] * a
            if ev is not None:
                ev[2].record()
            if model.finalize and epoch == last_epoch:
                # the host hook re-measures each batch of the last epoch under
                # the *post-update* parameters and decodes the accumulated sum
                if xb is None:            # graph path: fetch the rows for real
                    xb = x_gpu.index_select(0, order_t[start : start + size])
                exact, _, _ = _estep(xb)
                if acc is None:
                    acc = {key: exact[key].clone() for key in keys}
                else:
                    for key in keys:
                        acc[key] = acc[key] + exact[key]
            updates += 1
            examples += size
            if ev is not None:
                ev[3].record()
            if instrument is not None:
                # Recorded on every device: a CPU run has no event pairs, but
                # silently returning an empty log is not an answer.
                pending.append((updates, ev))
            if callback is not None:
                snap = {key: values[key].to(torch.float64).cpu().numpy()
                        for key in keys}
                callback(State(snap, updates, dict(host_state.meta)), ctx)
            if updates % eval_every == 0 or updates == total_steps:
                if eval_mode == "stream":
                    value = float(stream_sum.item()) / max(stream_n, 1)
                else:
                    value = _score(x_eval, chunk_rows, values)
                last_loss = value
                history.append({
                    "update": updates,
                    "epoch": epoch,
                    "examples_processed": examples,
                    "objective": value,
                    "wall_time": time.perf_counter() - wall0,
                })
    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_train = time.perf_counter() - t_loop0

    final_vals = acc if (model.finalize and acc is not None) else values
    host_values = {key: final_vals[key].to(torch.float64).cpu().numpy()
                   for key in keys}
    meta = dict(host_state.meta)
    accumulators: dict[str, Any] = {}
    if model.finalize and acc is not None:
        accumulators["final_values"] = {key: np.array(v, copy=True)
                                       for key, v in host_values.items()}
        accumulators["final_complete"] = True
    params = model.decode(State(host_values, updates, meta,
                                host_state.runtime, accumulators))

    t_score = time.perf_counter()
    final = _score(x_eval, chunk_rows, final_vals)
    if final_score == "host":
        final = float(model.objective(eval_np, params))
    score_s = time.perf_counter() - t_score
    # ``eval_mode="stream"`` scores each batch under the parameters in force
    # when it was measured (prequential); ``exact`` scores the final parameters
    # on the data.  They share the ``objective`` key, so they must not be
    # compared or ratioed unless the kind matches (T39).
    objective_kind = ("prequential" if eval_mode == "stream"
                      else ("held_out" if eval_data is not None else "full_data"))
    if not history or history[-1]["objective"] != final:
        history.append({
            "update": updates,
            "epoch": last_epoch,
            "examples_processed": examples,
            "objective": final,
            "wall_time": time.perf_counter() - wall0,
        })
    for rec in history:
        rec.setdefault("objective_kind", objective_kind)

    nonfinite = None
    if instrument is not None:
        for step_no, evs in pending:
            rec: dict[str, Any] = {"step": step_no}
            if evs is not None:
                rec["gather_ms"] = evs[0].elapsed_time(evs[1])
                rec["estep_ms"] = evs[1].elapsed_time(evs[2])
                rec["fold_ms"] = evs[2].elapsed_time(evs[3])
            instrument.append(rec)
        nonfinite = int(bad_total.item())

    wall_total = time.perf_counter() - wall0
    extras: dict[str, Any] = {
        "n": n,
        "d": d,
        "batch_size": batch_size,
        "batch_size_requested": requested_batch,
        "batch_size_input": config.batch_size,
        "identification_min_batch": min_batch,
        "nepochs": config.nepochs,
        "epochs_completed": examples / max(1, n),
        "updates_per_epoch": per_epoch,
        "state_keys": list(keys),
        "seed": config.seed,
        "objective_kind": objective_kind,
        "compute_plan": None,
        "residency": {
            "loop": "device",
            "fold": "device",
            "decode": "device",
            "eval": eval_mode,
            "final_score": final_score,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "cuda_graph": bool(cuda_graph),
            "stat_dtype": str(stat_dtype).replace("torch.", ""),
            "level": spec.name if spec is not None else "custom",
            "graphs": len(captured_steps),
            "graph_replays": graph_replays,
            "upload_s": upload_s,
            "eval_upload_s": eval_upload_s,
            "wall_train_s": wall_train,
            "score_s": score_s,
            "wall_total_s": wall_total,
            "downloaded_bytes": int(sum(v.nbytes for v in host_values.values())),
            "nonfinite_logits": nonfinite,
            "bit_identity_to_cpu": False,
        },
    }
    extras.update(model.describe())
    return FitResult(
        optimizer=model.name,
        params=params,
        history=history,
        updates=updates,
        examples_processed=examples,
        wall_time=wall_total,
        cpu_time=time.process_time() - cpu0,
        peak_memory=0,
        objective=float(final),
        extras=extras,
    )
