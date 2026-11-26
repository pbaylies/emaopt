"""Gaussian mixture models with persistent sufficient statistics.

For component ``k`` a full E-step produces

    N_k = sum_i r_ik,     S_k = sum_i r_ik x_i,     Q_k = sum_i r_ik x_i x_i^T

and every covariance family is a decode of those three statistics.  Batch EM
rebuilds them from scratch each pass (``Replace``); the stateful variant folds
each mini-batch estimate in with an EMA and decodes the same way.

Covariance families (state cost in brackets):

* ``spherical``  one variance per component               [k*d diagonal of Q]
* ``diag``       one variance per (component, dimension)  [k*d]* ``tied``       one shared full covariance               [k*d + d*d]
* ``full``       one covariance per component               [k*d*d]
* ``lowrank``    diagonal + rank-r factor loading           [k*(d*r + r*r)]

The full family is implemented but is deliberately not the default: its state
is ``O(K D^2)``, which is the first place the "cheap persistent state" premise
stops being obvious.  ``lowrank`` is the family that keeps a full covariance's
off-diagonal structure at wide ``d``: ``Sigma_j = diag(psi_j) + L_j L_j^T`` with
``r`` defaulting to ``d/8``, so state is ``K*(d*r + r^2)`` instead of ``K*d^2``
(at d=512, k=8: 2 MB instead of 16.8 MB) and the E-step costs ``n*k*d*r`` rather
than ``n*k*d^2``.
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any

import numpy as np

from ..core import (
    DataSpec,
    FitConfig,
    FitResult,
    State,
    StatisticalOptimizer,
    UpdateContext,
    fit,
    label_squares,
    label_sums,
    logsumexp_rows,
    squared_distances,
)
from ..state import EMA, Replace

_LOG_2PI = np.log(2.0 * np.pi)


def _weighted_outer(resp: np.ndarray, data: np.ndarray) -> np.ndarray:
    """``sum_i r_ij x_i x_i^T`` for every component, through BLAS.

    The three-operand ``np.einsum("nk,ni,nj->kij", ...)`` runs a naive loop with
    no BLAS dispatch: measured **30.6 s** at n=8192, k=8, d=512 against **0.20 s**
    for this form (154x) with the same answer to 8e-12.  Each component's
    weighted second moment is one GEMM.
    """
    out = np.empty((resp.shape[1], data.shape[1], data.shape[1]),
                   dtype=np.float64)
    for j in range(resp.shape[1]):
        out[j] = data.T @ (data * resp[:, j:j + 1])
    return out

#: Lazy handle on the optional compiled host E-step (T21).  ``None`` = not probed
#: yet, ``False`` = unavailable.  :mod:`emaopt.hostkernels` decides; the NumPy
#: path in this file is both the reference it is verified against and the
#: fallback when the shared object is not built.
_HOST_KERNELS: Any = None


def _host_kernels() -> Any:
    """The compiled host kernel module, or ``False`` when it is not built."""
    global _HOST_KERNELS
    if _HOST_KERNELS is None:
        try:
            from .. import hostkernels
        except Exception:            # pragma: no cover - numpy-only install
            hostkernels = False
        _HOST_KERNELS = hostkernels
    return _HOST_KERNELS


def _tied_device_kernels() -> Any:
    """The opt-in tied E-step, or ``False`` when not requested.

    Two backends, and the default is now the GEMM one: ``emaopt.tiedgemm`` runs
    cuBLAS for the distance and a warp-cooperative epilogue for the softmax, and
    measured 4.00 ms against 401.75 ms for ``emaopt.tiedgpu``'s fused row kernel
    at 1M x 512 with k=64, at the same fp32 accuracy (2e-5 against the NumPy host
    path).  ``EMAOPT_TIED_BACKEND=row`` keeps the fused kernel, which is worth it
    only where a fit cannot afford the ``(rows, comps)`` logits buffer.

    Both return the same statistics dict, so nothing above this point changes.
    """
    if not os.environ.get("EMAOPT_DEVICE_ESTEP", "").strip().lower() in {
            "1", "cuda", "on", "true", "yes"}:
        return False
    backend = os.environ.get("EMAOPT_TIED_BACKEND", "gemm").strip().lower()
    if backend not in {"row", "fused", "slab", "tiedgpu"}:
        try:
            from .. import tiedgemm

            if tiedgemm.available():
                return tiedgemm
        except Exception:            # pragma: no cover - numpy-only install
            pass
    try:
        from .. import tiedgpu
    except Exception:                # pragma: no cover - numpy-only install
        return False
    # Requested, or the GEMM backend is unusable here (an older provider
    # directory without the launcher): the fused kernel is the fallback.
    return tiedgpu if tiedgpu.available() else False


def _device_kernels() -> Any:
    """The opt-in CUDA E-step module, or ``False`` when it was not requested.

    Device execution is a request, never a default: a fit that silently changed
    precision would make every recorded number ambiguous.  ``EMAOPT_DEVICE_ESTEP``
    opts in, the artifact must pass its self-check, and :mod:`devicekernels`
    still gates on ``WORK_GATE`` before any call leaves the host.
    """
    if not os.environ.get("EMAOPT_DEVICE_ESTEP", "").strip().lower() in {
            "1", "cuda", "on", "true", "yes"}:
        return False
    try:
        from .. import devicekernels
    except Exception:                # pragma: no cover - numpy-only install
        return False
    return devicekernels if devicekernels.available() else False

#: Points per component a full covariance needs before its off-diagonals are
#: worth estimating.  ``4 * d`` is the identifiability floor (the
#: Marchenko-Pastur bulk of a d-dimensional sample reaches 2.25x the true
#: variance at ``n = 4d``); the extra 25% absorbs uneven component counts, since
#: ``4dk`` per batch is only the average.  Measured at d=512, k=8 on correlated
#: data: batch 8192 (~1024/component, 2d) left all eight components regularised
#: and scored the diagonal family's 903.6 nats/point; batch 20000 (~2500, 4.9d)
#: identified all eight and scored 793.9, 109.3 nats better than diag.
_FULL_POINTS_PER_DIM = 5.0


def covariance_parameter_count(family: str, k: int, d: int, rank: int = 0) -> int:
    """Free covariance parameters a family spends, for cost/quality comparisons.

    ``spherical`` is one variance per component, ``tied`` one shared full matrix,
    ``lowrank`` a diagonal plus a rank-r loading, ``full`` one full matrix per
    component.  A ``full`` component with fewer than ``4 * d`` effective
    observations is decoded as its diagonal and counted in
    ``params["regularized_components"]``: the off-diagonal cannot be estimated
    stably from so few points.
    """
    if family == "spherical":
        return k
    if family == "tied":
        return d * (d + 1) // 2
    if family == "lowrank":
        # The free covariance parameters are the diagonal ``psi`` and the
        # ``d x r`` loading.  The ``r x r`` sketch Gramian is state used to
        # decode those parameters, not another fitted parameter.
        r = max(1, min(int(rank or max(2, d // 8)), d))
        return k * (d + d * r)
    if family == "diag":
        return k * d
    if family == "full":
        return k * d * (d + 1) // 2
    raise ValueError(f"unknown covariance type {family!r}")


class GaussianMixture(StatisticalOptimizer):
    """EM for Gaussian mixtures over persistent (N, S, Q) statistics."""

    name = "gmm"

    def __init__(
        self,
        n_components: int,
        *,
        covariance: str = "diag",
        updater: Any = None,
        #: Variance floor.  Where it applies it is a **fraction of the data's own
        #: mean variance** (:attr:`reg_covar_relative`), so the floor moves with
        #: the data instead of sitting at a fixed absolute number.  At unit
        #: variance that is the absolute 1e-6 it always was; at variance 1e-6 it
        #: is 1e-12, where the absolute reading put the floor on top of the
        #: signal and silently zeroed the block -- measured on the LAION feature
        #: file, whose ICA half is stored at variance 1e-6.  ``tied`` reads the
        #: scale from the shared covariance's mean diagonal; ``diag`` and
        #: ``spherical`` from the pooled mean variance of the state, so one
        #: number floors a whole state and a collapsed component cannot set its
        #: own floor; ``full`` floors each identified component's eigenvalues and
        #: ``lowrank`` its residual ``psi`` off the same rule, so no family is
        #: left scale-dependent.  ``reg_covar_relative=False`` restores the
        #: absolute floor everywhere.
        #: Recorded fits (T34) used the absolute floor, so numbers move only
        #: where an input's variance is far from 1.
        reg_covar: float = 1e-6,
        #: Read ``reg_covar`` as a fraction of the data's mean variance rather
        #: than an absolute number (see :meth:`reg_covar_floor`).  ``False``
        #: restores the absolute floor for every family.
        reg_covar_relative: bool = True,
        init: str = "kmeans++",
        #: Units the k-means seed runs in.  ``"raw"`` (default) is the
        #: historical behaviour: one partition over the columns as given, so a
        #: block whose variance is six orders of magnitude below its neighbours'
        #: gets no say in where the components start.  ``"std"`` seeds in units
        #: of each column's own spread and maps the centroids back, so every
        #: block gets a vote; the statistics are still measured on the raw rows.
        init_scale: str = "raw",
        n_init_iter: int = 10,
        init_sample: int | None = None,
        n_init: int = 1,
        seed: int = 0,
        max_epochs: int = 10,
        revival: bool = False,
        revival_threshold: float = 1e-3,
        revival_split: float = 0.5,
        finalize: bool = False,
        rank: int | None = None,
        identification_window: float | None = None,
    ) -> None:
        super().__init__(updater or Replace())
        if covariance not in ("spherical", "diag", "tied", "full", "lowrank"):
            raise ValueError(f"unknown covariance type {covariance!r}")
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1")
        if float(reg_covar) < 0.0:
            raise ValueError("reg_covar must be >= 0")
        if init not in ("kmeans++", "kmeanspp", "random"):
            raise ValueError(
                "init must be 'kmeans++'/'kmeanspp' or 'random' for a mixture"
            )
        if init_scale not in ("raw", "std"):
            raise ValueError("init_scale must be 'raw' or 'std'")
        self.n_components = int(n_components)
        self.covariance = covariance
        self.reg_covar = float(reg_covar)
        self.reg_covar_relative = bool(reg_covar_relative)
        self.init = init
        self.init_scale = init_scale
        self.n_init_iter = int(n_init_iter)
        #: Rows offered to the k-means seed.  ``None`` uses the coreset rule
        #: ``clamp(32k, 512, n)``; a value >= n uses every row.  The seed is the
        #: only place the mixture's local optimum is chosen, so making it
        #: configurable is what lets a caller spend more than 32k rows on a
        #: 64-component model.  Measured (5 seeds x
        #: 3 shapes): the historical ``max(2048, 64k)`` sample with 3 restarts
        #: costs 16.9-19.1 ms and is *never* better on held-out nll than the
        #: coreset sample with one restart (init 2.4-2.5 ms); 4 of 15 seed-shape
        #: cells are strictly better, the rest tie exactly.
        if init_sample is not None and int(init_sample) < 1:
            raise ValueError("init_sample must be >= 1")
        if int(max_epochs) < 1:
            raise ValueError("max_epochs must be >= 1")
        self.init_sample = None if init_sample is None else int(init_sample)
        self.n_init = max(1, int(n_init))
        self.seed = int(seed)
        self.max_epochs = int(max_epochs)
        # e-means' two compute-saving extras, ported to the mixture:
        #   revival  -- a component whose mass collapses under the EMA is
        #               replaced by one half of a size-proportional donor, so
        #               the model keeps k live components instead of one parked
        #               at a mean no data supports.
        #   finalize -- accumulate exact responsibilities-statistics over the
        #               last epoch and decode those, the direct analogue of
        #               e-means' final exact-mean pass (one extra E-step over
        #               one epoch's examples, not another pass).
        self.revival = bool(revival)
        self.revival_threshold = float(revival_threshold)
        self.revival_split = float(revival_split)
        self.finalize = bool(finalize)
        # lowrank only: an explicit rank, else d // 8.  The sketch matrix and the
        # resolved rank are cached so measurement and decode use the same ones.
        if rank is not None and int(rank) < 1:
            raise ValueError("rank must be >= 1 when supplied")
        if identification_window is not None:
            if not 0.0 < float(identification_window) <= 1.0:
                raise ValueError(
                    "identification_window must be in (0, 1] when supplied"
                )
            #: ``None`` gates the full family on the raw counts (one batch's
            #: mass).  A float is the fold's step size, which converts those
            #: counts to window evidence: an EMA averages ~(2-a)/a batches.
            identification_window = float(identification_window)
        self.identification_window = identification_window
        self.rank = None if rank is None else int(rank)
        self._rank = 0
        self._projection: np.ndarray | None = None
        if covariance == "full":
            extra = ("scatter",)
        elif covariance == "lowrank":
            # A rank-r factor model needs the *centred* cross moment between the
            # data and a fixed r-dimensional sketch, plus the sketch's own second
            # moment -- never the d*d second moment.  That is what makes the wide
            # arm affordable: k*(d*r + r*r) state instead of k*d*d.
            extra = ("sq", "sz", "sxz", "sz1")
        elif covariance == "tied":
            # A shared full covariance needs one *global* second moment, not one
            # per component: sum_j Q_j = sum_i x_i x_i^T, because a point's
            # responsibilities sum to one.  That is d*d state, not k*d*d.
            extra = ("gscatter",)
        else:
            extra = ("sq",)
        self.state_keys = ("counts", "sums") + extra

    # -- initialization ----------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        k = self.n_components
        rng = np.random.default_rng(self.seed)
        # Warm start from k-means: centroids *and* per-cluster scatter.  Using the
        # global variance for every component (the obvious first draft) makes one
        # broad component able to swallow two clusters at d>=3, which shows up as
        # a degenerate local optimum rather than a numerical problem.
        from .kmeans import LloydKMeans
        from ..core import assign

        if self.init_sample is None:
            from .smeans import coreset_size

            cap = coreset_size(n, k)
        else:
            cap = int(self.init_sample)
        sub = data
        if n > cap:
            sub = data[rng.choice(n, size=cap, replace=False)]
        niter = min(10, self.n_init_iter)
        seed_sub, seed_scale = self._seed_scaling(sub)
        seed_fit = LloydKMeans(
            k, niter=niter, seed=self.seed, n_init=self.n_init, init=self.init
        ).fit(seed_sub, niter=niter)
        means = seed_fit.params["centroids"] * seed_scale
        labels = assign(sub, means)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        var = np.var(sub, axis=0)
        # The seed gets the same scale-relative floor the decode uses: at
        # variance 1e-6 the absolute reading seeded every component at twice the
        # data's own variance, before the first batch was ever measured.
        floor = self.reg_covar_floor(var)
        global_var = var + floor

        sums = np.zeros((k, d), dtype=np.float64)
        np.add.at(sums, labels, sub)
        meta = {"fallback_means": means.copy(), "seed_rows": int(sub.shape[0])}
        values: dict[str, np.ndarray] = {"counts": counts, "sums": sums}
        if self.covariance == "full":
            scatter = np.zeros((k, d, d), dtype=np.float64)
            for j in range(k):
                members = sub[labels == j]
                # decode() reads this as Q_j = sum_i r_ij x_i x_i^T -- an
                # UNCENTRED second moment -- and subtracts mean mean^T itself.
                # Storing a centred scatter here subtracts the mean twice, which
                # is indefinite for any cluster whose mean is not tiny.  One
                # batch pass survives it (Replace rebuilds the state); streaming
                # does not: the clipped-to-reg_covar covariance sits at the
                # cluster mean, so its density is enormous, it takes every point
                # within the first batches and every other component decays to
                # zero weight.
                cov = members.T @ members if members.size else np.zeros((d, d))
                if counts[j] < d + 1:  # too few points for a stable estimate
                    # decode() reads this slot as sum_i r_ij x_i x_i^T -- an
                    # UNCENTRED second moment -- so the fallback must be the
                    # second moment of a Gaussian with variance global_var at
                    # the cluster's mean.  Writing diag(global_var) alone (the
                    # first attempt) handed decode a matrix that never contained
                    # the mean, so it subtracted mean*mean^T from it: negative
                    # variance in every dimension the mean is away from zero.
                    # Replace rebuilds the state on the next update and hid this;
                    # EMA keeps 90% of it per step and the fit exploded to 6.4e6
                    # nats/point at d=512, where the initialisation subsample is
                    # smaller than d.  tests/test_gmm_full_conditioning.py pins it.
                    safe_n = max(counts[j], 1.0)
                    mu = sums[j] / safe_n if counts[j] > 0 else means[j]
                    cov = (np.diag(global_var) + np.outer(mu, mu)) * safe_n
                scatter[j] = cov + floor * counts[j] * np.eye(d)
            values["scatter"] = scatter
        elif self.covariance == "tied":
            values["gscatter"] = sub.T @ sub
        elif self.covariance == "lowrank":
            projection = self._sketch_projection(d)
            onehot = np.zeros((len(sub), k), dtype=np.float64)
            onehot[np.arange(len(sub)), labels] = 1.0
            z = sub @ projection
            sq = onehot.T @ (sub**2)
            empty = counts < 1e-9
            if np.any(empty):
                # ``decode`` divides the second moment by ``counts`` (which is
                # floored at 1e-9 below).  Store the tiny component's *scaled*
                # second moment, not a full-size one, or the decoded variance
                # becomes O(n / 1e-9) instead of the global fallback variance.
                sq[empty] = 1e-9 * (global_var + means[empty] ** 2)
            values["sq"] = sq
            values["sz"] = np.einsum("mk,mr,ms->krs", onehot, z, z)
            values["sxz"] = np.einsum("mk,md,mr->kdr", onehot, sub, z)
            values["sz1"] = onehot.T @ z
            meta["lowrank_rank"] = int(projection.shape[1])
        else:
            sq = np.zeros((k, d), dtype=np.float64)
            np.add.at(sq, labels, sub**2)
            # regularise dimensions a cluster does not cover at all
            empty = counts < 1e-9
            if np.any(empty):
                # Same convention as the lowrank branch: the decode floors
                # counts at 1e-9, so the fallback second moment must be scaled
                # by that tiny mass to decode to the global variance.
                sq[empty] = 1e-9 * (global_var + means[empty] ** 2)
            values["sq"] = sq
        values["counts"] = np.maximum(counts, 1e-9)
        return State(values, 0, meta)

    def _seed_scaling(self, sub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The rows the k-means seed sees, and the per-column factor back.

        ``"raw"`` returns ``sub`` and a factor of one, so the default seed is
        the historical one to the bit.  ``"std"`` divides each column by its own
        sample standard deviation: the seed partition is the only place the
        mixture's local optimum is chosen, and in raw units a block at variance
        1e-6 is invisible beside one at variance 1, so one responsibility split
        is asked to serve both.  A zero-variance column keeps its units (the
        factor stays one) rather than dividing by zero.
        """
        if self.init_scale != "std":
            return sub, np.ones(sub.shape[1], dtype=np.float64)
        scale = sub.std(axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0.0), scale, 1.0)
        return sub / scale, scale

    # -- closed-form construction from labels ------------------------------- #
    @classmethod
    def from_labels(
        cls,
        data: np.ndarray,
        labels: np.ndarray,
        *,
        covariance: str = "tied",
        reg_covar: float = 1e-6,
        reg_covar_relative: bool = True,
        n_components: int | None = None,
    ) -> tuple["GaussianMixture", dict[str, Any]]:
        """The exact M-step for a labelled dataset: no EM, no initialization.

        When the responsibilities are known -- because the data came with labels,
        or because a cheaper clusterer produced them -- the maximum-likelihood
        mixture for those responsibilities is a closed form over per-label
        statistics::

            N_j = |{i : label_i = j}|      S_j = sum_i x_i over label j
            Q_j = sum_i x_i x_i^T           (per component, ``full``)
            G   = sum_i x_i x_i^T           (shared, ``tied``)

        and the decode already knows what to do with them, because that is the
        same state a batch EM pass would have measured.  This is the LAION-400M
        recipe's last stage used as intended (``Fit Clusters.ipynb`` fits 64 tied
        components): given embeddings and cluster labels the model is a reduction,
        not a training run.  Cost is one pass over the data and an O(k d^2)
        reduction for ``tied``, so the state stays small at any ``d``.

        ``lowrank`` is rejected: its factors are fitted, not reducible.

        Returns ``(model, params)`` so callers can score held-out data with
        ``model.objective(x, params)``.
        """
        data = np.asarray(data, dtype=np.float64)
        labels = np.asarray(labels).astype(np.int64).ravel()
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        if labels.size != data.shape[0]:
            raise ValueError(
                f"labels have {labels.size} entries for {data.shape[0]} rows"
            )
        if np.any(labels < 0):
            raise ValueError("labels must be non-negative integers")
        k = int(n_components if n_components is not None else labels.max() + 1)
        if labels.max() >= k:
            raise ValueError(f"label {int(labels.max())} exceeds n_components={k}")
        if covariance == "lowrank":
            raise ValueError("lowrank has no closed form: its factors are fitted")

        model = cls(k, covariance=covariance, reg_covar=reg_covar,
                   reg_covar_relative=reg_covar_relative)
        n, d = data.shape
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        sums = label_sums(data, labels, k)
        values: dict[str, np.ndarray] = {"counts": counts, "sums": sums}
        if covariance == "full":
            scatter = np.zeros((k, d, d), dtype=np.float64)
            for j in range(k):
                block = data[labels == j]
                if block.size:
                    scatter[j] = block.T @ block
            values["scatter"] = scatter
        elif covariance == "tied":
            # a point contributes to exactly one label, so the global second
            # moment is unweighted -- the same fact the streaming measurement uses
            values["gscatter"] = data.T @ data
        else:
            values["sq"] = label_squares(data, labels, k)

        # A label that never occurs is not estimable at all.  Give it a tiny mass
        # at the global mean with the global variance, so the decode returns a
        # broad component there instead of a reg_covar spike (which would take
        # every point near it).
        empty = counts < 0.5
        global_mean = data.mean(axis=0)
        var = data.var(axis=0)
        global_var = var + model.reg_covar_floor(var)
        if np.any(empty):
            tiny = 1e-9
            counts[empty] = tiny
            sums[empty] = global_mean[None, :] * tiny
            if covariance == "full":
                for j in np.flatnonzero(empty):
                    values["scatter"][j] = tiny * (
                        np.diag(global_var) + np.outer(global_mean, global_mean)
                    )
            elif covariance != "tied":
                values["sq"][empty] = tiny * (global_var + global_mean**2)[None, :]

        meta = {"fallback_means": np.repeat(global_mean[None, :], k, axis=0)}
        params = model.decode(State(values, 0, meta))
        return model, params

    # -- e-means extras ----------------------------------------------------- #
    def begin_fit(self, state: State, config: FitConfig, n: int, batch_size: int,
                  total_steps: int) -> None:
        state.runtime["final_epoch"] = int(config.nepochs) - 1

    def on_batch(self, state: State, batch: np.ndarray, ctx: UpdateContext) -> None:
        """Accumulate exact statistics over the last epoch, for finalization.

        ``measure`` already reduces a batch to exactly the state's keys with the
        current parameters, so the accumulator is the sum of those reductions:
        one exact M-step over the last epoch's examples, using the parameters
        that the streaming run arrived at.  Written into ``accumulators`` and
        applied by :meth:`decode`, so the fit's final answer is not an EMA of the
        last few mini-batches' statistics.
        """
        if not self.finalize:
            return
        final_epoch = state.runtime.get(
            "final_epoch", state.meta.get("_final_epoch", -1))
        if ctx.epoch != final_epoch:
            return
        stats = self.measure(batch, state)
        accumulated = state.accumulators.get("final_values")
        if accumulated is None:
            accumulated = {
                key: np.zeros_like(np.asarray(value, dtype=np.float64))
                for key, value in stats.values.items()
            }
        for key, value in stats.values.items():
            accumulated[key] = accumulated[key] + value
        state.accumulators["final_values"] = accumulated
        if ctx.step + 1 >= ctx.total_steps:
            state.accumulators["final_complete"] = True

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        new = super().update_state(state, statistics, ctx)
        if self.revival:
            new = self._revive(new, ctx)
        return new.inherit_pockets(state)

    def _revive(self, state: State, ctx: UpdateContext) -> State:
        """Split a size-proportional donor to revive collapsed components.

        A component is dead when its mass collapses below the threshold *or*,
        for the diagonal families, when every dimension sits at the variance
        floor: a pinned component keeps a few counts forever and no mass rule
        catches it.

        The donor is drawn with probability proportional to its mass and its
        mass is halved between two components whose means are moved symmetrically
        apart along the donor's widest-variance axis (so the donor's mean is
        preserved), each half carrying the donor's covariance.  The split
        distance is ``revival_split`` times the donor's standard deviation on
        that axis, and both halves' second moments are written in the
        *uncentred* form the decode expects -- the same convention a split of
        ``Q = N (C + mu mu^T)`` requires.

        ``tied`` and ``lowrank`` are left alone: their extra state is a shared or
        sketched second moment with no per-component split to define.
        """
        if self.covariance not in ("diag", "spherical", "full"):
            return state
        counts = np.array(state["counts"], dtype=np.float64)
        sums = np.array(state["sums"], dtype=np.float64)
        k, d = sums.shape
        if k < 2:
            return state
        threshold = max(1e-12, self.revival_threshold * ctx.batch_size / max(1, k))
        second_key = "scatter" if self.covariance == "full" else "sq"
        second = np.array(state[second_key], dtype=np.float64)
        dead = counts < threshold
        if self.covariance in ("diag", "spherical"):
            # Mass alone does not mean alive: a component pinned at the variance
            # floor keeps N of 1-3 counts and never recovers, because its
            # responsibilities collapse onto its own points.  Flag it
            # geometrically -- every dimension at reg_covar -- so revival
            # actually rescues the failure it was written for (T32).
            safe_c = np.maximum(counts, 1e-12)
            var = second / safe_c[:, None] - (sums / safe_c[:, None]) ** 2
            # the same pooled floor the decode uses, so "pinned at the floor"
            # means the same number in both places
            mass = max(float(counts.sum()), 1e-12)
            mean_all = sums.sum(axis=0) / mass
            pooled_var = np.maximum(second.sum(axis=0) / mass - mean_all**2, 0.0)
            floor = self.reg_covar_floor(pooled_var)
            dead = dead | (var.max(axis=1) <= floor * (1.0 + 1e-9))
        dead = np.flatnonzero(dead)
        if dead.size == 0:
            return state
        rng = np.random.default_rng(self.seed + int(ctx.step))
        revived = 0
        for target in dead:
            active = np.flatnonzero(counts >= threshold)
            if active.size == 0:
                break
            probs = counts[active] / counts[active].sum()
            donor = int(rng.choice(active, p=probs))
            if counts[donor] <= 2.0 * threshold:
                continue
            mass = counts[donor]
            mu = sums[donor] / mass
            if self.covariance == "full":
                cov = second[donor] / mass - np.outer(mu, mu)
                var = np.maximum(np.diag(cov), self.reg_covar_floor(np.diag(cov)))
            else:
                cov = None
                var = np.maximum(second[donor] / mass - mu**2, floor)
            # Split along the donor's *widest* dimension.  A direction drawn
            # uniformly on the sphere is almost orthogonal to the axis the donor
            # varies along once d is large, so the halves land inside the donor's
            # own spread and the next E-step hands them straight back to it.  The
            # docstring promised per-dimension sigma and the code used a random
            # rotation (T32); this is that sigma, on the axis that carries it.
            axis = int(np.argmax(var))
            scale = math.sqrt(max(float(var[axis]), 1e-12))
            delta = np.zeros(d)
            delta[axis] = self.revival_split * scale
            half = mass / 2.0
            sums[donor] = (mu + delta) * half
            sums[target] = (mu - delta) * half
            counts[donor] = half
            counts[target] = half
            if self.covariance == "full":
                second[donor] = half * (cov + np.outer(mu + delta, mu + delta))
                second[target] = half * (cov + np.outer(mu - delta, mu - delta))
            else:
                second[donor] = half * (var + (mu + delta) ** 2)
                second[target] = half * (var + (mu - delta) ** 2)
            revived += 1

        if revived == 0:
            return state
        values = dict(state.values)
        values["counts"] = counts
        values["sums"] = sums
        values[second_key] = second
        meta = dict(state.meta)
        meta["_revivals"] = int(meta.get("_revivals", 0)) + revived
        return State(values, state.step, meta)

    # -- decode ------------------------------------------------------------- #
    def decode(self, state: State) -> dict[str, Any]:
        # ``accumulators`` is the home; the ``_final_*`` meta keys are read too so
        # a state serialized before the split still decodes to the same answer.
        complete = state.accumulators.get(
            "final_complete", state.meta.get("_final_complete", False))
        final = (state.accumulators.get("final_values")
                 or state.meta.get("_final_values"))
        if complete and final:
            state = State(
                {k: np.asarray(v, dtype=np.float64) for k, v in final.items()},
                state.step,
                state.meta,
                state.runtime,
                state.accumulators,
            )
        counts = np.asarray(state["counts"], dtype=np.float64)
        sums = np.asarray(state["sums"], dtype=np.float64)
        k, d = sums.shape
        safe = np.maximum(counts, 1e-12)
        means = sums / safe[:, None]
        empty = counts <= 1e-9
        if np.any(empty):
            means[empty] = state.meta.get("fallback_means", means)[empty]

        if self.covariance == "full":
            scatter = np.asarray(state["scatter"], dtype=np.float64)
            cov = scatter / safe[:, None, None] - means[:, :, None] * means[:, None, :]
            cov = 0.5 * (cov + np.swapaxes(cov, 1, 2))
            total_mass = max(float(counts.sum()), 1e-12)
            # A sample covariance in d dimensions needs n_j >> d to be a stable
            # estimate, not merely n_j > d.  The Marchenko-Pastur spectrum of a
            # d-dimensional sample with n observations spans (1 -+ sqrt(d/n))^2,
            # so at n_j = 4d the extremes are already 0.25x and 2.25x the true
            # variance -- and at n_j = d the small end reaches zero.  Flooring
            # those eigenvalues at reg_covar turns a component into a spike whose
            # log density at its own mean is O(-d * log(reg) / 2): EM then parks
            # one point on it.  Measured at d=512, batch 4096, k=8 with a d + 1
            # threshold, two components squeaked past and the fit still exploded
            # to 1354990 nats/point; with the diagonal fallback it lands with the
            # diagonal family.  Drop the correlations that cannot be estimated and
            # keep the diagonal, which the same second moment identifies; the
            # count travels with the parameters so a run never claims a family it
            # could not fit.
            # T30: gate on *evidence*, not on the raw batch mass.  An EMA fold
            # averages ~(2-a)/a batches, so ``counts`` -- one batch's worth --
            # understates the observations behind the covariance by that factor
            # (19x at a=0.1).  The default stays raw counts for two reasons: an
            # average of per-batch covariances keeps a between-batch component
            # that pooling rows would remove, so (2-a)/a is an upper bound on
            # the evidence rather than the evidence; and every recorded
            # full-family result was gated on raw counts.  Passing
            # ``identification_window=a`` opts into the optimistic reading, and
            # the decode records the factor it used.
            window = self.identification_window
            factor = 1.0 if window is None else (2.0 - window) / window
            unidentified = (counts * factor) < (4 * d)
            if np.any(empty):
                # An empty component has no scatter at all; give it the global
                # diagonal instead of a reg_covar spike at its fallback mean.
                global_second = scatter.sum(axis=0) / total_mass
                global_mean = (means * (counts / total_mass)[:, None]).sum(axis=0)
                # read the floor off the raw second moment, which is positive by
                # construction, not off the centred variance, which can be <= 0
                floor = self.reg_covar_floor(np.diag(global_second))
                global_var = np.maximum(
                    np.diag(global_second) - global_mean**2, floor
                )
                cov[empty] = np.diag(global_var)
            ident = np.flatnonzero(~empty & ~unidentified)
            if ident.size:
                # project to the PSD cone: a component with tiny weight can
                # produce a numerically indefinite scatter once means drift.
                cov[ident] = self._psd_batch(cov[ident])
            unident = np.flatnonzero(~empty & unidentified)
            if unident.size:
                diag = np.diagonal(cov[unident], axis1=1, axis2=2)
                floors = self._reg_covar_floors(diag.mean(axis=1))
                cov[unident] = (np.eye(d) * np.maximum(
                    diag, floors[:, None])[:, None, :])
            weights = counts / total_mass
            return {"weights": weights, "means": means, "covariances": cov,
                    "covariance": "full", "counts": counts,
                    "identification_factor": float(factor),
                    "regularized_components": int(unidentified.sum())}

        total = counts.sum()
        if self.covariance == "tied":
            gscatter = np.asarray(state["gscatter"], dtype=np.float64)
            mass = max(total, 1e-12)
            # E[x x^T] = C_shared + sum_j w_j mu_j mu_j^T, so the shared
            # covariance keeps the off-diagonals a diagonal decode cannot.
            cov = gscatter / mass - (means * (counts / mass)[:, None]).T @ means
            return {
                "weights": counts / mass,
                "means": means,
                "covariances": self._psd(cov, self.reg_covar_floor(cov)),
                "covariance": "tied",
                "counts": counts,
            }

        if self.covariance == "lowrank":
            return self._decode_lowrank(state, counts, means, safe)

        sq = np.asarray(state["sq"], dtype=np.float64)
        second = np.maximum(sq / safe[:, None] - means**2, 0.0)
        # One floor for the whole state, read off the pooled mean variance of the
        # statistics at hand -- the diag/spherical analogue of the tied
        # covariance's mean diagonal.  Reading it off each component instead
        # would let a collapsed component's own (near-zero) variance set its own
        # floor, so no component could ever register as pinned, and the revival
        # check below looks for exactly that.
        mass = max(float(total), 1e-12)
        mean_all = sums.sum(axis=0) / mass
        pooled_var = np.maximum(sq.sum(axis=0) / mass - mean_all**2, 0.0)
        floor = self.reg_covar_floor(pooled_var)
        # A component with fewer observations than dimensions has a sample
        # variance of (near) exactly zero -- a singleton's is zero -- so its
        # covariance collapses to reg_covar and its log density at its own mean
        # is O(-d log reg / 2): a spike EM then parks points on.  The full family
        # guards this with a 4d count threshold; diag/spherical only checked for
        # emptiness, so a singleton decoded to a reg_covar spike (T31).  Shrink
        # toward the pooled variance instead: (N var + m g)/(N + m) with m = d,
        # the standard empirical-Bayes form.  Components with N >= d are
        # untouched term for term, so no healthy fit moves.
        weak = counts < float(d)
        if np.any(weak):
            global_var = np.maximum(pooled_var, floor)
            n_weak = counts[weak, None]
            second[weak] = (
                n_weak * second[weak] + float(d) * global_var[None, :]
            ) / (n_weak + float(d))
        if self.covariance == "spherical":
            # one variance per component, shared across dimensions (sklearn's
            # spherical).  Pooling over components as well is a strictly weaker
            # model, and it was measurably worse than the reference.
            var = second.mean(axis=1) + floor
            cov = np.repeat(var[:, None], d, axis=1)
        else:  # diag
            cov = second + floor
        return {
            "weights": counts / max(total, 1e-12),
            "means": means,
            "covariances": cov,
            "covariance": self.covariance,
            "counts": counts,
        }

    # -- low-rank family ---------------------------------------------------- #
    def _sketch_projection(self, d: int) -> np.ndarray:
        """Fixed random projection ``X @ Omega`` used by the lowrank family.

        Seeded from the model so every batch, and the initialization pass, sketch
        into the same subspace -- otherwise the accumulated moments are moments
        of unrelated random variables.
        """
        if self._projection is None or self._projection.shape[0] != d:
            rank = int(self.rank) if self.rank else max(2, d // 8)
            rank = max(1, min(rank, d))
            self._rank = rank
            self._projection = np.random.default_rng(self.seed + 977).normal(
                size=(d, rank)
            ) / np.sqrt(rank)
        return self._projection

    def _decode_lowrank(
        self, state: State, counts: np.ndarray, means: np.ndarray, safe: np.ndarray
    ) -> dict[str, Any]:
        """``Sigma_j = diag(psi_j) + L_j L_j^T`` from the sketched moments.

        With ``Z = X Omega`` the centred moments are ``Phi_j = Cov[x, z]`` and
        ``Zt_j = Cov[z, z]``, and the Nystroem approximation
        ``Sigma_j - diag ~= Phi_j Zt_j^{-1} Phi_j^T`` is a rank-r projection of
        the covariance.  ``r = d`` makes the sketch invertible, so the decode is
        then *exactly* the full covariance -- the rank-d test asserts that.

        ``psi_j`` is set so the decoded diagonal equals the measured diagonal:
        the low-rank part is an estimate of the off-diagonal structure, not a
        licence to move the variance the data actually showed.

        This is randomized Nystroem *moment matching*, not maximum-likelihood
        factor analysis.  It agrees with a fitted FA in the limit ``r = d`` (the
        rank-d test asserts exactness against the full covariance), but at small
        ``r`` it optimises a different objective and its loadings are not FA's;
        no parity with a fitted FA is claimed without a measured comparison
        (T34).
        """
        sq = np.asarray(state["sq"], dtype=np.float64)
        sz = np.asarray(state["sz"], dtype=np.float64)
        sxz = np.asarray(state["sxz"], dtype=np.float64)
        sz1 = np.asarray(state["sz1"], dtype=np.float64)
        k, d = sq.shape
        rank = sz.shape[1]
        second = np.maximum(sq / safe[:, None] - means**2, 0.0)
        zbar = sz1 / safe[:, None]
        centered_zz = sz / safe[:, None, None] - zbar[:, :, None] * zbar[:, None, :]
        centered_xz = sxz / safe[:, None, None] - means[:, :, None] * zbar[:, None, :]
        psi = np.empty((k, d), dtype=np.float64)
        loadings = np.empty((k, d, rank), dtype=np.float64)
        # ``psi`` is a residual variance, so it carries a scale exactly as the
        # diag family's variance does: one floor for the state, read off the
        # mean of the per-component variance estimates (unit scale -> 1e-6).
        reg_floor = self.reg_covar_floor(second.mean(axis=0))
        for j in range(k):
            gram = 0.5 * (centered_zz[j] + centered_zz[j].T)
            evals, evecs = np.linalg.eigh(gram)
            floor = max(float(np.max(np.abs(evals))), 1.0) * 1e-12
            evals = np.maximum(evals, floor)
            # Lambda Lambda^T = Phi Z^{-1} Phi^T; the eigenvectors of Zt supply
            # the inverse square root without forming it.
            loading = (centered_xz[j] @ evecs) / np.sqrt(evals)
            # The Nystroem estimate can put more diagonal variance into the
            # loading than the component actually has, especially when r is
            # large relative to the per-component sample.  Shrink those loading
            # rows until the diagonal fits; otherwise flooring psi at the
            # state's floor makes the decoded covariance wider than the data.
            # This is a no-op when the sketch already fits.
            ridge = np.einsum("dr,dr->d", loading, loading)
            cap = np.maximum(second[j] - reg_floor, 0.0)
            scale = np.minimum(1.0, cap / np.maximum(ridge, 1e-300))
            if np.any(scale < 1.0):
                loading = loading * np.sqrt(scale)[:, None]
            loadings[j] = loading
            psi[j] = np.maximum(
                second[j] - np.einsum("dr,dr->d", loading, loading), reg_floor
            )
        total = counts.sum()
        return {
            "weights": counts / max(total, 1e-12),
            "means": means,
            "covariances": psi,
            "loadings": loadings,
            "covariance": "lowrank",
            "counts": counts,
        }

    def _lowrank_parts(
        self, means: np.ndarray, psi: np.ndarray, loadings: np.ndarray
    ) -> tuple[np.ndarray, ...]:
        """Shared Woodbury constants for the stacked and wide lowrank paths.

        Centering by the mean component keeps the expanded quadratic form from
        cancelling large offset terms.  The returned mean_factor and mean_q are
        already in that centered frame.
        """
        d = means.shape[1]
        rank = loadings.shape[2]
        inv_psi = 1.0 / psi
        roots = np.sqrt(psi)
        centre = means.mean(axis=0)
        mc = means - centre
        factors = loadings * inv_psi[:, :, None]
        u = loadings / roots[:, :, None]
        gram = np.eye(rank) + np.einsum("kdr,kds->krs", u, u)
        _sign, logdet_gram = np.linalg.slogdet(gram)
        inv_gram = np.linalg.inv(gram)
        const = -0.5 * (d * _LOG_2PI + np.log(psi).sum(axis=1)
                        + logdet_gram)
        mean_factor = np.einsum("kd,kdr->kr", mc, factors)
        mean_q = np.einsum("kd,kd->k", mc * inv_psi, mc)
        return (inv_psi, centre, mc, factors, inv_gram, const,
                mean_factor, mean_q)

    def _lowrank_log_prob(self, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """``log N(x | mu_j, diag(psi_j) + L_j L_j^T)`` via the Woodbury identity.

        ``Sigma = Psi^{1/2} (I + U U^T) Psi^{1/2}`` with ``U = Psi^{-1/2} L``, so
        the log-determinant and the quadratic form cost ``O(d*r^2 + r^3)`` per
        component instead of ``O(d^3)``.

        At moderate width the components' factors are stacked: one
        ``x @ [B_0 ... B_{k-1}]`` product replaces k skinny products, and the
        normalizer uses the same stacked weighted products.  Measured
        8192 x 64, k=16, r=8: 15.3 -> 8.0 ms; 4096 x 128, k=8, r=16:
        7.1 -> 2.7 ms, with max difference 1.3e-12 nats.  At d=512 the stacked
        product is slower, so wide rows keep a per-component path.  When that
        fallback would issue at least 128 small GEMMs (``k * chunks >= 128``),
        one centered GEMM per component over an 8 MB row block is much faster:
        8192 x 320, k=64, r=40 measured 369 -> 152 ms.  Otherwise the chunked
        fallback wins (d=512, k=8 measured 18.8 against 19.5 ms).
        """
        means = params["means"]
        psi = np.asarray(params["covariances"], dtype=np.float64)
        loadings = np.asarray(params["loadings"], dtype=np.float64)
        n, d = x.shape
        k, rank = loadings.shape[0], loadings.shape[2]

        # Stack the per-component factors along the columns.  The expansion
        # below is algebraically the direct Woodbury form; centring x and the
        # means keeps its first term from losing digits on offset data.
        if d <= 256 and n * k * rank <= (1 << 23):
            (inv_psi, centre, mc, factors, inv_gram, const,
             mean_factor, mean_q) = self._lowrank_parts(means, psi, loadings)
            xc = x - centre
            flat = factors.transpose(1, 0, 2).reshape(d, k * rank)
            wu = (xc @ flat).reshape(n, k, rank)
            wu -= mean_factor[None, :, :]
            weighted = (xc * xc) @ inv_psi.T
            cross = xc @ (mc * inv_psi).T
            quad = weighted - 2.0 * cross + mean_q[None, :]
            out = np.empty((n, k), dtype=np.float64)
            for j in range(k):
                wuj = wu[:, j, :]
                corr = wuj @ inv_gram[j].T
                out[:, j] = const[j] - 0.5 * (
                    quad[:, j] - np.einsum("ij,ij->i", corr, wuj))
            return out

        # The fallback's row chunk is sized for a ~2 MB residual block.  At
        # high k that means many small GEMMs; one centered GEMM per component
        # over a larger block avoids the repeated dispatch and cache churn.
        base_rows = max(256, min(2048, (1 << 21) // max(8 * d, 1)))
        chunks = (n + base_rows - 1) // base_rows
        if k * chunks >= 128:
            (inv_psi, centre, mc, factors, inv_gram, const,
             mean_factor, mean_q) = self._lowrank_parts(means, psi, loadings)
            xc = x - centre
            quad = (xc * xc) @ inv_psi.T
            quad -= 2.0 * (xc @ (mc * inv_psi).T)
            quad += mean_q[None, :]
            rows = max(512, min(n, (8 << 20) // max(8 * d, 1)))
            out = np.empty((n, k), dtype=np.float64)
            wu = np.empty((min(rows, n), rank), dtype=np.float64)
            corr = np.empty_like(wu)
            for start in range(0, n, rows):
                end = min(start + rows, n)
                nb = end - start
                xb = xc[start:end]
                wb = wu[:nb]
                cb = corr[:nb]
                for j in range(k):
                    np.matmul(xb, factors[j], out=wb)
                    wb -= mean_factor[j]
                    np.matmul(wb, inv_gram[j].T, out=cb)
                    out[start:end, j] = const[j] - 0.5 * (
                        quad[start:end, j]
                        - np.einsum("ij,ij->i", cb, wb))
            return out

        out = np.empty((n, k), dtype=np.float64)
        eye = np.eye(rank)
        # Aim for a ~2 MB residual block, capped so short rows do not make one
        # chunk larger than the caller's batch.
        rows = base_rows
        w = np.empty((min(rows, n), d), dtype=np.float64)
        for j in range(k):
            root = np.sqrt(psi[j])
            u = loadings[j] / root[:, None]
            gram = eye + u.T @ u
            _sign, logdet_gram = np.linalg.slogdet(gram)
            logdet = float(np.sum(np.log(psi[j]))) + float(logdet_gram)
            inv_gram = np.linalg.inv(gram)
            const = -0.5 * (d * _LOG_2PI + logdet)
            for start in range(0, n, rows):
                end = min(start + rows, n)
                block = w[: end - start]
                np.subtract(x[start:end], means[j], out=block)
                np.divide(block, root, out=block)
                wu = block @ u
                z = wu @ inv_gram.T
                quad = np.einsum("ij,ij->i", block, block) - np.einsum(
                    "ij,ij->i", z, wu
                )
                out[start:end, j] = const - 0.5 * quad
        return out

    def reg_covar_floor(self, cov: np.ndarray) -> float:
        """The absolute eigenvalue floor that ``reg_covar`` implies for ``cov``.

        With ``reg_covar_relative`` (the default) the floor is
        ``reg_covar * trace(C) / d``, the covariance's mean variance.  For
        ``diag``/``spherical`` pass the pooled variance *vector* instead and the
        same expression reduces to its mean; :meth:`_reg_covar_floors` is the
        vectorised form.  That is the scale-equivariant reading of the same
        knob: rescaling every feature by ``s`` scales ``C`` by ``s**2`` and the
        floor with it, so the fitted shape -- and the likelihood up to an
        additive constant -- cannot depend on the units the caller happens to
        use.  The absolute reading is not equivariant; it is what
        ``reg_covar_relative=False`` restores.

        It is the one rule every family reads: ``tied``'s eigenvalues,
        ``diag``/``spherical`` variances, ``full``'s per-component eigenvalues
        (identified and unidentified alike), ``lowrank``'s residual ``psi``, the
        variance seed in :meth:`initialize`/:meth:`from_labels`, the revival
        donor's variance, and the scoring fallbacks.  A degenerate scale (zero,
        negative or non-finite -- an all-zero statistic, or a covariance whose
        diagonal is broken) returns the absolute ``reg_covar`` rather than a zero
        floor, because a zero floor leaves a singular matrix to LAPACK.

        The failure this replaces, measured on the LAION feature file: its ICA
        half is stored at variance 1.0e-6 (sklearn's old ``1/sqrt(n)`` source
        scaling), so an absolute 1e-6 floor sits *exactly* on the block.  The
        joint fit then conditioned on the PCA half 112 nats worse than the
        block's own density and its held-out gain collapsed from 1.23 to 0.15
        nats/point.  On the same data this floor is 4.9e-10 and the block
        participates without the caller rescaling anything.
        """
        if not self.reg_covar_relative:
            return self.reg_covar
        scale = float(np.mean(cov)) if np.ndim(cov) == 1 else float(
            np.mean(np.diag(cov)))
        if not np.isfinite(scale) or scale <= 0.0:
            return self.reg_covar
        return self.reg_covar * scale

    def _reg_covar_floors(self, scales: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`reg_covar_floor` for a vector of scales.

        The same rule as the scalar method, including the degenerate guard, so a
        per-component caller (the ``full`` PSD projection) cannot disagree with
        a whole-state one (``tied``, ``diag``, ``spherical``).
        """
        scales = np.asarray(scales, dtype=np.float64)
        if not self.reg_covar_relative:
            return np.full(scales.shape, self.reg_covar)
        good = np.isfinite(scales) & (scales > 0.0)
        return np.where(good, self.reg_covar * scales, self.reg_covar)

    def _psd(self, cov: np.ndarray, floor: float | None = None) -> np.ndarray:
        """Symmetric PSD projection with the regularisation floor.

        The default floor is :meth:`reg_covar_floor` read off ``cov`` itself, so
        an unflagged call cannot silently fall back to the absolute reading.
        """
        cov = 0.5 * (cov + cov.T)
        evals, evecs = np.linalg.eigh(cov)
        if floor is None:
            floor = self.reg_covar_floor(cov)
        evals = np.maximum(evals, float(floor))
        return (evecs * evals) @ evecs.T

    def _psd_batch(self, covs: np.ndarray,
                   floors: np.ndarray | float | None = None) -> np.ndarray:
        """PSD projection of a ``(k, d, d)`` stack in one LAPACK call.

        The full family decoded one component at a time; at d=8, k=8 that
        is eight 8x8 eigendecompositions per decode and the decode runs
        once per update and once per scoring pass, so the per-call
        overhead was a visible share of the fit.

        ``floors`` defaults to one :meth:`reg_covar_floor` per component, read
        off that component's own mean diagonal: an absolute eigenvalue floor is
        the same scale bug as an absolute variance floor, and at variance 1e-6
        it lifts every eigenvalue of a 512-d covariance onto the floor.
        """
        sym = 0.5 * (covs + np.swapaxes(covs, 1, 2))
        evals, evecs = np.linalg.eigh(sym)
        if floors is None:
            floors = self._reg_covar_floors(
                np.diagonal(sym, axis1=1, axis2=2).mean(axis=1))
        floors = np.asarray(floors, dtype=np.float64)
        evals = (np.maximum(evals, floors[:, None]) if floors.ndim
                 else np.maximum(evals, float(floors)))
        return (evecs * evals[:, None, :]) @ np.swapaxes(evecs, 1, 2)

    #: How many tied factorisations an optimizer keeps for scoring passes.  A
    #: scoring loop reuses one; a fit moves the covariance every epoch, so a
    #: larger cache would hold entries no later call can match.
    _TIED_CACHE_ENTRIES = 4

    def _tied_root(self, cov: np.ndarray) -> tuple[np.ndarray, float]:
        """``(root, logdet)`` for a shared covariance, cached on this optimizer.

        ``root`` is ``L^-T`` from ``cov = L L^T``, so ``root @ root.T`` is the
        precision and ``logdet`` falls out of ``diag(L)``.  Scoring one fitted
        model many times -- ``predict_proba`` per calibration point, ``log_prob``
        per traced pass -- used to redo this on every call: ~10 ms at d=512 on a
        pinned pool, and most of a second when the pool is not pinned.

        Keyed on a digest of the symmetrised matrix's exact bytes, never a
        tolerance: a near miss has to be a miss, because a close-enough
        factorisation would quietly change the answer.
        """
        sym = 0.5 * (cov + cov.T)
        key = hashlib.blake2b(np.ascontiguousarray(sym), digest_size=16).digest()
        cache = getattr(self, "_tied_root_cache", None)
        if cache is None:
            cache = self._tied_root_cache = {}
        if key in cache:
            self._tied_root_hits = getattr(self, "_tied_root_hits", 0) + 1
            return cache[key]
        d = sym.shape[0]
        try:
            chol = np.linalg.cholesky(sym)
        except np.linalg.LinAlgError:
            # Numerical breakdown: keep the identifiable diagonal part.
            sym = np.diag(np.maximum(np.diag(sym), self.reg_covar_floor(sym)))
            try:
                chol = np.linalg.cholesky(sym)
            except np.linalg.LinAlgError:   # pragma: no cover - floored above
                raise
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
        root = np.linalg.solve(chol, np.eye(d)).T
        while len(cache) >= self._TIED_CACHE_ENTRIES:
            cache.pop(next(iter(cache)))
        cache[key] = (root, logdet)
        return root, logdet

    def tied_factorisation_stats(self) -> dict[str, int]:
        """How many tied factorisations scoring reused, and how many it holds."""
        return {"hits": int(getattr(self, "_tied_root_hits", 0)),
                "entries": len(getattr(self, "_tied_root_cache", {}))}

    # -- measurement -------------------------------------------------------- #
    def _shared_log_prob(
        self, x: np.ndarray, cov: np.ndarray, means: np.ndarray
    ) -> np.ndarray:
        """``(n, k)`` log densities for one covariance shared by all components.

        The general path solves ``(Sigma, x - mu_j)`` once per component, so it
        pays ``k n d^2`` per E-step.  A shared covariance factors the quadratic
        form through ``R`` with ``P = R R^T`` (``P = Sigma^-1``),

            (x-mu)^T P (x-mu) = |x R - mu R|^2
                              = |xR|^2 - 2 (xR).(mu R) + |mu R|^2,

        which needs **one** ``n x d @ d x d`` product plus one ``n x d @ d x k``
        product for the whole batch -- k times less arithmetic and no k-loop.
        ``R`` is a square root of the precision, not the precision itself:
        ``|xP|^2`` would be ``x P P^T x^T``, the square of the quadratic form,
        which is wrong by however much ``P`` fails to be idempotent.  The ``full``
        family cannot use this (each component has its own Sigma) and keeps the
        per-component solve.

        Measured on the 1M-row LAION reproduction (tied, k=64, d=320, batch
        8192): 6.1 s per update to well under one, because the E-step was the
        whole cost and it was doing 64 redundant ``d x d`` solves.
        """
        d = x.shape[1]
        cov = np.asarray(cov, dtype=np.float64)
        cov = 0.5 * (cov + cov.T)
        # A square root of the precision, not the precision itself: ``root @
        # root.T`` must equal ``cov^-1``, and ``L^-T`` gives exactly that from
        # ``cov = L L^T`` with one triangular solve, while ``logdet`` falls out of
        # ``diag(L)``.  This path used an explicit ``inv`` and paid for it: 1.3 s
        # of a 1.4 s tied scoring pass at d=512 on this host, where ``getrf`` and
        # ``syevd`` are ~250x slower than ``potrf`` (measured: inv 483 ms,
        # cholesky 1.9 ms, solve(I) 512 ms, eigh 1396 ms).  Swapping ``inv`` for
        # ``solve`` would not have helped -- ``gesv`` goes through ``getrf`` -- so
        # the fix has to stay on the Cholesky route.
        root, logdet = self._tied_root(cov)
        # Shift both sides by the mean of the means before expanding.  The
        # expansion is exact either way, but uncentred it subtracts two numbers of
        # size |x R|^2 to leave a quadratic form of size d: with a mean far from
        # zero and a low-rank covariance (so R is large) that cancellation costs
        # real digits -- measured 0.22 nats on the rank-6 test below.  After the
        # shift both terms are the size of the quadratic form itself.
        centre = means.mean(axis=0)
        means_p = (means - centre) @ root             # (k, d)
        means_q = np.einsum("ij,ij->i", means_p, means_p)
        n = x.shape[0]
        # One reusable block instead of an (n, d) projection per call: the
        # old form allocated 4 MB at 8192 x 64 and measured 2.43 ms against
        # 1.29 chunked (4096 x 128: 2.67 against 1.54).  Subtracting the centre
        # before the product keeps the offset cancellation fix above intact.
        rows = max(256, min(2048, (1 << 21) // max(8 * d, 1)))
        work = np.empty((min(rows, n), d), dtype=np.float64)
        projected = np.empty_like(work)
        out = np.empty((n, means.shape[0]), dtype=np.float64)
        const = -0.5 * (d * _LOG_2PI + logdet)
        for start in range(0, n, rows):
            end = min(start + rows, n)
            size = end - start
            block = work[:size]
            proj = projected[:size]
            np.subtract(x[start:end], centre, out=block)
            np.matmul(block, root, out=proj)
            quad = (
                np.einsum("ij,ij->i", proj, proj)[:, None]
                - 2.0 * (proj @ means_p.T)
                + means_q[None, :]
            )
            # a quadratic form is non-negative by construction; round-off can
            # put a point exactly on a mean a few ulps below zero
            np.maximum(quad, 0.0, out=quad)
            out[start:end] = const - 0.5 * quad
        return out

    def component_log_prob(self, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """(n, k) log N(x | mean_k, cov_k), covariance-family specific."""
        means = params["means"]
        family = params.get("covariance", self.covariance)
        covs = np.asarray(params["covariances"], dtype=np.float64)
        d = x.shape[1]
        if family == "lowrank":
            return self._lowrank_log_prob(x, params)
        if family == "tied" and covs.ndim == 2:
            kernels = _host_kernels()
            if (kernels and getattr(kernels, "tied_available", lambda: False)()
                    and np.shape(x)[1] == means.shape[1]):
                got = kernels.tied_logits(x, means, covs)
                if got is not None:
                    return got
            return self._shared_log_prob(x, covs, means)
        if family == "full":
            # One batched LAPACK call for all k components.  The component loop
            # issued k separate solve() calls per E-step (80 for a 5-step k=8
            # fit with per-update scoring), and the per-call overhead was most
            # of the wall clock at d=8.  The stacked solve is the same
            # arithmetic, so the answer is unchanged.
            #
            # Whitening beats that solve at this shape.  ``solve`` against
            # (k, d, n) factors each d x d matrix and then sweeps n
            # right-hand sides without blocking on the solution side: measured
            # 2.13 ms per call at k=d=8, n=3000, which was most of the
            # full-covariance gap against sklearn.  Whitening with each
            # component's own inverse Cholesky factor is the same quadratic
            # form as one d x d matmul per component: 0.40 ms measured, max abs
            # difference 1.3e-13 nats.  Anything non-positive-definite falls
            # through to the solve path, which already handles it.
            k_comp = means.shape[0]
            if covs.ndim == 3 and covs.shape[0] == k_comp:
                try:
                    chol = np.linalg.cholesky(covs)
                except np.linalg.LinAlgError:
                    chol = None
                if chol is not None:
                    logdets = 2.0 * np.log(
                        np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
                    root = np.linalg.inv(chol)
                    # Row chunks keep the whitened (rows, d) block in cache.
                    # The old form allocated a fresh (n, d) difference per
                    # component: 3.56 -> 2.99 ms at 8192 x 32, 4.34 -> 3.80 at
                    # 4096 x 64, 11.78 -> 10.44 at 4096 x 128.  The arithmetic
                    # is element-for-element the same.
                    n = x.shape[0]
                    rows = max(256, min(2048, (1 << 21) // max(8 * d, 1)))
                    diff = np.empty((min(rows, n), d), dtype=np.float64)
                    whitened = np.empty((n, k_comp), dtype=np.float64)
                    for j in range(k_comp):
                        const = -0.5 * (d * _LOG_2PI + logdets[j])
                        for start in range(0, n, rows):
                            end = min(start + rows, n)
                            block = diff[: end - start]
                            np.subtract(x[start:end], means[j], out=block)
                            z = block @ root[j].T
                            whitened[start:end, j] = const - 0.5 * np.einsum(
                                "ij,ij->i", z, z)
                    return whitened
                diff = x[None, :, :] - means[:, None, :]        # (k, n, d)
                diff_t = np.swapaxes(diff, 1, 2)                # (k, d, n)
                signs, logdets = np.linalg.slogdet(covs)
                if np.all(signs > 0):
                    sol = np.linalg.solve(covs, diff_t)
                    quad = np.einsum("kdn,kdn->kn", diff_t, sol)
                    return (-0.5 * (d * _LOG_2PI + logdets[:, None] + quad)).T
            out = np.empty((x.shape[0], means.shape[0]), dtype=np.float64)
            for j in range(means.shape[0]):
                cov = covs[j]
                sign, logdet = np.linalg.slogdet(cov)
                if sign <= 0:  # numerical breakdown: fall back to a diagonal solve
                    cov = np.diag(np.maximum(
                        np.diag(cov), self.reg_covar_floor(cov)))
                    sign, logdet = np.linalg.slogdet(cov)
                diff = x - means[j]
                sol = np.linalg.solve(cov, diff.T).T
                quad = np.einsum("ij,ij->i", diff, sol)
                out[:, j] = -0.5 * (d * _LOG_2PI + logdet + quad)
            return out
        k_comp, dim = means.shape
        kernels = _host_kernels()
        if (covs.ndim == 2 and covs.shape == (k_comp, dim) and kernels
                and kernels.logits_available()):
            # The compiled single-pass kernel (T21): same expression, one pass,
            # no (n, k) temporaries.  Falls through to BLAS below whenever the
            # artifact is absent, the family is not a plain k x d diagonal, or
            # EMAOPT_HOST_KERNEL=numpy.
            x_c = np.ascontiguousarray(x, dtype=np.float64)
            if x_c.shape[1] == dim:
                return kernels.diag_logits(x_c, means, covs)
        # Only the BLAS paths below need the reciprocal: the compiled kernel
        # takes the covariances directly, so this is not paid on the fast path.
        inv_var = 1.0 / covs
        if covs.ndim == 2:
            # Diagonal/spherical quadratic form through BLAS.  Expanding
            # ``sum_d (x_d - mu_jd)^2 / var_jd`` avoids the old n*k*d
            # difference tensor, which was ~2.1 GB at 8192x64x512 and made
            # the E-step memory-bound.  The two products are exact
            # rearrangements; measured max abs difference against the
            # broadcast form is 1.4e-12 nats, with a ~50x E-step speedup.
            mean_inv = means * inv_var
            quad = (x * x) @ inv_var.T - 2.0 * (x @ mean_inv.T)
            quad += np.einsum("kd,kd->k", means, mean_inv)[None, :]
            np.maximum(quad, 0.0, out=quad)
            logdet = np.sum(np.log(covs), axis=1)[None, :]
            return -0.5 * quad - 0.5 * (d * _LOG_2PI + logdet)
        # One variance vector shared by every component; this path is rare
        # and the broadcast form is already small.
        diff = x[:, None, :] - means[None, :, :]
        quad = np.einsum("nkd,d->nk", diff * diff, inv_var)
        logdet = float(np.sum(np.log(covs)))
        return -0.5 * quad - 0.5 * (d * _LOG_2PI + logdet)

    def predict_proba(
        self, x: np.ndarray, params: dict[str, Any] | None = None
    ) -> np.ndarray:
        """Posterior component probabilities, rows summing to one.

        This is the calibrated-probability input for :mod:`emaopt.calibration`.
        The decoded parameters are required; a fitted model is represented by
        the ``params`` returned from :meth:`fit`, not stored on the optimizer.
        """
        if params is None:
            raise ValueError(
                "params are required; pass the params from GaussianMixture.fit"
            )
        return self.responsibilities(np.asarray(x, dtype=np.float64), params)

    def responsibilities(self, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """(n, k) responsibilities, rows summing to one."""
        logp = self.component_log_prob(x, params)
        logw = np.log(np.maximum(params["weights"], 1e-300))
        logits = logp + logw[None, :]
        log_norm = logsumexp_rows(logits)
        out = np.exp(logits - log_norm[:, None])
        out[~np.isfinite(out)] = 0.0
        total = out.sum(axis=1, keepdims=True)
        return out / np.maximum(total, 1e-300)

    def statistics_from(self, data: np.ndarray, params: dict[str, Any]) -> State:
        """The state :meth:`measure` would produce for ``data`` under ``params``.

        Split out from ``measure`` so the same reduction can be applied without a
        running state: a single exact M-step on data the fit never saw (see
        :meth:`refine`), or a construction from labels.

        On the NumPy path this builds one ``(n, d)`` ``data**2`` temporary for
        the second moment, separately from the ``x * x`` the log-density needs.
        Sharing them would mean threading a buffer through two public methods
        for one transient at batch size; the compiled kernel (T21) already fuses
        both into its single pass, so the shipped diag/spherical fit does not pay
        it.  Recorded in T34 rather than half-fixed here.
        """
        data = np.asarray(data, dtype=np.float64)
        device = _device_kernels()
        if device and self.covariance in ("diag", "spherical"):
            covs = np.asarray(params["covariances"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            work = int(data.shape[0]) * int(means.shape[0]) * int(means.shape[1])
            if (covs.ndim == 2 and covs.shape == means.shape
                    and data.ndim == 2 and data.shape[1] == means.shape[1]
                    and work >= device.work_gate()):
                got = self._device_statistics(
                    device, data, means, covs, params.get("weights"))
                return State(
                    {"counts": got["counts"], "sums": got["sums"],
                     "sq": got["sq"]},
                    0,
                    {"nll": got["nll"],
                     "device": got.get("precision", "cuda"),
                     "device_resident": bool(got.get("resident"))},
                )
        tied_device = _tied_device_kernels()
        if tied_device and self.covariance == "tied":
            covs = np.asarray(params["covariances"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            work = int(data.shape[0]) * int(means.shape[0]) * int(means.shape[1])
            if (covs.ndim == 2 and data.ndim == 2
                    and covs.shape == (data.shape[1], data.shape[1])
                    and means.shape[1] == data.shape[1]
                    and work >= tied_device.WORK_GATE):
                got = self._tied_device_statistics(
                    tied_device, data, means, covs, params.get("weights"))
                return State(
                    {"counts": got["counts"], "sums": got["sums"],
                     "gscatter": got["gscatter"]},
                    0,
                    {"nll": got["nll"], "device": got.get("device"),
                     "precision": got.get("precision")},
                )
        kernels = _host_kernels()
        if (kernels and self.covariance == "tied"
                and getattr(kernels, "tied_available", lambda: False)()):
            covs = np.asarray(params["covariances"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            if (covs.ndim == 2 and data.ndim == 2
                    and covs.shape == (data.shape[1], data.shape[1])
                    and means.shape[1] == data.shape[1]):
                got = kernels.tied_estep(
                    data, means, covs, params.get("weights"))
                return State(
                    {"counts": got["counts"], "sums": got["sums"],
                     "gscatter": got["gscatter"]},
                    0,
                    {"nll": got["nll"]},
                )
        if (kernels and self.covariance in ("diag", "spherical")
                and kernels.available()):
            covs = np.asarray(params["covariances"], dtype=np.float64)
            means = np.asarray(params["means"], dtype=np.float64)
            if (covs.ndim == 2 and covs.shape == means.shape
                    and data.ndim == 2 and data.shape[1] == means.shape[1]):
                got = kernels.diag_estep(
                    data, means, covs, params.get("weights"))
                return State(
                    {"counts": got["counts"], "sums": got["sums"],
                     "sq": got["sq"]},
                    0,
                    {"nll": got["nll"]},
                )
        # one E-step, then both the responsibilities and the NLL from it: the
        # earlier form called component_log_prob twice per update (once inside
        # responsibilities, once for the objective), which doubled the most
        # expensive term in the loop
        logits = self.component_log_prob(data, params) + np.log(
            np.maximum(params["weights"], 1e-300)
        )[None, :]
        log_norm = logsumexp_rows(logits)
        resp = np.exp(logits - log_norm[:, None])
        resp[~np.isfinite(resp)] = 0.0
        resp /= np.maximum(resp.sum(axis=1, keepdims=True), 1e-300)
        counts = resp.sum(axis=0)
        sums = resp.T @ data
        values: dict[str, np.ndarray] = {"counts": counts, "sums": sums}
        if self.covariance == "full":
            values["scatter"] = _weighted_outer(resp, data)
        elif self.covariance == "lowrank":
            z = data @ self._sketch_projection(data.shape[1])
            values["sq"] = resp.T @ (data**2)
            # same einsum-to-GEMM change as ``_weighted_outer``: a component at
            # a time, so each moment is a product rather than a naive triple loop
            sz = np.empty((resp.shape[1], z.shape[1], z.shape[1]))
            sxz = np.empty((resp.shape[1], data.shape[1], z.shape[1]))
            for j in range(resp.shape[1]):
                zj = z * resp[:, j:j + 1]
                sz[j] = z.T @ zj
                sxz[j] = data.T @ zj
            values["sz"] = sz
            values["sxz"] = sxz
            values["sz1"] = resp.T @ z
        elif self.covariance == "tied":
            # a point's responsibilities sum to one, so the global second moment
            # needs no per-component weighting
            values["gscatter"] = data.T @ data
        else:
            values["sq"] = resp.T @ (data**2)
        nll = -float(log_norm.mean())
        return State(values, 0, {"nll": nll})

    def _tied_device_statistics(self, device: Any, data: np.ndarray,
                                means: np.ndarray, covs: np.ndarray,
                                weights: np.ndarray | None) -> dict[str, Any]:
        """Resident tied E-step, keyed on the batch's storage identity.

        A fit measures the same batch with changing parameters many times, so
        keeping the rows and captured kernel alive turns every step after the
        first into a few-KB parameter upload plus one launch.
        """
        session_cls = getattr(device, "Session", None)
        # The precision is fixed when the session is built, so it belongs in the
        # key: without it, changing ``EMAOPT_TIED_PRECISION`` between two fits on
        # the same model and batch silently kept the first mode's session.
        key = (id(device),
               device.precision() if hasattr(device, "precision") else "",
               int(data.ctypes.data), data.shape, data.strides, data.dtype.str,
               int(means.shape[0]))
        session = getattr(self, "_tied_device_session", None)
        if session is None or getattr(self, "_tied_session_key", None) != key:
            if session is not None:
                session.close()
            session = session_cls(data, int(means.shape[0]),
                                  precision=device.precision())
            self._tied_device_session = session
            self._tied_session_key = key
            self._tied_session_data = data
        return session.estep(means, covs, weights)

    def _device_statistics(self, device: Any, data: np.ndarray,
                           means: np.ndarray, covs: np.ndarray,
                           weights: np.ndarray | None) -> dict[str, Any]:
        """One device E-step, reusing a resident session when the data holds.

        A fit calls this with the same ``data`` array once per epoch, so the
        data upload and the buffer allocation can be paid once instead of per
        step.  The session is keyed on the array's address, shape, strides and
        dtype: a changed batch opens a new one.  ``EMAOPT_DEVICE_RESIDENT=0``
        falls back to the single-shot wrapper.
        """
        session_cls = getattr(device, "Session", None)
        disabled = os.environ.get("EMAOPT_DEVICE_RESIDENT", "").strip().lower()
        if session_cls is None or disabled in {"0", "off", "false", "no", "numpy"}:
            return device.estep(data, means, covs, weights)
        key = (int(data.ctypes.data), data.shape, data.strides, data.dtype.str,
               int(means.shape[0]))
        session = getattr(self, "_device_session", None)
        if session is None or getattr(self, "_device_session_key", None) != key:
            if session is not None:
                session.close()
            session = session_cls(data, int(means.shape[0]))
            self._device_session = session
            self._device_session_key = key
            # The key is the host address, so the host array must outlive the
            # session.  Without this reference a freed batch's address can be
            # recycled by the next one, and the cache would return statistics
            # computed from the previous batch under the same key.
            self._device_session_data = data
        return session.estep(means, covs, weights)

    def measure(self, batch: np.ndarray, state: State) -> State:
        return self.statistics_from(batch, self.decode(state))

    def refine(self, data: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """One exact M-step on ``data`` under ``params``' responsibilities.

        This is what the e-means paper calls the slower finalization for test
        data: the mixture is re-estimated on the data it is about to be scored
        on, using its own responsibilities, without any gradient or pass over the
        training set.  Only unlabelled data is used, so it is a transductive
        refinement rather than leakage -- score the refined parameters on the same
        data and the comparison is "mixture before vs after one M-step", which is
        the question.

        On data drawn from the same distribution, one step from a fitted mixture
        is close to a no-op (it is near a fixed point); on a shifted stream it is
        the same mechanism ``finalize=True`` uses during training.
        """
        stats = self.statistics_from(data, params)
        fallback = np.repeat(
            np.asarray(params["means"], dtype=np.float64).mean(axis=0)[None, :],
            self.n_components,
            axis=0,
        )
        return self.decode(State(stats.values, 0, {"fallback_means": fallback}))

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean negative log-likelihood of the mixture (nats per point)."""
        data = np.asarray(data, dtype=np.float64)
        logits = self.component_log_prob(data, parameters) + np.log(
            np.maximum(parameters["weights"], 1e-300)
        )[None, :]
        return -float(logsumexp_rows(logits).mean())

    # -- convenience -------------------------------------------------------- #
    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int | None = None,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
        enforce_identifiability: bool = True,
        drop_last: bool | None = None,
        tol: float | None = 1e-3,
    ) -> FitResult:
        # Keep the final partial batch for streaming/diagonal families: the old
        # default silently discarded up to one batch per pass (30k rows / 8192
        # meant 18% of the data and only six EMA updates).  Full covariance is
        # the exception: a short final batch can put every component below its
        # identification floor and decode the whole model as diagonal.  Callers
        # can override either choice explicitly.
        if drop_last is None:
            drop_last = self.covariance == "full"
        cfg = FitConfig(
            batch_size=batch_size,
            nepochs=self.max_epochs if nepochs is None else int(nepochs),
            seed=seed,
            track_memory=track_memory,
            enforce_identifiability=enforce_identifiability,
            drop_last=drop_last,
            # ``max_epochs`` is a cap, not a cost.  Measured on the 3000 x 8,
            # k=8 bench fixture: |d NLL| <= 1e-3 stops at epoch 4 for
            # spherical/diag/tied and 7 for full, against 60 forced epochs at an
            # identical third digit of NLL.  The reference stops on convergence
            # too (sklearn's tol), so without this the default arm pays 12-15x
            # the reference's iteration count for a ~0.001-nat difference -- a
            # stopping rule, not a kernel.  ``tol=None`` restores the full
            # budget, and ``extras['converged_at_update']`` records the stop.
            tol=tol,
        )
        return fit(self, data, cfg, eval_data=eval_data)

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        """Bytes of the persistent statistics, including the sketch Gramian.

        The plan resolver uses this to decide whether a batch must be streamed.
        ``lowrank`` is often the largest state even though it is the cheapest
        wide family to *decode*: it keeps ``k*d*r`` cross moments in addition to
        the diagonal moments and the ``r*r`` sketch Gramian.
        """
        k = self.n_components if k is None else int(k)
        base = k + k * d
        if self.covariance == "full":
            extra = k * d * d
        elif self.covariance == "tied":
            extra = d * d
        elif self.covariance == "lowrank":
            r = self.rank if self.rank is not None else max(2, d // 8)
            r = max(1, min(int(r), int(d)))
            extra = k * d + k * d * r + k * r * r + k * r
        else:
            extra = k * d
        # finalize keeps a second copy of the accumulated statistics in meta
        copies = 2 if self.finalize else 1
        return int(8 * copies * (base + extra))

    def identification_min_batch(self, d: int) -> int:
        """One update must carry ~``5d`` points per component for ``full``.

        Below that the decode falls back to the diagonal for every component, so
        the batch only pays for the full state's ``O(k d^2)`` cost without using
        the correlations it exists for.  Other families have no such floor: they
        spend ``O(k d)`` and stay estimable from small batches.
        """
        if self.covariance != "full":
            return 0
        return int(np.ceil(_FULL_POINTS_PER_DIM * int(d))) * self.n_components

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_components": self.n_components, "covariance": self.covariance,
                    # the floor is two numbers, and a record that omits them
                    # cannot say which one a fit was run with
                    "reg_covar": self.reg_covar,
                    "reg_covar_relative": self.reg_covar_relative,
                    "init": self.init, "init_scale": self.init_scale,
                    "init_sample": self.init_sample,
                    "n_init": self.n_init, "max_epochs": self.max_epochs,
                    # ``extras["seed"]`` is the batch-order seed from
                    # ``FitConfig``; the initialization seed is the model's own
                    # and was missing from records entirely.
                    "seed": int(getattr(self, "seed", 0))})
        if self.revival:
            out["revival"] = {
                "threshold": self.revival_threshold,
                "split": self.revival_split,
            }
        if self.finalize:
            out["finalize"] = True
        if self.covariance == "lowrank":
            out["rank"] = int(self._rank or (self.rank or 0))
        return out


def em_gmm(
    data: np.ndarray,
    n_components: int,
    *,
    covariance: str = "diag",
    niter: int = 50,
    seed: int = 0,
) -> FitResult:
    """Conventional batch EM (Replace, full batch) -- the reference baseline."""
    model = GaussianMixture(n_components, covariance=covariance, seed=seed)
    return model.fit(data, batch_size=None, nepochs=niter, seed=seed)
