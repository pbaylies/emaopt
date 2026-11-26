"""Sufficient statistics as first-class objects (ONLINE_PLAN phase 1).

The library already abstracts *how* a statistic is accumulated behind
``StateUpdater``.  This module abstracts *what* is being accumulated: a
:class:`SufficientStat` is an object that

* folds a block of data into itself (:meth:`SufficientStat.update`),
* merges with another accumulator of the same kind (:meth:`SufficientStat.merge`),
* decodes parameters from what it holds (:meth:`SufficientStat.decode`),
* states its memory footprint in bytes (:attr:`SufficientStat.nbytes`), and
* serialises its configuration (:meth:`SufficientStat.config`).

The contract is deliberately small.  Every algorithm in the toolbox decomposes
this way -- ``sums``/``counts`` for centroids, a Gramian for PCA, ``X^T Z`` and
``Z^T Z`` for NMF/dictionary learning, word-topic counts for LDA, a
nonlinear cross moment for ICA -- so a stat is the same object at one batch and
at a thousand.

Two accumulation laws are supported, matching the two regimes the library
already measures:

``mode="sum"``
    ``values <- values + block_statistic``: every block carries its own weight.
    This is the streaming/one-pass reduction; merging partitions is exact up to
    floating-point summation order.

``mode="ema"``
    ``values <- (1 - alpha) values + alpha block_statistic``: recent blocks
    dominate, and ``alpha`` may be a
    :class:`~emaopt.state.schedules.Schedule`.  This is the e-means law applied
    to whatever statistic the stat owns.  Merging two decayed histories is a
    superposition of them, which agrees with the single stream up to the law's
    own error -- the acceptance tolerance the plan states, not an exact
    identity.

``act_every`` is what makes the protocol fit ICA (and dictionary learning):
statistics are accumulated on every block, but the parameters derived from them
are refreshed only every ``act_every`` blocks, so accumulation and the expensive
parameter step have independent frequencies.  ``fastica`` uses
``act_every=1``; the TD form can act less often.

Stats are array-only and have no data reference: a stat can be pickled, sent to
a worker, merged, and decoded without the training set.
"""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Iterable, Mapping

import numpy as np

from ..core import State, UpdateContext, label_sums, squared_distances

_EPS = 1e-12
_INIT_PRIOR = 2.0 ** -16


def _digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Stable hash of a parameter set, for the merge compatibility check."""
    h = hashlib.sha1()
    for key in sorted(arrays):
        arr = np.ascontiguousarray(arrays[key], dtype=np.float64)
        h.update(key.encode())
        h.update(repr(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# protocol
# --------------------------------------------------------------------------- #
class SufficientStat:
    """A mergeable, decodable estimate of one algorithm's statistic.

    Subclasses implement :meth:`block_statistic` and :meth:`decode`; the base
    owns the accumulation law, the merge, the step counter and the reporting
    surface.
    """

    name: str = "sufficient_stat"
    keys: tuple[str, ...] = ()
    #: "sum" (additive, exact merge) or "ema" (decayed, merge is a superposition)
    mode: str = "sum"

    def __init__(
        self,
        *,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
        updater: Any = None,
    ) -> None:
        if mode not in ("sum", "ema"):
            raise ValueError(f"unknown accumulation mode {mode!r}")
        self.mode = mode
        #: Optional law object that owns the "ema" fold (T48).  Giving one makes
        #: the stat and a fit share the *same* law instead of two transcriptions
        #: of it; a bare alpha/schedule is wrapped in an EMA on first use, which
        #: is exactly what a fit does with the same argument.
        self.updater = updater
        self._ema_law: Any = None
        if updater is not None:
            self.mode = "ema"
        if isinstance(alpha, str):
            from .schedules import make_schedule

            alpha = make_schedule(alpha)
        self.alpha = alpha
        self.act_every = max(1, int(act_every))
        self.seed = int(seed)
        self.values: dict[str, np.ndarray] = {}
        self.params: dict[str, Any] = {}
        self.meta: dict[str, Any] = {}
        self.weight: float = 0.0
        self.n_examples: int = 0
        self.steps: int = 0
        self.total_steps: int | None = None
        self.dataset_size: int | None = None
        self.last_alpha: float = float("nan")
        self._last_delta: float = float("nan")
        #: parameters are stale w.r.t. the accumulated values (set by a fold that
        #: did not act, and by a merge; cleared by act).  Without this, reducing
        #: four partitions would act three times on the way down and land on a
        #: different fixed point than the single stream's one act.
        self._dirty: bool = False

    # -- accumulation ------------------------------------------------------- #
    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        """Reduce one block to the arrays this stat accumulates."""
        raise NotImplementedError

    def prepare(self, data: np.ndarray | None = None) -> "SufficientStat":
        """Seed parameters/values before the first block (data may be unused)."""
        return self

    def configure(
        self, *, total_steps: int | None = None, dataset_size: int | None = None
    ) -> "SufficientStat":
        """Announce the planned horizon so schedules can resolve progress."""
        self.total_steps = total_steps
        self.dataset_size = dataset_size
        return self

    def _context(self, block_weight: float) -> UpdateContext:
        """The context a law sees for this fold -- the same one a fit would build."""
        return UpdateContext(
            step=self.steps,
            batch_size=max(1, int(block_weight)),
            dataset_size=max(1, int(self.dataset_size or block_weight)),
            total_steps=max(1, int(self.total_steps or 1)),
        )

    def _law(self) -> Any:
        """The updater that resolves ``mode="ema"`` (one law for stat and fit)."""
        if self.updater is not None:
            return self.updater
        if self._ema_law is None:
            from .ema import EMA        # local: ema imports affine, not this module

            self._ema_law = EMA(self.alpha)
        return self._ema_law

    def step_alpha(self, block_weight: float) -> float:
        """Scalar ``a`` for this fold (the per-key law lives in :meth:`_law`)."""
        alpha = self.alpha
        if isinstance(alpha, (int, float, np.floating)):
            return float(np.clip(alpha, 0.0, 1.0))
        ctx = UpdateContext(
            step=self.steps,
            batch_size=max(1, int(block_weight)),
            dataset_size=max(1, int(self.dataset_size or block_weight)),
            total_steps=max(1, int(self.total_steps or 1)),
        )
        if hasattr(alpha, "alpha"):
            value = alpha.alpha(ctx)
        else:  # pragma: no cover - duck-typed callable schedule
            value = alpha(ctx)
        return float(np.clip(value, 0.0, 1.0))

    def _fold(self, statistics: Mapping[str, np.ndarray], block_weight: float) -> None:
        keys = tuple(statistics)
        if self.mode == "sum":
            a = self.step_alpha(block_weight)
            self.last_alpha = a
            for key, value in statistics.items():
                value = np.asarray(value, dtype=np.float64)
                if key not in self.values:
                    self.values[key] = np.array(value, copy=True)
                else:
                    self.values[key] = self.values[key] + value
            if self.weight == 0.0:
                self.weight = float(block_weight)
            else:
                self.weight += float(block_weight)
        else:
            # ``mode="ema"`` is the EMA law, not a second copy of it: the law
            # resolves the per-key alphas (schedules, per_key, ESS, weights) and
            # the fold below is the same one ``AffineUpdater.update`` runs.
            law = self._law()
            ctx = self._context(block_weight)
            # A key with no previous value of the right shape takes the reset
            # branch below, so the law has to see it as *fresh*: a law whose step
            # depends on a per-key counter would otherwise count that key twice
            # (once by the reset, once by the counter) and make the next block's
            # alpha too small.  ``AffineUpdater.update`` computes this the same
            # way.
            fresh = frozenset(
                key for key in keys
                if key not in self.values
                or np.shape(self.values[key]) != np.shape(statistics[key])
            )
            alphas_for = getattr(law, "alphas_for", None)
            if alphas_for is None:            # a plain StateUpdater: one alpha
                a = float(np.clip(law.alpha(ctx), 0.0, 1.0))
                alphas = {key: a for key in keys}
            else:
                alphas, meta = alphas_for(ctx, keys, dict(self.meta), fresh)
                self.meta = meta
            for key, value in statistics.items():
                value = np.asarray(value, dtype=np.float64)
                a = float(np.clip(alphas.get(key, 1.0), 0.0, 1.0))
                if key not in self.values:
                    self.values[key] = np.array(value, copy=True)
                else:
                    self.values[key] = (1.0 - a) * self.values[key] + a * value
            a = float(np.clip(alphas[keys[0]], 0.0, 1.0)) if keys else 1.0
            self.last_alpha = a
            if self.weight == 0.0:
                self.weight = float(block_weight)
            else:
                # The stat's weight follows the alpha blend, as it always has in
                # ema mode; a law's own accumulated mass (WeightedMean's ``_w``,
                # RunningMean's ``_n``) lives in meta rather than here.
                self.weight = (1.0 - a) * self.weight + a * float(block_weight)
        self.n_examples += int(block_weight)
        self.steps += 1
        self._dirty = True

    def update(self, block: np.ndarray, *, weight: float | None = None) -> "SufficientStat":
        """Fold one block in and refresh parameters when an act is due."""
        block = np.asarray(block, dtype=np.float64)
        w = float(block.shape[0] if weight is None else weight)
        if w <= 0.0:
            return self
        statistics = self.block_statistic(block, w)
        self._fold(statistics, w)
        if self.steps % self.act_every == 0:
            self.act()
        return self

    def act(self) -> None:
        """Derive parameters from the accumulated statistic (default: none)."""
        self._dirty = False

    def _maybe_act(self) -> None:
        """Refresh derived parameters if the statistic moved since the last act."""
        if self._dirty:
            self.act()

    # -- decoding ----------------------------------------------------------- #
    def decode(self) -> Any:
        """Turn the accumulated statistic into parameters."""
        raise NotImplementedError

    # -- merge algebra ------------------------------------------------------ #
    def copy(self) -> "SufficientStat":
        clone = copy.copy(self)
        clone.values = {k: np.array(v, copy=True) for k, v in self.values.items()}
        clone.params = {
            k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else copy.deepcopy(v))
            for k, v in self.params.items()
        }
        clone.meta = copy.deepcopy(self.meta)
        return clone

    def measurement_fingerprint(self) -> str | None:
        """Hash of the parameters this statistic was *measured* under.

        ``None`` means the stat has no such notion (a pure moment accumulator
        such as ``GramianStat`` measures nothing that depends on parameters).
        """
        return None

    def _check_compatible(self, other: "SufficientStat") -> None:
        if type(other) is not type(self):
            raise TypeError(f"cannot merge {type(other).__name__} into {type(self).__name__}")
        mine, theirs = self.measurement_fingerprint(), other.measurement_fingerprint()
        if mine is not None and theirs is not None and mine != theirs:
            # Two partitions of one dataset self-seeded from their own first
            # block have different centroids/parameters; adding their statistics
            # describes no single estimator, and the merged decode can land near
            # zero (measured: two derivations of a 3-blob set merged to ~0 while
            # the single stream keeps +-6).  Refuse instead of returning that.
            raise ValueError(
                f"cannot merge {type(self).__name__} statistics measured under "
                "different parameters: seed every partition with the same "
                "`prepare=` block (or fixed parameters), or keep them separate"
            )
        for key, value in self.values.items():
            if key in other.values and np.shape(other.values[key]) != np.shape(value):
                raise ValueError(
                    f"stat {key} shape {np.shape(other.values[key])} != {np.shape(value)}"
                )

    def merge(self, other: "SufficientStat") -> "SufficientStat":
        """Combine two accumulators of the same kind into a new one.

        Values and weights add.  For ``mode="sum"`` this is the exact reduction;
        for ``mode="ema"`` the decayed histories superpose, which is the
        approximation the plan's merge gate is stated against.
        """
        self._check_compatible(other)
        merged = self.copy()
        for key, value in other.values.items():
            if key in merged.values:
                merged.values[key] = merged.values[key] + np.asarray(value, dtype=np.float64)
            else:
                merged.values[key] = np.array(value, copy=True)
        merged.weight = self.weight + other.weight
        merged.n_examples = self.n_examples + other.n_examples
        merged.steps = self.steps + other.steps
        merged._dirty = True
        return merged

    @classmethod
    def merge_all(cls, stats: Iterable["SufficientStat"]) -> "SufficientStat":
        iterator = iter(stats)
        try:
            first = next(iterator)
        except StopIteration as exc:  # pragma: no cover - defensive
            raise ValueError("merge_all needs at least one stat") from exc
        merged = first.copy()
        for other in iterator:
            merged = merged.merge(other)
        return merged

    # -- reset -------------------------------------------------------------- #
    def reset(self) -> None:
        self.values = {}
        self.params = {}
        self.weight = 0.0
        self.n_examples = 0
        self.steps = 0
        self.last_alpha = float("nan")
        self._last_delta = float("nan")
        self._dirty = False

    def reset_accumulation(self) -> "SufficientStat":
        """Clear the accumulated statistic but keep decoded parameters.

        Epoch-style EM accumulates one epoch's sufficient statistics under
        fixed parameters, takes one M-step, then needs the accumulators empty
        for the next epoch.  Parameters (and metadata) survive, unlike
        :meth:`reset`.
        """
        self.values = {}
        self.weight = 0.0
        self.n_examples = 0
        self.steps = 0
        self.last_alpha = float("nan")
        self._dirty = True
        return self

    # -- reporting ---------------------------------------------------------- #
    @property
    def nbytes(self) -> int:
        total = 0
        for group in (self.values, self.params):
            for value in group.values():
                if isinstance(value, np.ndarray):
                    total += int(value.nbytes)
        return total

    def memory_report(self) -> dict[str, int]:
        out = {"total_bytes": int(self.nbytes), "values_bytes": 0, "params_bytes": 0}
        for key, value in self.values.items():
            if isinstance(value, np.ndarray):
                out["values_bytes"] += int(value.nbytes)
                out[f"value:{key}"] = int(value.nbytes)
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                out["params_bytes"] += int(value.nbytes)
                out[f"param:{key}"] = int(value.nbytes)
        return out

    def _alpha_config(self) -> Any:
        alpha = self.alpha
        if isinstance(alpha, (int, float, np.floating)):
            return float(alpha)
        if hasattr(alpha, "alpha0"):
            return {"schedule": type(alpha).__name__, "alpha0": float(alpha.alpha0)}
        return type(alpha).__name__

    def extra_config(self) -> dict[str, Any]:
        return {}

    def config(self) -> dict[str, Any]:
        """JSON-serialisable description for experiment records."""
        out: dict[str, Any] = {
            "stat": self.name,
            "mode": self.mode,
            "alpha": self._alpha_config(),
            "act_every": int(self.act_every),
            "keys": list(self.values) or list(self.keys),
            "steps": int(self.steps),
            "n_examples": int(self.n_examples),
            "weight": float(self.weight),
            "shapes": {k: list(np.shape(v)) for k, v in self.values.items()},
            "nbytes": int(self.nbytes),
            "seed": int(self.seed),
        }
        if self.updater is not None:
            out["updater"] = getattr(self.updater, "name",
                                     type(self.updater).__name__)
        out.update(self.extra_config())
        return out

    # -- array bridge ------------------------------------------------------- #
    def to_state(self) -> State:
        return State({k: np.array(v, copy=True) for k, v in self.values.items()},
                     self.steps, copy.deepcopy(self.meta))


# --------------------------------------------------------------------------- #
# centroid stat: k-means / VQ
# --------------------------------------------------------------------------- #
class CentroidStat(SufficientStat):
    """Per-cluster first moment and counts -- the e-means statistic.

    Values are ``sums`` (k, d) and ``counts`` (k,), exactly the arrays
    :class:`~emaopt.algorithms.kmeans.KMeans` keeps, so ``mode="ema"`` with
    ``alpha=1`` reduces a block to one Lloyd rebuild and ``mode="sum"`` is the
    streaming reduction of the same statistic.
    """

    name = "centroid_stat"
    keys = ("sums", "counts")

    def __init__(
        self,
        k: int,
        d: int | None = None,
        *,
        centroids: np.ndarray | None = None,
        init: str = "kmeans++",
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
        prior: float = _INIT_PRIOR,
        updater: Any = None,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every, seed=seed,
                         updater=updater)
        self.k = int(k)
        self.d = None if d is None else int(d)
        self.init = init
        self.prior = float(prior)
        self._given = None if centroids is None else np.asarray(centroids, dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def _kmeanspp(self, block: np.ndarray) -> np.ndarray:
        n, d = block.shape
        k = self.k
        if k > n:
            raise ValueError(f"cannot seed {k} centroids from {n} rows")
        centroids = np.empty((k, d), dtype=np.float64)
        centroids[0] = block[self._rng.integers(n)]
        closest = squared_distances(block, centroids[:1]).ravel()
        trials = min(n, 2 + int(np.log(max(k, 2))))
        for j in range(1, k):
            total = float(closest.sum())
            if total <= 0.0 or not np.isfinite(total):
                centroids[j] = block[self._rng.integers(n)]
                continue
            candidates = np.unique(self._rng.choice(n, size=trials, p=closest / total))
            d2 = squared_distances(block, block[candidates])
            potentials = np.minimum(closest[:, None], d2).sum(axis=0)
            best = int(np.argmin(potentials))
            centroids[j] = block[candidates[best]]
            closest = np.minimum(closest, d2[:, best])
        return centroids

    def prepare(self, data: np.ndarray | None = None) -> "CentroidStat":
        if self._given is not None:
            centroids = np.array(self._given, dtype=np.float64, copy=True)
        elif data is not None:
            centroids = self._kmeanspp(np.asarray(data, dtype=np.float64))
        else:  # pragma: no cover - defensive
            raise ValueError("CentroidStat needs `centroids=` or a block to seed from")
        if self.d is not None and centroids.shape != (self.k, self.d):
            raise ValueError(f"centroids shape {centroids.shape} != {(self.k, self.d)}")
        self.d = int(centroids.shape[1])
        self.params["fallback"] = centroids.copy()
        self.values = {
            "sums": centroids * self.prior,
            "counts": np.full(self.k, self.prior, dtype=np.float64),
        }
        # Provenance of the measurement, recorded once: every partition of a
        # reduction must have been seeded with the same parameters.
        self.meta["_measure_sig"] = _digest({"centroids": centroids})
        self.act()
        self._dirty = False
        return self

    def centroids(self) -> np.ndarray:
        if "sums" not in self.values:
            return np.asarray(self.params["centroids"], dtype=np.float64)
        counts = np.asarray(self.values["counts"], dtype=np.float64)
        sums = np.asarray(self.values["sums"], dtype=np.float64)
        safe = np.maximum(counts, _EPS)
        centroids = sums / safe[:, None]
        empty = counts <= _EPS
        if np.any(empty):
            fallback = self.params.get("fallback", centroids)
            centroids[empty] = np.asarray(fallback, dtype=np.float64)[empty]
        return centroids

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if "sums" not in self.values or "centroids" not in self.params:
            self.prepare(block)
        # measurement is against the *acted* centroids, not the ones implied by
        # the values being accumulated: that separation is what lets a partition
        # freeze its parameters (large ``act_every``) and still merge exactly.
        centroids = np.asarray(self.params["centroids"], dtype=np.float64)
        dist = squared_distances(block, centroids)
        labels = np.argmin(dist, axis=1)
        sums = label_sums(block, labels, self.k)   # scatter -> per-column
        counts = np.bincount(labels, minlength=self.k).astype(np.float64)
        inertia = float(dist[np.arange(block.shape[0]), labels].sum())
        return {"sums": sums, "counts": counts, "inertia": np.array([inertia])}

    def act(self) -> None:
        self.params["centroids"] = self.centroids()

    def decode(self) -> dict[str, Any]:
        counts = np.asarray(self.values.get("counts", np.zeros(self.k)), dtype=np.float64)
        total = float(counts.sum())
        weights = counts / total if total > 0 else np.full(self.k, 1.0 / self.k)
        return {
            "centroids": self.centroids(),
            "counts": counts,
            "weights": weights,
        }

    def measurement_fingerprint(self) -> str | None:
        # The *seed* parameters, not the live decode: the decode drifts as
        # blocks are folded, and two identically seeded partitions remain a
        # valid additive reduction after they drift apart (that is what makes
        # ``partitioned_stat`` exact).  What must match is where they started.
        return self.meta.get("_measure_sig")

    def extra_config(self) -> dict[str, Any]:
        return {"k": self.k, "d": self.d, "init": self.init, "prior": self.prior}


# --------------------------------------------------------------------------- #
# Gramian stat: PCA
# --------------------------------------------------------------------------- #
class GramianStat(SufficientStat):
    """Second moment ``sum x x^T`` (and optionally ``sum x`` for centring).

    Decode is the symmetric eigen-decomposition the PCA module already uses, so
    a blocked pass over a memmap is the same answer as a full covariance --
    report 8.7's ``PCACovariance.from_data`` path, expressed as a stat so it can
    also be merged and fed from a worker.
    """

    name = "gramian_stat"

    def __init__(
        self,
        d: int | None = None,
        *,
        center: bool = True,
        n_components: int | None = None,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
        updater: Any = None,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every, seed=seed,
                         updater=updater)
        self.d = None if d is None else int(d)
        self.center = bool(center)
        self.n_components = n_components
        self.keys = ("gram", "total", "sum") if center else ("gram", "total")

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if self.d is None:
            self.d = int(block.shape[1])
        elif int(block.shape[1]) != self.d:
            # ``d`` is the dimension the stat was declared for; a block of a
            # different width would silently produce a covariance of the wrong
            # shape and a PCA with the wrong component count.
            raise ValueError(
                f"block width {block.shape[1]} != declared d={self.d}"
            )
        stats: dict[str, np.ndarray] = {
            "gram": block.T @ block,
            "total": np.array([weight], dtype=np.float64),
        }
        if self.center:
            stats["sum"] = block.sum(axis=0)
        return stats

    def covariance(self) -> np.ndarray:
        total = max(float(np.asarray(self.values["total"]).ravel()[0]), _EPS)
        cov = np.asarray(self.values["gram"], dtype=np.float64) / total
        if self.center:
            mean = np.asarray(self.values["sum"], dtype=np.float64) / total
            cov = cov - np.outer(mean, mean)
        return 0.5 * (cov + cov.T)

    def act(self) -> None:
        self.params["covariance"] = self.covariance()

    def decode(self) -> dict[str, Any]:
        cov = self.covariance()
        d = int(cov.shape[0])
        k = d if self.n_components is None else int(min(self.n_components, d))
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1][:k]
        total = max(float(np.clip(evals, 0.0, None).sum()), _EPS)
        total_weight = max(float(np.asarray(self.values["total"]).ravel()[0]), _EPS)
        mean = (
            np.asarray(self.values["sum"], dtype=np.float64) / total_weight
            if self.center
            else np.zeros(d, dtype=np.float64)
        )
        return {
            "components": evecs[:, order],
            "eigenvalues": np.maximum(evals[order], 0.0),
            "explained_variance_ratio": np.maximum(evals[order], 0.0) / total,
            "mean": mean,
            "covariance": cov,
        }

    def extra_config(self) -> dict[str, Any]:
        return {
            "d": self.d,
            "center": self.center,
            "n_components": self.n_components,
        }


# --------------------------------------------------------------------------- #
# adapter: any existing StatisticalOptimizer as a stat
# --------------------------------------------------------------------------- #
class OptimizerStat(SufficientStat):
    """Expose an existing optimizer as a :class:`SufficientStat`.

    This is the retrofit bridge for the algorithms whose loops are already
    tuned and measured (k-means/e-means, the mixture, the quantizer): the
    optimizer keeps owning its measurement and decode, and the stat owns
    accumulation, merge and reporting.  With ``mode="ema", alpha=1`` and one
    full-data block it is exactly one ``Replace`` update of the optimizer, which
    is what makes the stat path pinnable against the existing suite.

    Do not read that pin as epoch-style EM on several blocks.  With
    ``alpha=1`` and ``act_every=1``, each block *replaces* the accumulated
    state and takes its own M-step; that is online mini-batch EM.  A full-data
    EM pass is ``mode="sum"``, ``act_every=<blocks per pass>``,
    ``freeze_params=True``, followed by ``reset_accumulation()``; the state then
    holds exactly the whole pass's E-step statistics and acts once.
    """

    name = "optimizer_stat"

    def __init__(
        self,
        optimizer: Any,
        *,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        freeze_params: bool = False,
        updater: Any = None,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every,
                         seed=int(getattr(optimizer, "seed", 0)),
                         updater=updater)
        self.optimizer = optimizer
        self.keys = tuple(getattr(optimizer, "state_keys", ()))
        #: Epoch-style EM: measure every block under the parameters decoded at
        #: the last act instead of re-deriving them from the partial state.
        self.freeze_params = bool(freeze_params)
        #: The State built for the current block, reused by :meth:`act` so a
        #: block costs one copy of the arrays rather than two (~34 MB per block
        #: at full covariance, k=8, d=512).
        self._block_state: State | None = None

    def prepare(self, data: np.ndarray | None = None, config: Any = None) -> "OptimizerStat":
        if data is None:  # pragma: no cover - defensive
            raise ValueError("OptimizerStat needs the data to seed the optimizer")
        state = self.optimizer.initialize(np.asarray(data, dtype=np.float64), config)
        self.values = {k: np.array(v, copy=True) for k, v in state.values.items()}
        self.meta = copy.deepcopy(state.meta)
        self.act()
        arrays = {
            k: np.asarray(v, dtype=np.float64)
            for k, v in self.params.items()
            if isinstance(v, np.ndarray)
        }
        self.meta["_measure_sig"] = _digest(arrays) if arrays else None
        return self

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if self.freeze_params and self.params and hasattr(self.optimizer, "statistics_from"):
            statistics = self.optimizer.statistics_from(block, self.params)
        else:
            self._block_state = self.to_state()
            statistics = self.optimizer.measure(block, self._block_state)
        return {k: np.asarray(v, dtype=np.float64) for k, v in statistics.values.items()}

    def act(self) -> None:
        # reuse the block's State when there is one (see ``_block_state``)
        state = self._block_state if self._block_state is not None else self.to_state()
        self._block_state = None
        self.params = self.optimizer.decode(state)
        self._dirty = False

    def decode(self) -> dict[str, Any]:
        return self.optimizer.decode(self.to_state())

    def measurement_fingerprint(self) -> str | None:
        return self.meta.get("_measure_sig")

    def extra_config(self) -> dict[str, Any]:
        describe = getattr(self.optimizer, "describe", None)
        out = {"optimizer": getattr(self.optimizer, "name", type(self.optimizer).__name__)}
        if callable(describe):
            out["optimizer_config"] = describe()
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def merge_stats(stats: Iterable[SufficientStat]) -> SufficientStat:
    """Reduce an iterable of stats with :meth:`SufficientStat.merge`."""
    stats = list(stats)
    if not stats:
        raise ValueError("merge_stats needs at least one stat")
    merged = stats[0].copy()
    for other in stats[1:]:
        merged = merged.merge(other)
    return merged


__all__ = [
    "SufficientStat",
    "CentroidStat",
    "GramianStat",
    "OptimizerStat",
    "merge_stats",
]
