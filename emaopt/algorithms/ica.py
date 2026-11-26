"""Independent component analysis: FastICA reference and a stateful variant.

Both estimate a linear unmixing ``W`` such that ``g(W x)`` is maximally
non-Gaussian.  They differ in what is estimated from how much data:

* reference ``FastICA``: whiten on the full dataset, run the symmetric
  fixed-point iteration to convergence.
* ``StatefulICA``: persist the covariance (whitening) and the fixed-point
  statistics ``E[g(w_k^T x) x]`` / ``E[g'(w_k^T x)]`` with an EMA, and take one
  decorrelated fixed-point step per batch.

ICA is the PLAN's variance probe: if persistent statistics help here, the
mechanism is likely variance reduction of noisy higher-order moments rather
than anything specific to clustering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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
)
from ..state import EMA
from ..state.sufficient import SufficientStat
from .. import accel

_EPS = 1e-12


def _logcosh_g(x: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    # the tanh is the E-step's dominant cost on wide data (see emaopt.accel:
    # libmvec when available, numpy otherwise -- identical results either way)
    gx = accel.tanh(x) if alpha == 1.0 else np.tanh(alpha * x)
    return gx, alpha * (1.0 - gx**2)


def symmetric_decorrelation(w: np.ndarray) -> np.ndarray:
    """``W <- (W W^T)^{-1/2} W`` so rows are orthonormal."""
    s, u = np.linalg.eigh(w @ w.T)
    s = np.maximum(s, _EPS)
    return (u * (1.0 / np.sqrt(s))) @ u.T @ w


def _eigh_whitening(
    cov: np.ndarray, n_components: int | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Whitening plus the conditioning record, from one eigendecomposition.

    ``rcond`` is the ratio of the smallest *retained* eigenvalue to the largest
    (for a truncated ``k`` that is ``s[k-1]/s[0]``, the number that decides
    whether the kept subspace is identified at all); ``clipped`` counts retained
    eigenvalues floored at ``_EPS`` before the inverse square root.  Both are
    diagnostics -- the whitening itself is unchanged -- and they are what makes a
    jittered or rank-starved solve visible instead of silent.
    """
    s, u = np.linalg.eigh(0.5 * (cov + cov.T))
    order = np.argsort(s)[::-1]
    s = s[order]
    u = u[:, order]
    d = s.size
    k = d if n_components is None else min(int(n_components), d)
    clipped = int(np.count_nonzero(s[:k] <= _EPS))
    s = np.maximum(s, _EPS)
    top = float(s[0]) if s[0] > 0 else _EPS
    diag: dict[str, Any] = {
        "rcond": float(s[k - 1] / top),
        "rcond_full": float(s[-1] / top),
        "clipped": clipped,
        "max_eig": float(s[0]),
        "min_eig": float(s[k - 1]),
        "components": int(k),
        "dim": int(d),
    }
    if k >= d:
        return (u * (1.0 / np.sqrt(s))) @ u.T, diag
    return (u[:, :k] / np.sqrt(s[:k])).T, diag


def whitening_matrix(cov: np.ndarray, n_components: int | None = None) -> np.ndarray:
    """``W_z`` with ``W_z C W_z^T = I``; shape ``(d, d)``, or ``(k, d)`` truncated.

    With ``n_components = k < d`` the returned matrix is the whitening of the
    leading ``k`` eigen-directions, ``diag(1/sqrt(s_k)) V_k^T``.  Truncating by
    *slicing the rows* of the full symmetric ``W_z`` (the earlier behaviour) keeps
    the ``k`` **lowest**-variance directions, because ``eigh`` returns ascending
    eigenvalues, and then rotates them by the full basis -- not a whitening of
    anything.  Measured consequence: ICA features with unit-variance inputs had
    O(10) scale on a 128-d embedding set, so a mixture fitted on the concatenated
    features reported 531 nats/point where the data supports ~250.
    """
    return _eigh_whitening(cov, n_components)[0]


@dataclass
class ICAResult:
    unmixing: np.ndarray
    mixing: np.ndarray
    sources: np.ndarray
    whitening: np.ndarray
    n_iter: int
    converged: bool
    objective: float
    history: list[float] = field(default_factory=list)
    #: Resolution record for streaming/TD runs (examples, bytes, schedule).
    config: dict[str, Any] = field(default_factory=dict)
    examples_processed: int = 0
    nbytes: int = 0
    #: Per-feature mean removed before whitening.  ``None`` means the origin,
    #: which keeps older ``ICAResult`` constructions well-defined.
    mean: np.ndarray | None = None

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Unmix new rows in the same centred coordinates the fit used."""
        data = np.asarray(data, dtype=np.float64)
        mean = self.mean
        if mean is None:
            mean = np.zeros(data.shape[1], dtype=np.float64)
        else:
            mean = np.asarray(mean, dtype=np.float64).ravel()
        return (data - mean) @ self.unmixing.T


def _whiten(data: np.ndarray, n_components: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Whiten, keeping the ``n_components`` highest-variance directions."""
    data = np.asarray(data, dtype=np.float64)
    cov = (data.T @ data) / max(data.shape[0], 1)
    wz = whitening_matrix(cov, n_components)
    return data @ wz.T, wz


# local alias so FastICAStat.whitening_matrix() can call the module function of
# the same name without recursing into itself
_whitening_matrix = whitening_matrix


class FastICAStat(SufficientStat):
    """The FastICA fixed-point statistic as an accumulating object.

    State: the covariance used for whitening ``C`` (d, d), the fixed-point
    cross moment ``M = E[g(w^T z) z^T]`` (dim, dim) and the mean derivative
    ``E[g'(w^T z)]`` (dim,), where ``z`` is the whitened block and ``g`` is
    log-cosh.  The unmixing matrix ``W`` is a *derived* parameter: it is
    refreshed from ``M`` and ``dprime`` whenever an act is due, which is exactly
    the TD-FastICA recurrence (Zohar-Mannor-Monea 2017) when ``mode="ema"`` and
    the batch fixed point when ``mode="ema", alpha=1, act_every=1``.

    ``whitening="fixed"`` uses the covariance supplied by :meth:`prepare` and
    never re-estimates it -- the batch schedule, and what makes ``fastica()``
    reproduce its pre-stat numerics bit for bit.  ``whitening="running"`` folds
    each block's covariance with the same law as ``M``, which is the streaming
    form: one pass, ``O(d^2 + k d)`` state, no full whitened copy.
    """

    name = "fastica_stat"
    keys = ("C", "M", "dprime")

    def __init__(
        self,
        n_components: int | None = None,
        *,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
        whitening: str = "running",
        contrast: float = 1.0,
        whiten_every: int = 1,
        whiten_tol: float = 0.0,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every, seed=seed)
        if whitening not in ("running", "fixed"):
            raise ValueError(f"unknown whitening mode {whitening!r}")
        self.n_components = None if n_components is None else int(n_components)
        self.whitening = whitening
        self.contrast = float(contrast)
        #: Maximum blocks between whitening refreshes (T60).  ``1`` (the
        #: default) means "no cadence": with :attr:`whiten_tol` at 0 the solve
        #: runs every fold, which is the reference path.  ``4`` is the measured
        #: speed setting (<1e-4 of the unmixing at d=512).
        self.whiten_every = max(1, int(whiten_every))
        #: Relative Frobenius move of ``C`` that earns a new whitening solve.
        #: ``0.0`` (the default) is the reference path: one solve per fold.  A
        #: tolerance *replaces* the every-fold test rather than adding to it, so
        #: a value that is too large stops refreshing altogether.  T60 measured
        #: ``1e-2`` and ``1e-1`` freezing the whitening on the first block
        #: (|cos| 0.468 -- most of the unmixing lost), so neither is a speed
        #: setting.  ``1e-3`` was the only tolerance that saved a solve without a
        #: measurable cost (8 vs 9) and is still opt-in; the supported speed knob
        #: is the :attr:`whiten_every` cadence.
        self.whiten_tol = float(whiten_tol)
        self._wz_bucket: int | None = None
        #: Number of ``_whitening_matrix`` calls made -- deterministic, and the
        #: quantity T60 gates on when wall time is too noisy to trust.
        self.whitening_solves: int = 0
        #: Conditioning of the last solve: ``rcond``, ``clipped`` directions and
        #: the retained spectrum.  Empty until the first solve.
        self.whitening_diagnostics: dict[str, Any] = {}
        self.d: int | None = None
        self.dim: int | None = None
        self._rng = np.random.default_rng(seed)
        self._wz_cache: np.ndarray | None = None
        self._wz_source: np.ndarray | None = None
        self.acts: int = 0

    # -- setup -------------------------------------------------------------- #
    def prepare(
        self,
        data: np.ndarray | None = None,
        *,
        d: int | None = None,
        covariance: np.ndarray | None = None,
        mean: np.ndarray | None = None,
    ) -> "FastICAStat":
        """Seed ``W``, the centring mean, and (for fixed whitening) ``C``.

        ICA is defined on centred rows.  Passing ``data`` computes the mean and,
        when no covariance is supplied, the covariance of the centred rows.  A
        caller that supplies a covariance should pass its matching ``mean``;
        when it does not, the first block supplies one rather than silently
        whitening the offset.
        """
        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            d = int(data.shape[1])
            if mean is None:
                mean = data.mean(axis=0)
            else:
                mean = np.asarray(mean, dtype=np.float64).reshape(d)
            if covariance is None:
                centred = data - mean
                covariance = (centred.T @ centred) / max(data.shape[0], 1)
        elif covariance is not None:
            covariance = np.asarray(covariance, dtype=np.float64)
            d = int(covariance.shape[0])
        elif mean is not None:
            mean = np.asarray(mean, dtype=np.float64).ravel()
            d = int(mean.size)
        if d is None:  # pragma: no cover - defensive
            raise ValueError("FastICAStat.prepare needs data, d=, mean= or covariance=")
        self.d = int(d)
        if mean is not None:
            # A caller may pass a column vector (for example from a solver);
            # keep the stored/decoded mean one-dimensional so transform()
            # broadcasts against rows rather than columns.
            mean = np.asarray(mean, dtype=np.float64).ravel()
        if mean is not None and np.asarray(mean).size != self.d:
            raise ValueError(
                f"mean has {np.asarray(mean).size} entries for d={self.d}"
            )
        self.params["mean"] = (None if mean is None
                               else np.array(mean, dtype=np.float64, copy=True))
        self.dim = int(self.n_components is None and d or min(int(self.n_components or d), int(d)))
        self.params["W"] = symmetric_decorrelation(self._rng.normal(size=(self.dim, self.dim)))
        if covariance is not None:
            self.params["C"] = np.array(covariance, copy=True)
            if self.whitening == "fixed":
                self.values["C"] = np.array(covariance, copy=True)
        self._invalidate_whitening()
        return self

    def _invalidate_whitening(self) -> None:
        self._wz_cache = None
        self._wz_source = None
        self._wz_bucket = None

    def covariance(self) -> np.ndarray:
        if self.whitening == "running" and "C" in self.values:
            return np.asarray(self.values["C"], dtype=np.float64)
        if "C" in self.params:
            return np.asarray(self.params["C"], dtype=np.float64)
        if "C" in self.values:  # pragma: no cover - defensive
            return np.asarray(self.values["C"], dtype=np.float64)
        raise ValueError("FastICAStat has no covariance yet; call prepare()")

    def _cov_moved(self, cov: np.ndarray) -> float:
        """Relative Frobenius move of ``C`` since the whitening was built."""
        if self._wz_source is None:
            return float("inf")
        scale = max(float(np.linalg.norm(self._wz_source)), 1e-300)
        return float(np.linalg.norm(np.asarray(cov, dtype=np.float64)
                                    - self._wz_source) / scale)

    def whitening_matrix(self) -> np.ndarray:
        cov = self.covariance()
        if self._wz_cache is not None and self.whitening == "fixed":
            # fixed means fixed: the supplied covariance never moves
            return self._wz_cache
        if self._wz_cache is None:
            stale = True
        else:
            # **Content, not identity**, decides (T60).  The old check was
            # ``self._wz_source is not cov``; ``_fold`` assigns a *new* array on
            # every block, so that test was always true and a d x d solve ran
            # on every fold even when ``C`` had not moved.
            #
            # Two knobs, two meanings: ``whiten_every`` is a *maximum* staleness
            # and ``whiten_tol`` is an early refresh when the covariance moved
            # more than that.  The three pairs that matter:
            #   (every=1, tol=0)  -> every fold: the reference path, bit for bit
            #   (every=N, tol=0)  -> the cadence T60 measured
            #   (every=1, tol>0)  -> movement-only
            moved = (self.whiten_tol > 0.0
                     and self._cov_moved(cov) > self.whiten_tol)
            if self.whiten_every > 1:
                stale = moved or (self.steps // self.whiten_every
                                 != self._wz_bucket)
            else:
                # A tolerance *replaces* the identity test, so it can actually
                # suppress solves; with tol=0 the identity test is the reference
                # path (a new ``C`` per fold, and a cache hit for a second call
                # inside one block).
                stale = (moved if self.whiten_tol > 0.0
                         else self._wz_source is not cov)
        if stale:
            self._wz_cache, diagnostics = _eigh_whitening(cov, self.n_components)
            # hold the covariance we solved on; folds replace arrays rather than
            # mutating them, so a reference is a stable identity for the cache
            self._wz_source = cov
            self._wz_bucket = (self.steps // self.whiten_every
                               if self.whiten_every > 1 else None)
            self.whitening_solves += 1
            self.whitening_diagnostics = {**diagnostics,
                                          "solves": self.whitening_solves}
        return self._wz_cache

    @staticmethod
    def _initial_mean(block: np.ndarray) -> np.ndarray:
        """A first-block mean worth centring by, or zero.

        A one-pass running fit has no global mean.  Subtracting a block mean that
        is itself within the block's standard error adds noise rather than
        removing an offset (on a zero-mean fixture this shifted the TD source
        correlation from 0.90 to 0.84).  A mean larger than four standard errors
        is resolvable, so it is used; otherwise the historical origin-centred
        coordinate is retained.  An explicitly supplied mean skips this test.
        """
        block = np.asarray(block, dtype=np.float64)
        n = max(int(block.shape[0]), 1)
        mean = block.mean(axis=0)
        scale = float(np.linalg.norm(block.std(axis=0)))
        if np.linalg.norm(mean) <= 4.0 * scale / np.sqrt(n):
            return np.zeros(block.shape[1], dtype=np.float64)
        return mean

    def whiten(self, block: np.ndarray) -> np.ndarray:
        """Centre a block and project it into the whitened coordinates.

        ``(x - mu) W^T`` is evaluated as ``x W^T - mu W^T``.  The two forms are
        algebraically identical, but the projected one keeps the result at
        ``(n, dim)`` instead of materialising an ``(n, d)`` centred copy before
        the GEMM.  The saving grows with the input width: at 8192 x 512 with
        dim=64 the copy measured 15.5 ms against 1.7 ms for the projected form
        (a 33.6 MB temporary versus a 4.2 MB result).  At 20000 x 64, where the
        result is the same width as the input, the two times are comparable;
        the projected form still avoids the extra full-width allocation.
        """
        block = np.asarray(block, dtype=np.float64)
        wz = self.whitening_matrix()
        out = block @ wz.T
        mean = self.params.get("mean")
        if mean is not None:
            out -= np.asarray(mean, dtype=np.float64) @ wz.T
        return out

    # -- statistics --------------------------------------------------------- #
    def block_statistic_whitened(
        self, xw: np.ndarray, n: int | None = None
    ) -> dict[str, np.ndarray]:
        """Fixed-point moment of an already-whitened block (batch fast path)."""
        xw = np.asarray(xw, dtype=np.float64)
        rows = int(xw.shape[0] if n is None else n)
        w = np.asarray(self.params["W"], dtype=np.float64)
        gx, gprime = _logcosh_g(xw @ w.T, self.contrast)
        denom = max(rows, 1)
        return {
            "M": (gx.T @ xw) / denom,
            "dprime": gprime.mean(axis=0),
        }

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        block = np.asarray(block, dtype=np.float64)
        if "W" not in self.params:
            self.prepare(block)
        if self.params.get("mean") is None:
            # A fixed-whitening caller that supplied only the covariance still
            # gets a centred measurement when the first block has a resolvable
            # offset; a noisy zero-mean block keeps the origin.
            self.params["mean"] = self._initial_mean(block)
        n = max(int(block.shape[0]), 1)
        mean = self.params["mean"]
        centred = block if not np.any(mean) else block - mean
        if self.whitening == "running":
            cov = (centred.T @ centred) / n
            if "C" not in self.values and "C" not in self.params:
                # first block: its own covariance is the whitening it is
                # measured under, and the running estimate starts there
                self.params["C"] = cov
                self._invalidate_whitening()
            statistics = self.block_statistic_whitened(self.whiten(block), n)
            statistics["C"] = cov
            return statistics
        return self.block_statistic_whitened(self.whiten(block), n)

    def update_whitened(
        self,
        xw: np.ndarray,
        *,
        covariance: np.ndarray | None = None,
        weight: float | None = None,
    ) -> "FastICAStat":
        """Fold a whitened block (avoids re-whitening a fixed batch)."""
        xw = np.asarray(xw, dtype=np.float64)
        n = max(int(xw.shape[0]), 1)
        statistics = self.block_statistic_whitened(xw, n)
        if covariance is not None and self.whitening == "running":
            statistics["C"] = np.asarray(covariance, dtype=np.float64)
        self._fold(statistics, float(n if weight is None else weight))
        if self.steps % self.act_every == 0:
            self.act()
        return self

    # -- act / decode ------------------------------------------------------- #
    def act(self) -> None:
        if "M" not in self.values or "W" not in self.params:
            return
        w = np.asarray(self.params["W"], dtype=np.float64)
        m = np.asarray(self.values["M"], dtype=np.float64)
        dprime = np.asarray(self.values["dprime"], dtype=np.float64)
        if m.shape != w.shape:  # pragma: no cover - defensive
            raise ValueError(f"statistic shape {m.shape} != W shape {w.shape}")
        w_new = symmetric_decorrelation(m - dprime[:, None] * w)
        self._last_delta = float(
            np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0))
        )
        self.params["W"] = w_new
        self.acts += 1
        self._dirty = False

    def decode(self) -> dict[str, Any]:
        self._maybe_act()
        w = np.asarray(self.params["W"], dtype=np.float64)
        wz = self.whitening_matrix()
        unmixing = w @ wz
        mean = self.params.get("mean")
        if mean is None:
            mean = np.zeros(int(self.d or 0), dtype=np.float64)
        return {
            "unmixing": unmixing,
            "mixing": np.linalg.pinv(unmixing),
            "components": w.copy(),
            "whitening": np.array(wz, copy=True),
            "mean": np.array(mean, copy=True),
            "covariance": np.array(self.covariance(), copy=True),
        }

    def negentropy(self, data: np.ndarray, *, chunk: int | None = 65536) -> float:
        """Mean ``log cosh`` of the unmixed block (what the batch loop records)."""
        data = np.asarray(data, dtype=np.float64)
        w = np.asarray(self.params["W"], dtype=np.float64)
        n = int(data.shape[0])
        if chunk is None or chunk >= n:
            return float(np.mean(accel.logcosh(self.whiten(data) @ w.T)))
        total = 0.0
        for start in range(0, n, chunk):
            xw = self.whiten(data[start : start + chunk])
            total += float(accel.logcosh(xw @ w.T).sum())
        return total / max(n, 1)

    def merge(self, other: "FastICAStat") -> "FastICAStat":
        """Mass-weighted merge of the decayed moments.

        Unlike the additive stats (whose values are sums), ICA's values are
        per-block *means* -- ``M = E[g(w^T z) z^T]`` and ``E[g'(w^T z)]`` -- so
        the reduction is a weighted mean by each accumulator's decayed mass, not
        a sum.  This is the decay correction the plan asks for: two partitions
        of different length contribute in proportion to what they still
        remember.  Covariances merge the same way, which keeps the combination
        positive semi-definite.
        """
        self._check_compatible(other)
        merged = self.copy()
        w1 = max(float(self.weight), _EPS)
        w2 = max(float(other.weight), _EPS)
        total = w1 + w2
        weighted = self.mode == "ema"
        for key in set(self.values) | set(other.values):
            v1 = self.values.get(key)
            v2 = other.values.get(key)
            if v1 is None:
                merged.values[key] = np.array(v2, dtype=np.float64, copy=True)
            elif v2 is None:
                merged.values[key] = np.array(v1, dtype=np.float64, copy=True)
            else:
                a = np.asarray(v1, dtype=np.float64)
                b = np.asarray(v2, dtype=np.float64)
                if weighted or key == "C":
                    # moments are means: combine by decayed mass.  The covariance
                    # is a mean in both modes and must not be summed.
                    merged.values[key] = (a * w1 + b * w2) / total
                else:
                    merged.values[key] = a + b
        merged.weight = total
        merged.n_examples = self.n_examples + other.n_examples
        merged.steps = self.steps + other.steps
        c1 = self.params.get("C")
        c2 = other.params.get("C")
        if merged.whitening == "fixed" and c1 is not None and c2 is not None:
            merged.params["C"] = (
                np.asarray(c1, dtype=np.float64) * w1
                + np.asarray(c2, dtype=np.float64) * w2
            ) / total
            merged.values["C"] = np.array(merged.params["C"], copy=True)
        means = [p.get("mean") for p in (self.params, other.params)]
        if any(mean is not None for mean in means):
            m1, m2 = (np.zeros(int(self.d or 0)) if mean is None else mean
                      for mean in means)
            merged.params["mean"] = (np.asarray(m1) * w1
                                     + np.asarray(m2) * w2) / total
        merged._invalidate_whitening()
        merged._dirty = True
        return merged

    def extra_config(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "dim": self.dim,
            "whitening": self.whitening,
            "centered": self.params.get("mean") is not None,
            "contrast": self.contrast,
            "acts": int(self.acts),
            "last_delta": None if not np.isfinite(self._last_delta) else float(self._last_delta),
            "whitening_solves": int(self.whitening_solves),
            "whitening_diagnostics": dict(self.whitening_diagnostics),
        }

def fastica(
    data: np.ndarray,
    n_components: int | None = None,
    *,
    niter: int = 500,
    tol: float = 1e-8,
    alpha: float = 1.0,
    seed: int = 0,
    record_history: bool = True,
) -> ICAResult:
    """Symmetric FastICA on whitened data (the reference implementation).

    ``record_history=False`` skips the per-iteration negentropy.  That number is
    diagnostic only -- the fixed-point step never reads it, and convergence is
    decided by ``delta`` -- but evaluating ``mean(log cosh(xw @ W^T))`` costs an
    extra ``n x k`` product plus a ``log`` and a ``cosh`` over ``n k`` values.
    Measured at 1 000 448 x 512 with k=64: 0.95 s of a 2.0 s iteration, i.e. it
    nearly halves the run.  The returned ``objective`` is still exact, evaluated
    once at the end.

    The loop now runs on :class:`FastICAStat` with the batch schedule
    (``alpha=1``, ``act_every=1``, fixed whitening).  ``tests/test_streaming_ica.py``
    pins the result bit-for-bit against the pre-stat loop, so the abstraction is
    free where it changes nothing (ONLINE_PLAN gate i).
    """
    data = np.asarray(data, dtype=np.float64)
    stat = FastICAStat(
        n_components, mode="ema", alpha=1.0, act_every=1, seed=seed,
        whitening="fixed", contrast=alpha,
    )
    stat.prepare(data)
    xw = stat.whiten(data)
    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, int(niter) + 1):
        stat.update_whitened(xw)
        w = np.asarray(stat.params["W"], dtype=np.float64)
        if record_history:
            # accel.logcosh is libmvec's two vector transcendentals in one
            # pass: 0.043 ms against NumPy's 0.167 ms for this 4000 x 4
            # history (measured), agreeing to 8.9e-16.  It is a diagnostic
            # -- the fixed-point step never reads it -- so this changes no
            # unmixing, only the recorded curve's last bit.
            history.append(float(np.mean(accel.logcosh(xw @ w.T))))
        if stat._last_delta < tol:
            converged = True
            break
    params = stat.decode()
    unmixing = params["unmixing"]
    mixing = params["mixing"]
    mean = params["mean"]
    sources = (data - mean) @ unmixing.T
    return ICAResult(
        unmixing=unmixing,
        mixing=mixing,
        sources=sources,
        whitening=params["whitening"],
        mean=mean,
        n_iter=it,
        converged=converged,
        objective=-float(np.mean(np.log(np.cosh(xw @ w.T)))),
        history=history,
        config={"mode": "batch", "stat": stat.name, "act_frequency": 1},
        examples_processed=int(data.shape[0]),
        nbytes=int(stat.nbytes),
    )


def _streaming_moments(source: Any, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Centred covariance and mean of a stream, in one fixed-order pass.

    ``whitening="auto"`` uses this.  The covariance is a *decode*, not a per-row
    accumulate, so paying one extra pass for it is cheaper than re-solving a
    ``d x d`` eigenproblem on every fold (T62): at d=512 the solve is ~16 ms and
    the block Gramian ~6 ms, so one prepass buys back more than it costs.

    Each block is centred on its own mean and combined with the Chan/Welford
    between-block term.  Accumulating raw ``X^T X`` and subtracting ``mean
    mean^T`` at the end loses the variance of offset data (the same cancellation
    the PCA decode avoids by centring its blocks).
    """
    from ..stream import iter_blocks

    d = int(source.shape[1])
    mean = np.zeros(d, dtype=np.float64)
    scatter = np.zeros((d, d), dtype=np.float64)
    rows = 0
    for block in iter_blocks(source, max(1, int(batch_size)), nepochs=1,
                             shuffle=False, seed=0):
        b = np.asarray(block, dtype=np.float64)
        nb = int(b.shape[0])
        if nb == 0:
            continue
        bmean = b.mean(axis=0)
        centred = b - bmean
        scatter += centred.T @ centred
        if rows:
            delta = bmean - mean
            scatter += (rows * nb / (rows + nb)) * np.outer(delta, delta)
        mean = (rows * mean + nb * bmean) / (rows + nb)
        rows += nb
    if rows == 0:
        raise ValueError("cannot estimate a covariance from an empty stream")
    return scatter / rows, mean


def fastica_td(
    data: Any,
    n_components: int | None = None,
    *,
    batch_size: int = 4096,
    nepochs: int = 1,
    decay: float = 0.05,
    act_every: int = 1,
    tol: float = 1e-8,
    seed: int = 0,
    eval_data: np.ndarray | None = None,
    max_bytes: int | None = None,
    budget: Any = None,
    max_updates: int | None = None,
    record_history: bool = True,
    contrast: float = 1.0,
    shuffle: bool = True,
    whiten_every: int = 1,
    whiten_tol: float = 0.0,
    whitening: str = "auto",
    covariance: np.ndarray | None = None,
    mean: np.ndarray | None = None,
) -> ICAResult:
    """TD-FastICA: one streaming pass over exponential fixed-point moments.

    ``data`` may be an array or a path to a memory-mapped ``.npy``.  The state is
    the running covariance, ``M`` and ``dprime`` (``O(d^2 + k d)``); no whitened
    copy of the dataset is materialised, which is what lets the run fit under a
    cap (``max_bytes``) instead of under whatever the machine happens to have.
    ``eval_data`` (default: a 20k-row subsample) scores the final unmixing; the
    returned ``sources`` are that evaluation set's sources, not the full data's.

    ``act_every`` decouples the decorrelation step from accumulation exactly as
    the plan's protocol expresses it.  ``decay`` is the EMA rate on the
    statistic; with ``decay=1.0`` the recurrence degenerates to the batch
    moment of the most recent block.

    ``whitening="auto"`` (the default) reads the stream once to form the
    covariance and then runs with that fixed whitening: **one** ``d x d`` solve
    for the whole run instead of one per block.  It is a two-pass schedule;
    ``whitening="running"`` remains the single-pass form for memmapped or
    ``max_bytes``-capped streams, at one solve per fold.
    """
    from ..budget import MemoryBudget, resolve_compute
    from ..stream import as_block_source, block_steps, iter_blocks, n_rows

    source = as_block_source(data)
    n = n_rows(source)
    d = int(source.shape[1])
    requested_whitening = whitening
    covariance_source = "supplied" if covariance is not None else None
    if whitening == "auto":
        if covariance is None:
            covariance, mean = _streaming_moments(
                source, max(1, min(int(batch_size), max(n, 1)))
            )
            covariance_source = "full_pass"
        whitening = "fixed"
    stat = FastICAStat(
        n_components, mode="ema", alpha=float(decay), act_every=int(act_every),
        seed=seed, whitening=whitening, contrast=contrast,
        whiten_every=whiten_every, whiten_tol=whiten_tol,
    )
    if whitening == "fixed":
        if covariance is None:
            raise ValueError("whitening='fixed' needs covariance= (the whitening "
                             "covariance, e.g. the PCA stage's)")
        stat.prepare(d=d, covariance=np.asarray(covariance, dtype=np.float64),
                     mean=mean)
    else:
        stat.prepare(d=d, mean=mean)
    bs = max(1, min(int(batch_size), n))
    plan = None
    memory = budget
    if memory is None and max_bytes is not None:
        memory = MemoryBudget(int(max_bytes), "cap")
    if memory is not None:
        plan = resolve_compute(
            n, d, int(n_components or 0), stat_bytes=stat.nbytes, budget=memory,
            nepochs=nepochs, requested_batch=bs,
        )
        if plan.mode == "stream":
            bs = max(1, min(bs, plan.block_size))
    steps = block_steps(n, bs, nepochs=nepochs)
    if max_updates is not None:
        steps = min(steps, int(max_updates))
    stat.configure(total_steps=steps, dataset_size=n)
    eval_x = _eval_subsample(source, eval_data=eval_data, seed=seed)
    history: list[float] = []
    examples = 0
    updates = 0
    wall0 = time.perf_counter()
    every = max(1, steps // 20)
    for block in iter_blocks(source, bs, nepochs=nepochs, shuffle=shuffle, seed=seed,
                             max_updates=max_updates):
        stat.update(block)
        updates += 1
        examples += int(block.shape[0])
        if stat.steps % stat.act_every == 0 and updates % every == 0 and record_history:
            history.append(-stat.negentropy(eval_x))
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
        "wall_time": float(time.perf_counter() - wall0),
        "compute_plan": None if plan is None else plan.to_dict(),
        "mode": "td",
        "whitening": requested_whitening,
        "whitening_effective": whitening,
        "covariance_source": covariance_source,
        "whiten_tol": float(whiten_tol),
        "whitening_solves": int(stat.whitening_solves),
    })
    return ICAResult(
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


def _eval_subsample(
    source: Any, *, eval_data: np.ndarray | None = None, seed: int = 0, max_rows: int = 20000
) -> np.ndarray:
    if eval_data is not None:
        return np.asarray(eval_data, dtype=np.float64)
    n = int(source.shape[0])
    if n <= max_rows:
        return np.asarray(source, dtype=np.float64)
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
    return np.asarray(source[idx], dtype=np.float64)


class StatefulICA(StatisticalOptimizer):
    """FastICA fixed point with persistent covariance and gradient statistics."""

    name = "stateful_ica"
    state_keys = ("C", "M", "dprime")

    def __init__(
        self,
        n_components: int | None = None,
        *,
        updater: Any = None,
        alpha: float = 1.0,
        seed: int = 0,
        sample_size: int = 4096,
    ) -> None:
        super().__init__(updater or EMA(alpha=0.1))
        self.n_components = n_components
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.sample_size = int(sample_size)

    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        rng = np.random.default_rng(self.seed)
        sample = data
        if n > self.sample_size:
            sample = data[rng.choice(n, size=self.sample_size, replace=False)]
        mean = data.mean(axis=0)
        centred = sample - mean
        cov = (centred.T @ centred) / max(sample.shape[0], 1)
        dim = d if self.n_components is None else int(self.n_components)
        # W maps d features to dim components, so it is (dim, d).  The earlier
        # (dim, dim) shape only worked when n_components == d; for a truncated fit
        # it made `xw @ W.T` a shape error on the first batch.
        w = symmetric_decorrelation(rng.normal(size=(dim, d)))
        return State(
            {
                "C": cov,
                "M": np.zeros((dim, d)),
                "dprime": np.zeros(dim),
                "W": w,
            },
            0,
            {"n_features": d, "mean": np.array(mean, copy=True)},
        )

    def _whiten(self, data: np.ndarray, state: State) -> np.ndarray:
        """Full whitened batch: ``W_z C W_z^T = I`` up to the EMA covariance."""
        cov = np.asarray(state["C"], dtype=np.float64)
        wz = whitening_matrix(cov)
        mean = np.asarray(state.meta.get("mean", np.zeros(cov.shape[0])),
                          dtype=np.float64)
        return (data - mean) @ wz.T

    def decode(self, state: State) -> dict[str, Any]:
        wz = whitening_matrix(np.asarray(state["C"], dtype=np.float64))
        w = np.asarray(state["W"], dtype=np.float64)
        # measure() whitens fully and maps with W, so the unmixing is W @ W_z with
        # the *full* W_z; slicing its rows by the component count broke that
        # identity for n_components < d.
        unmixing = w @ wz
        mean = np.asarray(state.meta.get("mean", np.zeros(wz.shape[1])),
                          dtype=np.float64)
        return {
            "unmixing": unmixing,
            "mixing": np.linalg.pinv(unmixing),
            "components": w.copy(),
            "whitening": wz,
            "mean": np.array(mean, copy=True),
        }

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        xw = self._whiten(batch, state)
        w = np.asarray(state["W"], dtype=np.float64)
        gx, gprime = _logcosh_g(xw @ w.T, self.alpha)
        mean = np.asarray(state.meta.get("mean", np.zeros(batch.shape[1])),
                          dtype=np.float64)
        centred = batch - mean
        # M is (dim, d): the fixed-point statistic E[g(w^T x) x] for every component
        stats = {
            "C": (centred.T @ centred) / max(batch.shape[0], 1),
            "M": (gx.T @ xw) / max(batch.shape[0], 1),
            "dprime": gprime.mean(axis=0),
        }
        negentropy = float(np.mean(accel.logcosh(xw @ w.T)))
        return State(stats, 0, {"negentropy": negentropy})

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        new = super().update_state(state, statistics, ctx)
        w = np.asarray(state["W"], dtype=np.float64)
        m = np.asarray(new["M"], dtype=np.float64)
        dprime = np.asarray(new["dprime"], dtype=np.float64)
        w_new = symmetric_decorrelation(m - dprime[:, None] * w)
        new.values["W"] = w_new
        return new

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Negative mean log-cosh of the unmixed data (lower is more non-Gaussian here)."""
        data = np.asarray(data, dtype=np.float64)
        mean = np.asarray(parameters.get("mean", np.zeros(data.shape[1])),
                          dtype=np.float64)
        sources = (data - mean) @ parameters["unmixing"].T
        return -float(np.mean(np.log(np.cosh(sources))))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = 1024,
        nepochs: int = 20,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory)
        return fit(self, data, cfg, eval_data=eval_data)


class FastICA:
    """Thin class wrapper so the reference shares the ``.fit`` surface.

    ``streaming=True`` runs the TD form (:func:`fastica_td`) instead of the
    batch fixed point; everything else about the surface is unchanged.
    """

    name = "fastica"

    def __init__(self, n_components: int | None = None, *, niter: int = 500,
                 tol: float = 1e-8, alpha: float = 1.0, seed: int = 0,
                 streaming: bool = False, batch_size: int = 4096,
                 decay: float = 0.05, act_every: int = 1,
                 max_bytes: int | None = None,
                 whiten_every: int = 1, whiten_tol: float = 0.0,
                 whitening: str = "auto",
                 covariance: np.ndarray | None = None,
                 mean: np.ndarray | None = None) -> None:
        self.n_components = n_components
        self.niter = int(niter)
        self.tol = float(tol)
        self.alpha = float(alpha)
        self.seed = int(seed)
        self.streaming = bool(streaming)
        self.batch_size = int(batch_size)
        self.decay = float(decay)
        self.act_every = int(act_every)
        self.max_bytes = max_bytes
        self.whiten_every = max(1, int(whiten_every))
        self.whiten_tol = float(whiten_tol)
        self.whitening = whitening
        self.covariance = covariance
        self.mean = mean

    def fit(self, data: np.ndarray, *, eval_data: np.ndarray | None = None) -> ICAResult:
        if self.streaming:
            return fastica_td(
                data, self.n_components, batch_size=self.batch_size, decay=self.decay,
                act_every=self.act_every, tol=self.tol, seed=self.seed,
                eval_data=eval_data, max_bytes=self.max_bytes, contrast=self.alpha,
                whiten_every=self.whiten_every, whiten_tol=self.whiten_tol,
                whitening=self.whitening, covariance=self.covariance,
                mean=self.mean,
            )
        return fastica(data, self.n_components, niter=self.niter, tol=self.tol,
                       alpha=self.alpha, seed=self.seed)

    def describe(self) -> dict[str, Any]:
        return {"optimizer": self.name, "n_components": self.n_components,
                "streaming": self.streaming,
                "batch_size": self.batch_size if self.streaming else None,
                "decay": self.decay if self.streaming else None}
