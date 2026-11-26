"""Robust PCA: low-rank + sparse decomposition.

Reference: principal component pursuit by inexact augmented Lagrange multipliers
(Candes et al.), solved with numpy SVDs.  Two robustness regimes are covered:

* ``robust_pca`` (batch): the convex PCP program, optionally debiased on the
  recovered support.  Nuclear-norm shrinkage biases ``L`` inward by ``1/mu`` per
  singular value; refitting the rank-``r`` SVD on the entries PCP calls inliers
  removes that bias (rank-1 probe: relative error 0.059 -> 0.0001).
* ``StreamingRobustPCA``: the PLAN's Tier-3 case.  The persistent state is an
  EMA of *element-wise trimmed* second moments; the basis is re-derived from the
  operator, never averaged.  Only the projection is tracked.

The streaming threshold is estimated once from the data scale (MAD, robust to
<50% contamination) and held fixed.  Deriving it from residuals against the
current basis looks natural and is wrong: residuals shrink to zero as the basis
converges, the threshold follows them down, and every entry gets trimmed.  That
bug showed up as alignment decaying 0.998 -> 0.647 over ten epochs from a good
initialization.  A fixed data-scale threshold holds 0.99 alignment from 10% to
40% gross corruption.
"""

from __future__ import annotations

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
from .dictionary import soft_threshold
from .subspace import top_basis

_EPS = 1e-12


def singular_value_threshold(m: np.ndarray, tau: float) -> tuple[np.ndarray, int]:
    """Singular-value soft threshold (the proximal map of the nuclear norm)."""
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    s = np.maximum(s - float(tau), 0.0)
    rank = int(np.sum(s > 0.0))
    return (u * s) @ vt, rank


def otsu_threshold(magnitudes: np.ndarray, bins: int = 256) -> float:
    """Otsu's between-class-variance cut (parameter-free two-mode split)."""
    values = np.asarray(magnitudes, dtype=np.float64).ravel()
    if values.size == 0:
        return 0.0
    lo = float(values.min())
    hi = float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return hi
    hist, edges = np.histogram(values, bins=int(bins), range=(lo, hi))
    hist = hist.astype(np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weight_bg = np.cumsum(hist)
    weight_fg = hist.sum() - weight_bg
    cum_mean = np.cumsum(hist * centers)
    mean_bg = cum_mean / np.maximum(weight_bg, _EPS)
    mean_fg = (cum_mean[-1] - cum_mean) / np.maximum(weight_fg, _EPS)
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between[~np.isfinite(between)] = -1.0
    return float(centers[int(np.argmax(between))])


def debias_low_rank(
    data: np.ndarray,
    sparse: np.ndarray,
    rank: int,
    *,
    threshold: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> np.ndarray:
    """Refit rank-``rank`` ``L`` on the entries PCP calls inliers.

    Hard-imputation alternating projection: observed (inlier) entries are held
    at their values, corrupted ones at the current low-rank prediction.

    The inlier set comes from Otsu's threshold on ``|S|``.  PCP's sparse output is
    bimodal -- exact zeros where the low-rank model already fits, gross values at
    the corruptions -- and the two modes are separated by orders of magnitude, so
    the split needs no magnitude constant.  ``|S| > lam`` does *not* work: the
    soft-threshold dust left behind by a converged run reaches ~5x ``lam``, so it
    labels good entries as corruptions and the refit collapses (measured relative
    error 0.06 -> 0.32 on a rank-1 problem).
    """
    x = np.asarray(data, dtype=np.float64)
    magnitudes = np.abs(np.asarray(sparse, dtype=np.float64))
    tau = otsu_threshold(magnitudes) if threshold is None else float(threshold)
    inlier = magnitudes <= tau
    if inlier.mean() < 0.5 or inlier.mean() > 0.999:
        # no separable sparse support: the PCP split is already the answer
        return x - np.asarray(sparse, dtype=np.float64)
    low = np.where(inlier, x, 0.0)
    scale = max(float(np.linalg.norm(low, "fro")), _EPS)
    for _ in range(max(1, int(max_iter))):
        u, s, vt = np.linalg.svd(low, full_matrices=False)
        r = int(min(max(rank, 1), s.size))
        prediction = (u[:, :r] * s[:r]) @ vt[:r]
        updated = np.where(inlier, x, prediction)
        if float(np.linalg.norm(updated - low, "fro")) <= tol * scale:
            low = updated
            break
        low = updated
    u, s, vt = np.linalg.svd(low, full_matrices=False)
    r = int(min(max(rank, 1), s.size))
    return (u[:, :r] * s[:r]) @ vt[:r]


def robust_pca(
    data: np.ndarray,
    *,
    lam: float | None = None,
    mu: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-7,
    rho: float = 1.5,
    refine_rank: int | None = None,
) -> dict[str, Any]:
    """Principal component pursuit: ``min ||L||_* + lam ||S||_1`` s.t. ``X = L + S``.

    ``lam`` defaults to ``1/sqrt(max(n, d))``, the value the recovery guarantee
    uses.  ``refine_rank`` enables the debiasing fit above and sets its rank;
    leaving it ``None`` returns the raw PCP solution.
    """
    x = np.asarray(data, dtype=np.float64)
    n, d = x.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(n, d))
    if mu is None:
        mu = 1.25 / max(float(np.linalg.norm(x, 2)), _EPS)
    l = np.zeros_like(x)
    s = np.zeros_like(x)
    s_previous = s.copy()
    y = np.zeros_like(x)
    norm_x = max(float(np.linalg.norm(x, "fro")), _EPS)
    converged = False
    it = 0
    rank = 0
    for it in range(1, int(max_iter) + 1):
        s = soft_threshold(x - l + y / mu, lam / mu)
        l, rank = singular_value_threshold(x - s + y / mu, 1.0 / mu)
        residual = x - l - s
        y += mu * residual
        primal = float(np.linalg.norm(residual, "fro"))
        dual = mu * float(np.linalg.norm(s - s_previous, "fro"))
        # Residual balancing (Boyd et al., ADMM 3.4.1): the penalty grows only
        # while the primal residual dominates, and shrinks when the dual one does.
        # Geometric growth alone pins mu at its cap after ~35 iterations and the
        # pair then crawls -- objective 921.0426 -> 921.0314 over the following
        # 3000 iterations, against 919.7409 for a balanced schedule on the same
        # problem.  The reported solution was therefore not the PCP optimum.
        if primal > 10.0 * dual:
            mu *= rho
        elif dual > 10.0 * primal:
            mu /= rho
        mu = min(mu, 1e10 * max(1.0 / norm_x, _EPS))
        s_previous = s
        if primal / norm_x < tol and dual / (mu * norm_x) < tol:
            converged = True
            break
    out: dict[str, Any] = {
        "low_rank": l,
        "sparse": s,
        "rank": rank,
        "lam": float(lam),
        "n_iter": it,
        "converged": converged,
        "relative_residual": float(np.linalg.norm(x - l - s, "fro") / norm_x),
        "sparsity": float(np.mean(np.abs(s) <= 1e-12)),
    }
    if refine_rank:
        refined = debias_low_rank(x, s, int(refine_rank))
        out["refined_low_rank"] = refined
        out["refine_rank"] = int(refine_rank)
        out["refined_relative_residual"] = float(
            np.linalg.norm(x - refined - s, "fro") / norm_x
        )
    return out


def robust_scale(values: np.ndarray) -> float:
    """Normal-consistent MAD: robust standard deviation, 50% breakdown point."""
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        return 0.0
    med = float(np.median(flat))
    return 1.4826 * float(np.median(np.abs(flat - med)))


class StreamingRobustPCA(StatisticalOptimizer):
    """Low-rank subspace from an EMA of element-wise trimmed second moments.

    Corruption model: ``x = l + s`` with ``s`` element-wise sparse gross values
    at a stable magnitude.  An entry further than ``threshold_scale`` robust
    standard deviations from the data median is treated as corrupted and pulled
    to the median before the outer product is formed.  The state is again only
    the operator; the basis is a decode of it.
    """

    name = "streaming_robust_pca"
    state_keys = ("C",)

    def __init__(
        self,
        n_components: int,
        *,
        updater: Any = None,
        threshold_scale: float = 3.0,
        power_iters: int = 2,
        sample_size: int = 4096,
        seed: int = 0,
    ) -> None:
        super().__init__(updater or EMA(alpha=0.1))
        self.n_components = int(n_components)
        self.threshold_scale = float(threshold_scale)
        self.power_iters = int(power_iters)
        self.sample_size = int(sample_size)
        self.seed = int(seed)

    # -- trimming ----------------------------------------------------------- #
    def _trim(self, data: np.ndarray, center: float, threshold: float) -> np.ndarray:
        """Pull entries beyond the robust scale to the data median."""
        data = np.asarray(data, dtype=np.float64)
        return np.where(np.abs(data - center) <= threshold, data, center)

    def sparse_component(self, data: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        """Entries that are both far from the data median and unexplained.

        Residual magnitude alone is not enough: the projection mixes coordinates,
        so one gross entry makes the *other* coordinates of its row look
        unexplained too (measured 17% false positives on clean entries at a
        tight rank-2 fit).  Requiring the entry itself to exceed the data-scale
        threshold removes that leakage.
        """
        data = np.asarray(data, dtype=np.float64)
        basis = np.asarray(parameters["components"], dtype=np.float64)
        threshold = float(parameters["threshold"])
        center = float(parameters.get("center", 0.0))
        residual = data - (data @ basis) @ basis.T
        corrupted = (np.abs(data - center) > threshold) & (np.abs(residual) > threshold)
        return np.where(corrupted, residual, 0.0)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        rng = np.random.default_rng(self.seed)
        sample = data
        if n > self.sample_size:
            sample = data[rng.choice(n, size=self.sample_size, replace=False)]
        center = float(np.median(sample))
        threshold = self.threshold_scale * max(robust_scale(sample.ravel()), _EPS)
        trimmed = self._trim(sample, center, threshold)
        cov = (trimmed.T @ trimmed) / max(trimmed.shape[0], 1)
        k = min(self.n_components, d)
        warm = top_basis(cov, k, iters=max(self.power_iters, 4))
        return State(
            {"C": cov, "basis": warm},
            0,
            {"n_features": d, "threshold": threshold, "center": center},
        )

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        center = float(state.meta.get("center", 0.0))
        threshold = float(state.meta["threshold"])
        trimmed = self._trim(batch, center, threshold)
        cov = (trimmed.T @ trimmed) / max(batch.shape[0], 1)
        basis = np.asarray(state["basis"], dtype=np.float64)
        residual = batch - (batch @ basis) @ basis.T
        unexplained = np.abs(residual) > threshold
        return State(
            {"C": cov},
            0,
            {
                "trimmed_fraction": float(np.mean(np.abs(batch - center) > threshold)),
                "sparse_fraction": float(np.mean(unexplained)),
            },
        )

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        new = super().update_state(state, statistics, ctx)
        warm = np.asarray(state["basis"], dtype=np.float64)
        new.values["basis"] = top_basis(
            new["C"], self.n_components, warm=warm, iters=self.power_iters
        )
        for key in ("threshold", "center", "n_features"):
            if key in state.meta:
                new.meta[key] = state.meta[key]
        for key in ("trimmed_fraction", "sparse_fraction"):
            if key in statistics.meta:
                new.meta[key] = statistics.meta[key]
        return new

    def decode(self, state: State) -> dict[str, Any]:
        basis = np.asarray(state["basis"], dtype=np.float64)
        cov = np.asarray(state["C"], dtype=np.float64)
        total = float(np.trace(cov))
        captured = float(np.trace(basis.T @ cov @ basis))
        return {
            "components": basis,
            "covariance": cov,
            "threshold": float(state.meta.get("threshold", 0.0)),
            "center": float(state.meta.get("center", 0.0)),
            "trimmed_fraction": float(state.meta.get("trimmed_fraction", 0.0)),
            "explained_ratio": captured / total if total > _EPS else 0.0,
        }

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared residual after removing the low-rank and sparse parts."""
        data = np.asarray(data, dtype=np.float64)
        basis = np.asarray(parameters["components"], dtype=np.float64)
        residual = data - (data @ basis) @ basis.T
        sparse = self.sparse_component(data, parameters)
        return float(np.mean(np.sum((residual - sparse) ** 2, axis=1)))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = 256,
        nepochs: int = 5,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory)
        return fit(self, data, cfg, eval_data=eval_data)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_components": self.n_components,
                    "threshold_scale": self.threshold_scale})
        return out
