"""Subspace tracking: persist the operator, never the eigenvectors.

The PLAN's Tier-3 warning is that averaging eigenvectors is ill-posed (sign,
rotation, crossings, changing gaps).  The well-posed state is the second-moment
*operator*; the basis is re-derived from it by block power iteration, warm
started from the previous basis.  What is tracked (and tested) is the subspace
projection, which is invariant to the sign and rotation freedom the basis has.
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

_EPS = 1e-12


def orthonormalise(z: np.ndarray) -> np.ndarray:
    """Thin QR with a deterministic sign convention (largest entry positive)."""
    q, _ = np.linalg.qr(np.asarray(z, dtype=np.float64))
    q = q[:, : np.asarray(z).shape[1]]
    idx = np.argmax(np.abs(q), axis=0)
    signs = np.sign(q[idx, np.arange(q.shape[1])])
    q = q * np.where(signs == 0.0, 1.0, signs)[None, :]
    return q


def top_basis(operator: np.ndarray, n_components: int, *, warm: np.ndarray | None = None,
              iters: int = 2) -> np.ndarray:
    """Leading ``n_components`` eigenvectors of a PSD operator by block power iteration."""
    operator = np.asarray(operator, dtype=np.float64)
    d = operator.shape[0]
    k = int(min(n_components, d))
    if warm is not None and warm.shape == (d, k):
        basis = warm
    else:
        basis = orthonormalise(np.eye(d)[:, :k])
    for _ in range(max(1, int(iters))):
        basis = orthonormalise(operator @ basis)
    return basis


class SubspaceTracker(StatisticalOptimizer):
    """EMA second-moment operator with a power-iteration subspace decode."""

    name = "subspace_tracker"
    state_keys = ("C",)

    def __init__(
        self,
        n_components: int,
        *,
        updater: Any = None,
        power_iters: int = 2,
        sample_size: int = 4096,
        seed: int = 0,
    ) -> None:
        super().__init__(updater or EMA(alpha=0.1))
        self.n_components = int(n_components)
        self.power_iters = int(power_iters)
        self.sample_size = int(sample_size)
        self.seed = int(seed)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        rng = np.random.default_rng(self.seed)
        sample = data
        if n > self.sample_size:
            sample = data[rng.choice(n, size=self.sample_size, replace=False)]
        cov = (sample.T @ sample) / max(sample.shape[0], 1)
        k = min(self.n_components, d)
        warm = orthonormalise(np.eye(d)[:, :k])
        return State({"C": cov, "basis": top_basis(cov, k, warm=warm,
                                                  iters=self.power_iters)}, 0,
                     {"n_features": d})

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        return State({"C": (batch.T @ batch) / max(batch.shape[0], 1)}, 0, {})

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        new = super().update_state(state, statistics, ctx)
        warm = np.asarray(state["basis"], dtype=np.float64)
        new.values["basis"] = top_basis(new["C"], self.n_components, warm=warm,
                                        iters=self.power_iters)
        return new

    def decode(self, state: State) -> dict[str, Any]:
        basis = np.asarray(state["basis"], dtype=np.float64)
        cov = np.asarray(state["C"], dtype=np.float64)
        total = float(np.trace(cov))
        captured = float(np.trace(basis.T @ cov @ basis))
        return {
            "components": basis,
            "covariance": cov,
            "explained_ratio": captured / total if total > _EPS else 0.0,
        }

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared residual outside the tracked subspace."""
        data = np.asarray(data, dtype=np.float64)
        basis = np.asarray(parameters["components"], dtype=np.float64)
        total = 0.0
        for start in range(0, data.shape[0], 4096):
            block = data[start : start + 4096]
            residual = block - (block @ basis) @ basis.T
            total += float(np.sum(residual**2))
        return total / max(data.shape[0], 1)

    def explained_energy(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        data = np.asarray(data, dtype=np.float64)
        total = float(np.mean(np.sum(data**2, axis=1)))
        if total <= _EPS:
            return 1.0
        return float(1.0 - self.objective(data, parameters) / total)

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
        out.update({"n_components": self.n_components, "power_iters": self.power_iters})
        return out
