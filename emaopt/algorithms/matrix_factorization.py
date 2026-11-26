"""Matrix factorization ``X ~ U V^T`` (no non-negativity constraint).

Same sufficient statistics as NMF -- ``X^T U`` and ``U^T U`` -- but solved by
least squares.  Two baselines live here: the persistent-statistics ALS arm and a
plain SGD arm, because "online MF" is a well-trodden control in the PLAN.
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
    fit,
)
from ..state import EMA, Replace

_EPS = 1e-12


class ALSFactorization(StatisticalOptimizer):
    """Alternating least squares over persistent ``(A, B)`` statistics."""

    name = "als_mf"
    state_keys = ("A", "B")

    def __init__(self, n_factors: int, *, updater: Any = None, ridge: float = 1e-6,
                 seed: int = 0, sample_size: int = 4096) -> None:
        super().__init__(updater or Replace())
        self.n_factors = int(n_factors)
        self.ridge = float(ridge)
        self.seed = int(seed)
        self.sample_size = int(sample_size)

    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        rng = np.random.default_rng(self.seed)
        sample = data
        if n > self.sample_size:
            sample = data[rng.choice(n, size=self.sample_size, replace=False)]
        u0 = rng.normal(scale=np.sqrt(max(float(data.var()), _EPS)), size=(sample.shape[0],
                                                                          self.n_factors))
        return State(
            {"A": sample.T @ u0, "B": u0.T @ u0},
            0,
            {"n_features": d, "n_samples": n},
        )

    def decode(self, state: State) -> dict[str, Any]:
        a = np.asarray(state["A"], dtype=np.float64)
        b = np.asarray(state["B"], dtype=np.float64)
        reg = b + self.ridge * np.eye(b.shape[0])
        basis = np.linalg.solve(reg, a.T).T
        return {"basis": basis, "gram": b}

    def _coef(self, data: np.ndarray, basis: np.ndarray) -> np.ndarray:
        gram = basis.T @ basis + self.ridge * np.eye(basis.shape[1])
        return np.linalg.solve(gram, basis.T @ data.T).T

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        basis = self.decode(state)["basis"]
        coef = self._coef(batch, basis)
        return State({"A": batch.T @ coef, "B": coef.T @ coef}, 0, {})

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        data = np.asarray(data, dtype=np.float64)
        basis = parameters["basis"]
        recon = self._coef(data, basis) @ basis.T
        return float(np.sqrt(np.mean((data - recon) ** 2)))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int = 15,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory)
        return fit(self, data, cfg, eval_data=eval_data)


class SGDFactorization:
    """Plain SGD matrix factorization with a decaying step (online baseline)."""

    name = "sgd_mf"

    def __init__(self, n_factors: int, *, lr: float = 1.5, niter: int = 20,
                 seed: int = 0) -> None:
        self.n_factors = int(n_factors)
        self.lr = float(lr)
        self.niter = int(niter)
        self.seed = int(seed)

    def fit(self, data: np.ndarray) -> dict[str, Any]:
        data = np.asarray(data, dtype=np.float64)
        n, d = data.shape
        k = self.n_factors
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(max(float(data.var()) / max(k, 1), _EPS))
        u = rng.normal(scale=scale, size=(n, k))
        v = rng.normal(scale=scale, size=(d, k))
        losses: list[float] = []
        lr0 = self.lr
        # Step from the *data* scale, estimated once: a curvature read from the
        # factors themselves creates a feedback loop (a shrinking V inflates lr
        # without bound -- observed 385 RMSE).  Capped at the measured stability
        # bound for this loss (0.5 converges, 0.8 diverges on a rank-2 problem).
        data_scale = float(np.mean(np.sum(data**2, axis=1))) / max(d, 1)
        lr = min(lr0 / max(data_scale, 1e-12), 0.5)
        for _ in range(self.niter):
            order = rng.permutation(n)
            for i in order:
                err = data[i] - u[i] @ v.T
                np.clip(err, -10.0, 10.0, out=err)
                old_u = u[i].copy()
                u[i] += lr * (v.T @ err) / d
                v += lr * np.outer(err, old_u) / d
                if not (np.all(np.isfinite(u[i])) and np.all(np.isfinite(v))):
                    raise FloatingPointError("SGD factorization diverged")
            losses.append(float(np.sqrt(np.mean((data - u @ v.T) ** 2))))
        rmse = float(np.sqrt(np.mean((data - u @ v.T) ** 2)))
        return {"u": u, "v": v, "rmse": rmse, "losses": losses}
