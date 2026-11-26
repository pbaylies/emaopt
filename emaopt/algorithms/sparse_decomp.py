"""Sparse diagonal + tied decomposition: the bridge from mixtures to sparse coding.

Model: ``x_i ~ sum_k r_ik N(w_k, diag(sigma^2))`` with an *L0 sparsity* constraint
-- each point may use at most ``s`` components.  Component structure is
per-component (means ``w_k``) while the covariance is a single shared diagonal
``sigma^2`` estimated from the global second-moment state, so the state is

    N_k  (per component)
    S_k  (per component, only s components active per point)
    Q_j  (global, j = 1..d:  sum_{i,k} r_ik x_ij^2)

rather than a ``D x D`` covariance per component.  This is the first-class
Tier-1 experiment in the PLAN: it tests whether persistent statistics help as
components stop being isotropic and start competing for sparse support.
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
    squared_distances,
)
from ..state import EMA, Replace

_LOG_2PI = np.log(2.0 * np.pi)


class SparseDiagTied(StatisticalOptimizer):
    """Sparse soft-assignment decomposition with a shared diagonal covariance."""

    name = "sparse_diag_tied"
    state_keys = ("counts", "sums", "sq")

    def __init__(
        self,
        n_components: int,
        *,
        sparsity: int = 2,
        updater: Any = None,
        reg_var: float = 1e-6,
        init: str = "kmeans++",
        seed: int = 0,
    ) -> None:
        super().__init__(updater or Replace())
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1")
        if int(sparsity) < 1:
            raise ValueError("sparsity must be >= 1")
        if float(reg_var) <= 0.0:
            raise ValueError("reg_var must be > 0")
        self.n_components = int(n_components)
        self.sparsity = int(sparsity)
        self.reg_var = float(reg_var)
        self.init = init
        self.seed = int(seed)

    # -- init --------------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        k = self.n_components
        from .kmeans import LloydKMeans

        seed_model = LloydKMeans(k, niter=10, seed=self.seed)
        atoms = seed_model.fit(data, niter=10).params["centroids"]
        counts = np.full(k, n / k)
        sums = atoms * (n / k)
        sq = np.sum(data**2, axis=0)
        return State(
            {"counts": counts, "sums": sums, "sq": sq},
            0,
            {"init_atoms": atoms.copy(), "n_features": d},
        )

    # -- decode ------------------------------------------------------------- #
    def decode(self, state: State) -> dict[str, Any]:
        counts = np.asarray(state["counts"], dtype=np.float64)
        sums = np.asarray(state["sums"], dtype=np.float64)
        sq = np.asarray(state["sq"], dtype=np.float64)
        safe = np.maximum(counts, 1e-12)
        atoms = sums / safe[:, None]
        empty = counts <= 1e-9
        if np.any(empty):
            atoms[empty] = state.meta.get("init_atoms", atoms)[empty]
        total = max(float(counts.sum()), 1e-12)
        explained = (counts[:, None] * atoms**2).sum(axis=0)
        sigma2 = np.maximum((sq - explained) / total, self.reg_var)
        return {
            "atoms": atoms,
            "sigma2": sigma2,
            "weights": counts / total,
            "counts": counts,
        }

    def encode(self, data: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        """Top-``s`` responsibility codes; rows sum to one over the support.

        The quadratic form is expanded through two BLAS products instead of
        materialising ``(n, k, d)`` differences.  At the wide LAION cell the old
        form allocated ~2 GB per 8 192-row batch; this is exact and keeps the
        working set at ``(n, k)``.
        """
        data = np.asarray(data, dtype=np.float64)
        atoms = np.asarray(parameters["atoms"], dtype=np.float64)
        sigma2 = np.asarray(parameters["sigma2"], dtype=np.float64)
        inv_var = 1.0 / sigma2
        # ||x - w_k||^2 / sigma2, expanded as in the dense GMM E-step.
        atom_inv = atoms * inv_var
        quad = (data * data) @ inv_var
        quad = quad[:, None] - 2.0 * (data @ atom_inv.T)
        quad += np.einsum("kd,kd->k", atoms, atom_inv)[None, :]
        np.maximum(quad, 0.0, out=quad)
        logits = -0.5 * quad + np.log(
            np.maximum(parameters["weights"], 1e-300)
        )[None, :]
        s = min(self.sparsity, atoms.shape[0])
        if s < atoms.shape[0]:
            keep = np.argpartition(-logits, s - 1, axis=1)[:, :s]
            mask = np.zeros_like(logits, dtype=bool)
            np.put_along_axis(mask, keep, True, axis=1)
            logits = np.where(mask, logits, -np.inf)
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-300)

    # -- four stages -------------------------------------------------------- #
    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        params = self.decode(state)
        codes = self.encode(batch, params)
        values = {
            "counts": codes.sum(axis=0),
            "sums": codes.T @ batch,
            "sq": np.sum(batch**2, axis=0),
        }
        recon = codes @ params["atoms"]
        err = float(np.mean(np.sum((batch - recon) ** 2, axis=1)))
        return State(values, 0, {"reconstruction": err})

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared reconstruction error under the sparse codes."""
        data = np.asarray(data, dtype=np.float64)
        codes = self.encode(data, parameters)
        recon = codes @ parameters["atoms"]
        return float(np.mean(np.sum((data - recon) ** 2, axis=1)))

    def sparsity_level(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Fraction of exactly-zero code entries."""
        codes = self.encode(np.asarray(data, dtype=np.float64), parameters)
        return float(np.mean(codes <= 1e-12))

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        """Persistent state is ``k`` counts, ``k*d`` sums and ``d`` squares."""
        k = self.n_components if k is None else int(k)
        return int(8 * (k + k * d + d))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int = 10,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
        drop_last: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory, drop_last=drop_last)
        return fit(self, data, cfg, eval_data=eval_data)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_components": self.n_components, "sparsity": self.sparsity})
        return out
