"""Non-negative matrix factorization.

``X ~ U V^T`` with ``U`` (n, k) and ``V`` (d, k) non-negative.  The conventional
reference is Lee & Seung multiplicative updates on the full matrix.  The stateful
variant keeps the two sufficient statistics an *exact* alternating least-squares
update of the basis needs::

    A = X^T U      (d, k)
    B = U^T U      (k, k)

folds each mini-batch's ``X_b^T U_b`` / ``U_b^T U_b`` in with an EMA, and decodes
the basis by solving ``B V^T = A^T`` (clipped at zero).  Coefficients for the
current batch are solved against that basis.  The state is ``O(dk + k^2)`` --
independent of ``n`` -- which is what makes the mini-batch variant cheap.
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
from ..state import EMA, Replace
from ..state.sufficient import SufficientStat

_EPS = 1e-12


def nmf_coef(data: np.ndarray, basis: np.ndarray, ridge: float) -> np.ndarray:
    """Non-negative coefficients of ``data`` against ``basis`` (ridge LS)."""
    gram = basis.T @ basis + ridge * np.eye(basis.shape[1])
    u = np.linalg.solve(gram, basis.T @ data.T).T
    return np.maximum(u, 0.0)


def nmf_solve_right(a: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
    """Columns ``V`` with ``V B = A`` -> ``V = A B^{-1}`` (ridge-stabilised)."""
    reg = b + ridge * np.eye(b.shape[0])
    return np.linalg.solve(reg, a.T).T


class NMFStat(SufficientStat):
    """The two NMF moment blocks as a sufficient statistic.

    Values are exactly what ``NMF.measure`` produces -- ``A = X_b^T U_b`` and
    ``B = U_b^T U_b`` -- so decoding with a fixed basis is the batch answer, and
    the basis is refreshed from the accumulated pair every ``act_every`` blocks.
    """

    name = "nmf_stat"
    keys = ("A", "B")

    def __init__(
        self,
        n_components: int,
        *,
        ridge: float = 1e-8,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every, seed=seed)
        self.n_components = int(n_components)
        self.ridge = float(ridge)
        self.d: int | None = None

    def prepare(self, basis: np.ndarray | None = None, *, d: int | None = None) -> "NMFStat":
        if basis is None:
            if d is None:  # pragma: no cover - defensive
                raise ValueError("NMFStat.prepare needs a basis or d=")
            rng = np.random.default_rng(self.seed)
            basis = np.maximum(rng.random((int(d), self.n_components)), _EPS)
        basis = np.asarray(basis, dtype=np.float64)
        self.d = int(basis.shape[0])
        self.params["basis"] = np.maximum(np.array(basis, copy=True), 0.0)
        self._dirty = False
        return self

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if "basis" not in self.params:
            self.prepare(d=int(block.shape[1]))
        basis = np.asarray(self.params["basis"], dtype=np.float64)
        coef = nmf_coef(block, basis, self.ridge)
        return {
            "A": block.T @ coef,
            "B": coef.T @ coef,
            "weight": np.array([weight], dtype=np.float64),
        }

    def act(self) -> None:
        if "A" not in self.values:
            return
        a = np.asarray(self.values["A"], dtype=np.float64)
        b = np.asarray(self.values["B"], dtype=np.float64)
        # ``nmf_solve_right`` adds the ridge itself; handing it a matrix that
        # already had one applied solved with two ridges, not one.
        basis = nmf_solve_right(a, b, self.ridge)
        self.params["basis"] = np.maximum(basis, 0.0)
        self._dirty = False

    def decode(self) -> dict[str, Any]:
        self._maybe_act()
        basis = np.asarray(self.params["basis"], dtype=np.float64)
        norms = np.linalg.norm(basis, axis=0)
        return {
            "basis": basis,
            "scale": norms,
            "weights": norms / max(norms.sum(), _EPS),
        }

    def extra_config(self) -> dict[str, Any]:
        return {"n_components": self.n_components, "ridge": self.ridge, "d": self.d}

    def estimate_state_bytes(self, d: int) -> int:
        k = self.n_components
        return int((d * k + k * k + k) * 8 * 2)


class NMF(StatisticalOptimizer):
    """Persistent-statistics NMF (alternating least squares with non-negativity)."""

    name = "nmf"
    state_keys = ("A", "B")
    supports_streaming = True

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        k = self.n_components if k is None else int(k)
        return int((d * k + k * k) * 8 * 2)

    def stat(self, *, mode: str = "sum", alpha: Any = 1.0) -> NMFStat:
        """The same moment blocks as a mergeable sufficient statistic."""
        return NMFStat(self.n_components, ridge=self.ridge, mode=mode, alpha=alpha,
                       seed=self.seed)

    def __init__(
        self,
        n_components: int,
        *,
        updater: Any = None,
        ridge: float = 1e-8,
        init: str = "random",
        seed: int = 0,
        quality: bool = False,
        quality_epochs: int = 250,
        quality_restarts: int = 3,
    ) -> None:
        super().__init__(updater or Replace())
        self.n_components = int(n_components)
        self.ridge = float(ridge)
        self.init = init
        self.seed = int(seed)
        #: Full-batch ALS needs roughly 250 alternating updates to reach the
        #: reconstruction floor sklearn's 1,000 multiplicative updates reaches.
        #: The shipped 20-epoch default is the streaming budget, so quality is
        #: opt-in rather than a silent 12x cost increase.
        self.quality = bool(quality)
        self.quality_epochs = max(1, int(quality_epochs))
        #: ALS reaches different non-negative fixed points from different random
        #: initialisations (the 800-row fixture has one basin at 5e-6 and one at
        #: 2e-7); a quality claim should not depend on drawing the lucky one.
        self.quality_restarts = max(1, int(quality_restarts))

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _solve_right(a: np.ndarray, b: np.ndarray, ridge: float) -> np.ndarray:
        """Columns ``V`` with ``V B = A`` -> ``V = A B^{-1}`` (ridge-stabilised)."""
        return nmf_solve_right(a, b, ridge)

    def _coef(self, data: np.ndarray, basis: np.ndarray, ridge: float) -> np.ndarray:
        return nmf_coef(np.asarray(data, dtype=np.float64), basis, ridge)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        k = self.n_components
        rng = np.random.default_rng(self.seed)
        u0 = np.maximum(rng.random((n, k)), _EPS)
        a = data.T @ u0
        b = u0.T @ u0
        return State({"A": a, "B": b}, 0, {"n_features": d, "n_samples": n})

    def decode(self, state: State) -> dict[str, Any]:
        a = np.asarray(state["A"], dtype=np.float64)
        b = np.asarray(state["B"], dtype=np.float64)
        # one ridge, added by the solve itself (it used to be added twice here)
        basis = self._solve_right(a, b, self.ridge)
        basis = np.maximum(basis, 0.0)
        norms = np.linalg.norm(basis, axis=0)
        return {
            "basis": basis,
            "scale": norms,
            "weights": norms / max(norms.sum(), _EPS),
        }

    # -- four stages -------------------------------------------------------- #
    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        basis = self.decode(state)["basis"]
        coef = self._coef(batch, basis, self.ridge)
        return State(
            {"A": batch.T @ coef, "B": coef.T @ coef},
            0,
            {"batch_error": float(np.mean((batch - coef @ basis.T) ** 2))},
        )

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared reconstruction error of the non-negative factors."""
        data = np.asarray(data, dtype=np.float64)
        basis = parameters["basis"]
        coef = self._coef(data, basis, self.ridge)
        residual = data - coef @ basis.T
        return float(np.mean(residual**2))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int | None = None,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
        quality: bool | None = None,
    ) -> FitResult:
        use_quality = self.quality if quality is None else bool(quality)
        if nepochs is None:
            nepochs = self.quality_epochs if use_quality else 20
        elif use_quality:
            nepochs = max(int(nepochs), self.quality_epochs)
        restarts = self.quality_restarts if use_quality else 1
        base_seed = int(seed)
        best: FitResult | None = None
        for restart in range(restarts):
            self.seed = base_seed + restart
            cfg = FitConfig(batch_size=batch_size, nepochs=int(nepochs),
                            seed=self.seed, track_memory=track_memory)
            result = fit(self, data, cfg, eval_data=eval_data)
            if best is None or result.objective < best.objective:
                best = result
        self.seed = base_seed
        assert best is not None
        if use_quality:
            best.extras["quality_restarts"] = restarts
            best.extras["quality_seed"] = int(best.extras.get("seed", base_seed))
        return best

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_components": self.n_components, "ridge": self.ridge,
                    "quality": self.quality,
                    "quality_epochs": self.quality_epochs,
                    "quality_restarts": self.quality_restarts})
        return out


def mu_nmf(
    data: np.ndarray,
    n_components: int,
    *,
    niter: int = 200,
    seed: int = 0,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Reference NMF: Lee & Seung multiplicative updates on the full matrix.

    Returns ``(U, V, losses)``.  This is the correctness oracle for the stateful
    variant: same model, same objective, a different update law.
    """
    data = np.asarray(data, dtype=np.float64)
    if np.any(data < 0):
        raise ValueError("NMF requires non-negative data")
    n, d = data.shape
    k = int(n_components)
    rng = np.random.default_rng(seed)
    scale = np.sqrt(max(data.mean(), _EPS) / k)
    u = np.maximum(rng.random((n, k)) * scale, _EPS)
    v = np.maximum(rng.random((d, k)) * scale, _EPS)
    losses: list[float] = []
    for _ in range(int(niter)):
        v *= (data.T @ u) / (v @ (u.T @ u) + _EPS)
        u *= (data @ v) / (u @ (v.T @ v) + _EPS)
        loss = float(np.mean((data - u @ v.T) ** 2))
        losses.append(loss)
        if len(losses) > 1 and abs(losses[-2] - loss) <= tol * max(1.0, losses[-2]):
            break
    return u, v, losses
