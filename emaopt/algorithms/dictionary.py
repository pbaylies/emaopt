"""Dictionary learning: sparse codes plus a shared dictionary.

``X ~ D Z`` with ``D`` (d, m) and ``Z`` (n, m) sparse (L1).  The dictionary
update (MOD) needs exactly two statistics::

    G = X^T Z      (d, m)
    H = Z^T Z      (m, m)

and solves ``D H = G``.  Those statistics are the thing a batch algorithm throws
away after every pass and a stateful algorithm keeps.
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


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def ista(
    data: np.ndarray,
    dictionary: np.ndarray,
    lam: float,
    *,
    niter: int = 60,
    tol: float = 1e-7,
) -> np.ndarray:
    """L1 sparse codes by ISTA (``0.5||x - Dz||^2 + lam ||z||_1``)."""
    data = np.asarray(data, dtype=np.float64)
    d = np.asarray(dictionary, dtype=np.float64)
    lip = float(np.linalg.norm(d, 2) ** 2) + _EPS
    z = np.zeros((data.shape[0], d.shape[1]), dtype=np.float64)
    prev = None
    for _ in range(int(niter)):
        grad = (z @ d.T - data) @ d
        z = soft_threshold(z - grad / lip, lam / lip)
        if prev is not None and np.max(np.abs(z - prev)) <= tol * max(1.0, np.max(np.abs(z))):
            break
        prev = z
    return z


class DictionaryStat(SufficientStat):
    """The MOD dictionary statistic ``G = X^T Z``, ``H = Z^T Z``.

    Values are exactly ``DictionaryLearner.measure``'s arrays; the atom matrix is
    derived by solving ``D H = G`` and normalising, so the stat can stream,
    merge and report its ``d*m + m*m`` footprint without the data.
    """

    name = "dictionary_stat"
    keys = ("G", "H")

    def __init__(
        self,
        n_atoms: int,
        *,
        lam: float = 0.1,
        ridge: float = 1e-6,
        ista_iters: int = 40,
        mode: str = "sum",
        alpha: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
        d: int | None = None,
    ) -> None:
        super().__init__(mode=mode, alpha=alpha, act_every=act_every, seed=seed)
        self.n_atoms = int(n_atoms)
        self.lam = float(lam)
        self.ridge = float(ridge)
        self.ista_iters = int(ista_iters)
        self.d = None if d is None else int(d)

    def prepare(
        self, atoms: np.ndarray | None = None, *, d: int | None = None
    ) -> "DictionaryStat":
        if atoms is None:
            d = self.d if d is None else int(d)
            if d is None:  # pragma: no cover - defensive
                raise ValueError("DictionaryStat.prepare needs atoms or d=")
            rng = np.random.default_rng(self.seed)
            atoms = rng.normal(size=(int(d), self.n_atoms))
            atoms /= np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), _EPS)
        atoms = np.asarray(atoms, dtype=np.float64)
        self.d = int(atoms.shape[0])
        self.meta["init_atoms"] = np.array(atoms, copy=True)
        self.params["atoms"] = np.array(atoms, copy=True)
        self._dirty = False
        return self

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if "atoms" not in self.params:
            self.prepare(d=int(block.shape[1]))
        atoms = np.asarray(self.params["atoms"], dtype=np.float64)
        codes = ista(block, atoms, self.lam, niter=self.ista_iters)
        return {"G": block.T @ codes, "H": codes.T @ codes,
                "weight": np.array([weight], dtype=np.float64)}

    def act(self) -> None:
        if "G" not in self.values:
            return
        g = np.asarray(self.values["G"], dtype=np.float64)
        h = np.asarray(self.values["H"], dtype=np.float64)
        init = np.asarray(self.meta["init_atoms"], dtype=np.float64)
        if float(np.abs(h).sum()) <= _EPS:
            atoms = init.copy()
        else:
            reg = h + self.ridge * np.eye(h.shape[0])
            atoms = np.linalg.solve(reg, g.T).T
            dead = np.linalg.norm(atoms, axis=0) <= _EPS
            if np.any(dead):
                atoms[:, dead] = init[:, dead]
        self.params["atoms"] = atoms / np.maximum(
            np.linalg.norm(atoms, axis=0, keepdims=True), _EPS
        )
        self._dirty = False

    def decode(self) -> dict[str, Any]:
        self._maybe_act()
        return {
            "atoms": np.asarray(self.params["atoms"], dtype=np.float64),
            "gram": np.asarray(self.values.get("H", np.zeros((self.n_atoms, self.n_atoms))),
                               dtype=np.float64),
            "n_atoms": self.n_atoms,
        }

    def extra_config(self) -> dict[str, Any]:
        return {"n_atoms": self.n_atoms, "lam": self.lam, "ista_iters": self.ista_iters,
                "d": self.d}


class DictionaryLearner(StatisticalOptimizer):
    """MOD dictionary learning over persistent ``(G, H)`` statistics."""

    name = "dictionary"
    state_keys = ("G", "H")
    supports_streaming = True

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        m = self.n_atoms
        return int((d * m + m * m) * 8 * 2)

    def stat(self, *, mode: str = "sum", alpha: Any = 1.0) -> DictionaryStat:
        """The same ``(G, H)`` pair as a mergeable sufficient statistic."""
        return DictionaryStat(self.n_atoms, lam=self.lam, ridge=self.ridge,
                              ista_iters=self.ista_iters, mode=mode, alpha=alpha,
                              seed=self.seed)

    def __init__(
        self,
        n_atoms: int,
        *,
        lam: float = 0.1,
        updater: Any = None,
        ridge: float = 1e-6,
        ista_iters: int = 40,
        seed: int = 0,
    ) -> None:
        super().__init__(updater or Replace())
        self.n_atoms = int(n_atoms)
        self.lam = float(lam)
        self.ridge = float(ridge)
        self.ista_iters = int(ista_iters)
        self.seed = int(seed)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        d = data.shape[1]
        rng = np.random.default_rng(self.seed)
        atoms = rng.normal(size=(d, self.n_atoms))
        atoms /= np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), _EPS)
        return State(
            {"G": np.zeros((d, self.n_atoms)), "H": np.zeros((self.n_atoms, self.n_atoms))},
            0,
            {"init_atoms": atoms, "n_features": d},
        )

    def decode(self, state: State) -> dict[str, Any]:
        g = np.asarray(state["G"], dtype=np.float64)
        h = np.asarray(state["H"], dtype=np.float64)
        init = np.asarray(state.meta.get("init_atoms"))
        if float(np.abs(h).sum()) <= _EPS:
            atoms = init.copy()
        else:
            reg = h + self.ridge * np.eye(h.shape[0])
            atoms = np.linalg.solve(reg, g.T).T
            dead = np.linalg.norm(atoms, axis=0) <= _EPS
            if np.any(dead):
                atoms[:, dead] = init[:, dead]
        atoms = atoms / np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), _EPS)
        return {"atoms": atoms, "gram": h, "n_atoms": self.n_atoms}

    # -- four stages -------------------------------------------------------- #
    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        atoms = self.decode(state)["atoms"]
        codes = ista(batch, atoms, self.lam, niter=self.ista_iters)
        return State(
            {"G": batch.T @ codes, "H": codes.T @ codes},
            0,
            {"batch_error": float(np.mean((batch - codes @ atoms.T) ** 2)),
             "batch_nnz": float(np.mean(codes != 0.0))},
        )

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        data = np.asarray(data, dtype=np.float64)
        codes = ista(data, parameters["atoms"], self.lam, niter=self.ista_iters)
        return float(np.mean((data - codes @ parameters["atoms"].T) ** 2))

    def codes(self, data: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        return ista(np.asarray(data, dtype=np.float64), parameters["atoms"], self.lam,
                    niter=self.ista_iters)

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

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_atoms": self.n_atoms, "lam": self.lam})
        return out
