"""Updater base class: the contract every state-update law implements."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core import State, UpdateContext


class StateUpdater:
    """Folds a batch statistic into persistent state.

    Subclasses implement :meth:`update`.  ``keys`` restricts the update to a
    subset of state keys, which is how an algorithm applies different laws to
    different statistics (e.g. bias-corrected EMA on sums, plain EMA on counts).
    """

    name: str = "base"

    def alpha(self, ctx: UpdateContext) -> float:
        """Current step size, for diagnostics and for algorithms that need it."""
        return 1.0

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:  # pragma: no cover - abstract
        raise NotImplementedError

    def reset(self) -> None:
        """Clear internal recursion state.

        ``fit`` does **not** call this: every recursion here lives in
        ``State.meta``, so a new fit starts from a new ``State`` and there is
        nothing object-side to clear.  It exists for updaters that keep
        object-level state (``PortfolioEMA._weights`` is diagnostic only) and
        for an embedder reusing one updater object across runs.
        """

    def config(self) -> dict[str, Any]:
        """JSON-serialisable description, for experiment records."""
        out: dict[str, Any] = {"name": self.name}
        for key, value in vars(self).items():
            if key in {"per_key"}:
                out[key] = sorted(value) if isinstance(value, Mapping) else value
            elif hasattr(value, "alpha") and not isinstance(value, (int, float)):
                out[key] = type(value).__name__
            elif isinstance(value, (int, float, str, bool)) or value is None:
                out[key] = value
        return out

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def _stat(statistics: State, key: str) -> np.ndarray | None:
        if key not in statistics.values:
            return None
        return np.asarray(statistics.values[key], dtype=np.float64)

    @staticmethod
    def _same_shape(previous: State, key: str, statistic: np.ndarray) -> bool:
        prev = previous.values.get(key)
        return prev is not None and np.shape(prev) == np.shape(statistic)

    @staticmethod
    def _fresh(previous: State, key: str, statistic: np.ndarray) -> State:
        values = dict(previous.values)
        values[key] = np.array(statistic, dtype=np.float64, copy=True)
        return State(values, previous.step, dict(previous.meta))
