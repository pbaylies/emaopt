"""Replace: the conventional one-pass rebuild (Lloyd / batch EM)."""

from __future__ import annotations

import numpy as np

from ..core import State, UpdateContext
from .base import StateUpdater


class Replace(StateUpdater):
    """``S <- s``.

    This is the "throw the past away" law.  It is a first-class member of the
    library because every EMA result must be compared against it on the same
    sampler, initialization and budget: with ``batch_size = n`` it reproduces
    Lloyd's algorithm and batch EM exactly.
    """

    name = "replace"

    def alpha(self, ctx: UpdateContext) -> float:
        return 1.0

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:
        keys = keys or tuple(statistics.values)
        values = dict(previous.values)
        for key in keys:
            statistic = self._stat(statistics, key)
            if statistic is None:
                continue
            values[key] = np.array(statistic, copy=True)
        return State(values, previous.step, dict(previous.meta))
