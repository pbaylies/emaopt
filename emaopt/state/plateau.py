"""PlateauEMA: halve the step when the batch likelihood stops improving.

This is the classic ``ReduceLROnPlateau`` rule, and it is the *deliberate
counter-example* to the H6 invariant.  The telos states that an adaptive step
must be normalized by a quantity that does not move with the parameters being
optimized; this updater reads ``statistics.meta["nll"]`` -- the fit quality of
the parameters it is about to move -- so it closes exactly the loop H6 forbids.

It exists to be measured, not to be defaulted.  The schedule sweep runs it
against the configuration-only laws: if reading the objective were free, this
would be the strongest arm, because it halves the step precisely where the
current model is stuck.  A measurement is what decides.

The signal lags by one update: ``measure`` scores the batch with the parameters
decoded from the *previous* state, so a halving triggered by batch ``t`` takes
effect on batch ``t+1``.  That is also true of the deep-learning version.

Only the scale of the noise matters -- a relative tolerance of ``1e-4`` in nats
counts as "no improvement", because a mini-batch NLL moves in the third decimal
on its own.  A statistic without an ``nll`` (k-means, NMF, ...) leaves the step
at its current level, so the same object is usable everywhere; the guardrail is
that it never *invents* a plateau from a signal it does not have.
"""

from __future__ import annotations

import numpy as np

from ..core import State, UpdateContext
from .base import StateUpdater


class PlateauEMA(StateUpdater):
    name = "plateau"

    def __init__(
        self,
        alpha0: float = 0.2,
        *,
        factor: float = 0.5,
        patience: int = 2,
        min_alpha: float = 0.02,
        relative_tol: float = 1e-4,
    ) -> None:
        if not 0.0 < alpha0 <= 1.0:
            raise ValueError("alpha0 must lie in (0, 1]")
        if not 0.0 < factor < 1.0:
            raise ValueError("factor must lie in (0, 1)")
        if patience < 0:
            raise ValueError("patience must be non-negative")
        self.alpha0 = float(alpha0)
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_alpha = float(min_alpha)
        self.relative_tol = float(relative_tol)

    def alpha(self, ctx: UpdateContext) -> float:
        """The level in force; the stateful variant is tracked per state."""
        return self.alpha0

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:
        keys = keys or tuple(statistics.values)
        level = float(previous.meta.get("_plateau_alpha", self.alpha0))
        best = previous.meta.get("_plateau_best")
        bad = int(previous.meta.get("_plateau_bad", 0))
        nll = statistics.meta.get("nll")
        if nll is not None and np.isfinite(float(nll)):
            nll = float(nll)
            # ``best * (1 - tol)`` flips direction when ``best`` is negative: a
            # flat NLL of -2000 never looked like a plateau, so the step never
            # halved (measured 2026-11-14; LAION-scale NLLs are negative, so
            # every plateau result there was really a constant alpha).  A
            # tolerance relative to |best| is sign-agnostic.
            if best is None or nll < float(best) - self.relative_tol * abs(float(best)):
                best, bad = nll, 0
            else:
                bad += 1
                if bad > self.patience:
                    level = max(self.min_alpha, level * self.factor)
                    bad = 0

        values = dict(previous.values)
        for key in keys:
            statistic = self._stat(statistics, key)
            if statistic is None:
                continue
            if not self._same_shape(previous, key, statistic):
                values[key] = np.array(statistic, copy=True)
                continue
            values[key] = (1.0 - level) * previous.values[key] + level * statistic

        meta = dict(previous.meta)
        meta["_plateau_alpha"] = level
        if best is not None:
            meta["_plateau_best"] = float(best)
        meta["_plateau_bad"] = int(bad)
        return State(values, previous.step, meta)

    def config(self) -> dict:
        return {
            "name": self.name,
            "alpha0": self.alpha0,
            "factor": self.factor,
            "patience": self.patience,
            "min_alpha": self.min_alpha,
        }
