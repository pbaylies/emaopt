"""MomentumEMA: the EMA step carried forward as a velocity (second-order memory).

An EMA is a first-order filter: it remembers one number per statistic.  A
momentum filter remembers the *direction* the statistic has been moving as well,
so a quantity that is still travelling keeps travelling for a few updates after
the batch statistic stops moving it::

    u_t = a (s_t - S_{t-1})       # the step EMA would take
    V_t = beta V_{t-1} + u_t      # that step, carried across updates
    S_t = S_{t-1} + V_t

``beta = 0`` is the ``EMA(alpha=a)`` law (to within floating-point association)
and a test pins that equality.  The
reason this is not the default anywhere is measurable, not stylistic: the
statistics being filtered are not coordinates, they are *sufficient statistics*
of a monotone M-step.  Overshooting a count makes it small (or negative), and a
component whose count falls stops being updated -- the same failure the per-key
schedule experiment found, amplified.  Overshooting a second moment can leave the
positive-semidefinite cone once the decode forms a covariance.

The clamp counters in ``meta["_momentum_clamped"]`` exist so that failure is
visible in a record rather than inferred: they count how often a step had to be
pulled back to keep a non-negative statistic non-negative.
"""

from __future__ import annotations

import numpy as np

from ..core import State, UpdateContext
from .base import StateUpdater

# Statistics that are counts or sums of non-negative quantities.  Momentum may
# push any of them below zero, which is not a smaller value: a negative count
# inverted by a decode is a negative weight.
_NONNEGATIVE_KEYS = frozenset({"counts", "sums", "sq", "aux"})


class MomentumEMA(StateUpdater):
    name = "momentum"

    def __init__(self, alpha: float = 0.1, *, beta: float = 0.5,
                 nonneg: bool = True) -> None:
        # ``self.base_alpha``, never ``self.alpha``: an instance attribute named
        # ``alpha`` shadowed the ``alpha(ctx)`` method every updater shares, so
        # ``updater.alpha(ctx)`` raised TypeError -- which is why ``alpha_at``
        # existed as a workaround.
        self.base_alpha = float(alpha)
        self.beta = float(beta)
        if not 0.0 <= self.beta < 1.0:
            raise ValueError("beta must lie in [0, 1)")
        self.nonneg = bool(nonneg)

    def alpha(self, ctx: UpdateContext) -> float:
        return self.base_alpha

    def alpha_at(self, ctx: UpdateContext) -> float:
        """Backwards-compatible alias for :meth:`alpha`."""
        return self.base_alpha

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:
        keys = keys or tuple(statistics.values)
        a = float(self.alpha_at(ctx))
        values = dict(previous.values)
        velocity = dict(previous.meta.get("_venv", {}))
        clamped = dict(previous.meta.get("_momentum_clamped", {}))
        b = float(ctx.batch_size)
        for key in keys:
            statistic = self._stat(statistics, key)
            if statistic is None:
                continue
            if not self._same_shape(previous, key, statistic):
                values[key] = np.array(statistic, copy=True)
                velocity[key] = np.zeros_like(values[key], dtype=np.float64)
                continue
            prev = np.asarray(previous.values[key], dtype=np.float64)
            vel = velocity.get(key)
            if vel is None or np.shape(vel) != np.shape(prev):
                vel = np.zeros_like(prev)
            step = a * (np.asarray(statistic, dtype=np.float64) - prev)
            vel = self.beta * vel + step
            new = prev + vel
            if self.nonneg and key in _NONNEGATIVE_KEYS:
                bad = int(np.count_nonzero(new < 0.0))
                if bad:
                    clamped[key] = int(clamped.get(key, 0)) + bad
                    new = np.maximum(new, 0.0)
                    # The velocity that drove the statistic through zero cannot
                    # be kept: it points out of the feasible set and would do it
                    # again next update.
                    vel = np.where(vel < 0.0, 0.0, vel)
            values[key] = new
            velocity[key] = vel
        meta = dict(previous.meta)
        meta["_venv"] = velocity
        if clamped:
            meta["_momentum_clamped"] = clamped
        return State(values, previous.step, meta)

    def config(self) -> dict:
        return {
            "name": self.name,
            "alpha": self.base_alpha,
            "beta": self.beta,
            "nonneg": self.nonneg,
        }
