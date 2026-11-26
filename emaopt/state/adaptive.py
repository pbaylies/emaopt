"""AdaptiveEMA: step size set by the effective sample count of the state.

This is the "effective-sample-count schedule" the PLAN lists as a candidate.
Instead of a fixed decay (which cannot be right for both ``n = 10^4`` and
``n = 10^9``), the state remembers how much evidence it has accumulated::

    n_eff <- (1 - a) n_eff + b        (a batch of b samples)
    a      <- clip(b / (b + n_eff), amin, amax)

A fresh state takes steps of order one, so early exploration and late averaging
come from one parameter-free rule.

What it actually decays like (corrected 2026-11-14): with ``u = n_eff/b`` the
recursion is ``u <- u + 1/(1+u)``, i.e. ``u + u^2/2 = t`` and ``n_eff ~
b*sqrt(2t)``.  The step is therefore **inverse-square-root** in the update count
(measured log-log slope -0.49), not the ``1/t`` the first docstring claimed.
Because ``sum a_t^2`` diverges, this is not a Robbins-Monro averaging schedule:
the state keeps adapting rather than asymptotically averaging.  That is a
property of the recursion, not a bug -- but it means the law should not be
compared against ``RobbinsMonro(gamma=1)`` on equal-step-size grounds alone.
"""

from __future__ import annotations

import math

import numpy as np

from ..core import UpdateContext
from .affine import AffineUpdater


class AdaptiveEMA(AffineUpdater):
    name = "adaptive"

    def __init__(self, *, amin: float = 0.0, amax: float = 1.0,
                 prior_count: float = 1.0, prior_mass: float = 1.0) -> None:
        self.amin = float(amin)
        self.amax = float(amax)
        self.prior_count = float(prior_count)
        self.prior_mass = float(prior_mass)

    def alpha(self, ctx: UpdateContext) -> float:
        """The step a state that has taken every step up to ``ctx.step`` applies.

        The recursion ``n <- (1-a) n + b`` has no elementary closed form, but
        ``u = n/b`` turns it into ``u <- u + 1/(1+u)``, so ``u + u^2/2 = t`` and
        ``n ~ b*sqrt(2t)``.  This used to report ``1/(t+1)`` (a *linear*
        evidence estimate) while ``update`` ran the recursion, so the diagnostic
        disagreed with the law by a square root.  A caller holding a particular
        state should read ``meta['_ess']`` -- this is the fresh-state estimate.
        """
        b = float(max(ctx.batch_size, 1))
        t = float(max(ctx.step, 0))
        n_eff = b * (math.sqrt(1.0 + 2.0 * t) - 1.0) + self.prior_count
        return float(np.clip(b / (b + n_eff), self.amin, self.amax))

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        ess = dict(meta.get("_ess", {}))
        b = float(ctx.batch_size)
        alphas: dict[str, float] = {}
        for key in keys:
            if key in fresh:
                alphas[key] = 1.0
                ess[key] = b * self.prior_mass
                continue
            n_eff = ess.get(key, self.prior_count)
            a = float(np.clip(b / (b + n_eff), self.amin, self.amax))
            alphas[key] = a
            ess[key] = (1.0 - a) * n_eff + b
        meta["_ess"] = ess
        return alphas, meta

    def config(self) -> dict:
        return {
            "name": self.name,
            "amin": self.amin,
            "amax": self.amax,
            "prior_count": self.prior_count,
        }
