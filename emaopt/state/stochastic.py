"""RobbinsMonro: EMA with a vanishing step size.

``a_t = a0 / (t0 + t) ** gamma``.  With ``gamma = 1`` this is the classical
stochastic-approximation schedule (and asymptotically a running mean); with
``gamma = 0.5`` it is the ``1/sqrt(t)`` rate used by most SGD-style estimators.
It interpolates between "trust the newest batch" and "keep everything", which
is what makes it a useful third point in an ablation.
"""

from __future__ import annotations

from typing import Any

from ..core import UpdateContext
from .affine import AffineUpdater


class RobbinsMonro(AffineUpdater):
    name = "robbins_monro"

    def __init__(self, alpha0: float = 1.0, *, gamma: float = 1.0, t0: float = 1.0,
                 amin: float = 1e-8) -> None:
        self.alpha0 = float(alpha0)
        self.gamma = float(gamma)
        self.t0 = float(t0)
        self.amin = float(amin)

    def alpha(self, ctx: UpdateContext) -> float:
        return max(self.amin, self.alpha0 / (self.t0 + ctx.step) ** self.gamma)

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        a = self.alpha(ctx)
        return {key: a for key in keys}, meta

    def config(self) -> dict:
        return {
            "name": self.name,
            "alpha0": self.alpha0,
            "gamma": self.gamma,
            "t0": self.t0,
        }
