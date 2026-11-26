"""EMA: exponentially forget old statistics at rate ``alpha``."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import UpdateContext
from .affine import AffineUpdater
from .schedules import ConstantSchedule, make_schedule


class EMA(AffineUpdater):
    """``S <- (1 - a_t) S + a_t s``.

    ``alpha`` may be a constant, a :class:`~emaopt.state.schedules.Schedule`
    object, or a name in ``SCHEDULES`` ("constant", "cosine", "linear",
    "inverse_sqrt", "inverse_power", "effective_sample").  ``per_key`` applies a
    different law to individual state keys (the PLAN's "separate schedules for
    different state variables").  Keys not yet present are initialized from the
    batch statistic, so a fresh state is immediately meaningful.
    """

    name = "ema"

    def __init__(
        self,
        alpha: float | str | Any = 0.1,
        *,
        schedule: Any = None,
        per_key: Mapping[str, Any] | None = None,
    ) -> None:
        if schedule is not None:
            alpha = schedule
        if isinstance(alpha, str):
            alpha = make_schedule(alpha)
        if isinstance(alpha, (int, float)):
            alpha = ConstantSchedule(float(alpha))
        self.schedule = alpha
        self.per_key = dict(per_key or {})

    def alpha(self, ctx: UpdateContext, key: str | None = None) -> float:
        if key is not None and key in self.per_key:
            law = self.per_key[key]
            if isinstance(law, (int, float)):
                return float(law)
            if isinstance(law, str):
                law = make_schedule(law)
            return float(law.alpha(ctx))
        return float(self.schedule.alpha(ctx))

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Per-key step size from the schedule (``per_key`` overrides included)."""
        return {key: self.alpha(ctx, key) for key in keys}, meta

    def config(self) -> dict:
        schedule = self.schedule
        out = {
            "name": self.name,
            "schedule": type(schedule).__name__,
            "alpha0": getattr(schedule, "alpha0", None),
        }
        if self.per_key:
            out["per_key"] = {
                k: (type(v).__name__ if not isinstance(v, (int, float)) else float(v))
                for k, v in self.per_key.items()
            }
        return out
