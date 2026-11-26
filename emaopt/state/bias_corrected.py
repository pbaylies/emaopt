"""BiasCorrectedEMA: EMA without the zero-initialization transient."""

from __future__ import annotations

from typing import Any, Mapping

from ..core import UpdateContext
from .ema import EMA


class BiasCorrectedEMA(EMA):
    """EMA without the zero-initialization transient, implemented by warm start.

    An EMA started at zero is biased towards zero: after one update at
    ``a = 0.1`` it reports ``0.1 s`` rather than ``s``.  This class removes
    that transient by *initialising the recursion at the first observed
    statistic* (``raw = s``, mass = 1), so the debias division below is a
    no-op in practice: ``values == raw == warm-started EMA``.  Measured,
    2026-11-14: the ``mass`` factor pins at 1.0 forever; the class is a
    warm-start EMA, not Adam-style correction (pinned by tests in
    ``test_state_updaters.py``).  Since the factor never engages, the second
    full copy of every array it needed (``meta['_raw']``) was removed: the state
    now holds one array per key plus a set of keys already observed.  The name
    is kept for records; :data:`WarmStartEMA` is the honest one.

    Two consequences with teeth:
    * the FIRST update discards a warm initialised state — keys present in
      ``previous.values`` but absent from ``meta['_raw']`` are reset to the
      batch statistic;
    * ``exempt`` is not a safe partial debias for ratio decoders.  Exempting
      one key of a pair (e.g. counts in ``centroid = sums/counts``) mixes
      reset scale (sums) with recursed scale (counts, arriving at ~a*stat
      from a negligible prior), inflating the ratio by ~1/a for ~1/a
      updates: measured k-means parity 1.46, never reaching target at
      alpha <= 0.1.  Debias a ratio's keys together or not at all.
    """

    name = "bias_corrected"

    def __init__(
        self,
        alpha: float | str | Any = 0.1,
        *,
        schedule: Any = None,
        per_key: Mapping[str, Any] | None = None,
        exempt: tuple[str, ...] = (),
    ) -> None:
        super().__init__(alpha, schedule=schedule, per_key=per_key)
        self.exempt = tuple(exempt)

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        seen = set(meta.get("_seen", ()))
        alphas: dict[str, float] = {}
        for key in keys:
            if key in self.exempt:
                alphas[key] = self.alpha(ctx, key)
                continue
            if key in fresh or key not in seen:
                # warm start: the first observation of a key replaces it, which
                # is what removes EMA's zero-initialization transient
                alphas[key] = 1.0
                seen.add(key)
                continue
            alphas[key] = self.alpha(ctx, key)
        meta["_seen"] = seen
        return alphas, meta
        return State(values, previous.step, meta)

    def config(self) -> dict:
        out = super().config()
        out["exempt"] = list(self.exempt)
        return out
