"""WeightedMean: cumulative mean with arbitrary per-batch weights.

The weight has to match what the statistic *is*.  If a batch statistic is a
per-batch **mean**, ``weight="batch_size"`` is MacQueen's incremental update
(``S += s; N += n``) in mean form -- the classic mini-batch k-means rule
(Sculley 2010 / MacQueen 1967).  If it is a **sum** (which is what this library's
states hold: ``sums``/``counts``, Gramians, word-topic counts), the batch size is
already inside the statistic and weighting by it counts the size twice: a full
batch of 100 followed by a short batch of 10 gives the short batch mass 10/110
of its *sum*, i.e. 0.0099 of its *mean* instead of 0.0909.  Short trailing
batches are routine (``drop_last=False``), so the default is now
``weight="unit"``, which is exactly MacQueen on sums.  Pass
``weight="batch_size"`` only for mean-valued statistics.
"""

from __future__ import annotations

from typing import Any

from ..core import UpdateContext
from .affine import AffineUpdater


class WeightedMean(AffineUpdater):
    name = "weighted_mean"

    def __init__(self, weight: str | float = "unit") -> None:
        self.weight = weight

    def batch_weight(self, ctx: UpdateContext) -> float:
        if isinstance(self.weight, str):
            if self.weight == "batch_size":
                return float(ctx.batch_size)
            if self.weight == "unit":
                return 1.0
            raise ValueError(f"unknown weight rule {self.weight!r}")
        return float(self.weight)

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        # ``(W S + w s)/(W + w)`` is the blend with ``a = w/(W+w)``
        total = dict(meta.get("_w", {}))
        w = self.batch_weight(ctx)
        alphas: dict[str, float] = {}
        for key in keys:
            if key in fresh:
                alphas[key] = 1.0
                total[key] = w
                continue
            w_old = total.get(key, w)
            alphas[key] = w / (w_old + w)
            total[key] = w_old + w
        meta["_w"] = total
        return alphas, meta

    def config(self) -> dict:
        return {"name": self.name, "weight": self.weight}
