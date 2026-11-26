"""RunningMean: cumulative mean over batches, equally weighted per batch."""

from __future__ import annotations

from typing import Any

from ..core import UpdateContext
from .affine import AffineUpdater


class RunningMean(AffineUpdater):
    """``S <- (n S + s) / (n + 1)`` with ``n`` the number of batches folded in.

    The equal-weight mean is the right law when every batch is drawn from the
    same distribution and the statistic is an unbiased estimate: it is the
    minimum-variance combination of the batch statistics.  It keeps every
    batch's information forever, which is exactly what fails when the
    assignment structure drifts (old assignments computed under poor centroids
    are still in the average).
    """

    name = "running_mean"

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        # ``S <- (n S + s)/(n+1)`` is the blend with ``a = 1/(n+1)``.  A fresh key
        # starts at ``a = 1`` (the reset branch would copy it anyway) so its
        # counter begins at one, as it did when this law owned its own loop.
        counts = dict(meta.get("_n", {}))
        alphas: dict[str, float] = {}
        for key in keys:
            n = 0 if key in fresh else int(counts.get(key, 1))
            alphas[key] = 1.0 / (n + 1)
            counts[key] = n + 1
        meta["_n"] = counts
        return alphas, meta
