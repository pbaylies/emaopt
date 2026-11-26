"""The affine fold: one implementation for every blend-shaped law.

Seven of the ten updaters are a per-key affine blend,

    S <- (1 - a) S + a s

and that algebra had been written three times: inside each updater's ``update``,
inside ``SufficientStat._fold`` (for ``mode="ema"``), and inside
``egmm_gpu_fit``'s ``_fold_alphas``.  The copies drifted: the device path
rejected three laws that are affine (RobbinsMonro, WeightedMean,
BiasCorrectedEMA) purely because nobody had taught the dispatch about them, and
the shape-mismatch reset branch was pasted into eight files.

A law implements :meth:`AffineUpdater.blend` and nothing else:

    alphas, meta = updater.blend(ctx, keys, dict(previous.meta))
    values[key] <- (1 - alphas[key]) * previous.values[key] + alphas[key] * s

Why ``blend`` returns alphas rather than the new state
------------------------------------------------------
The device loop applies the same fold to tensors, so it needs the *number*, not
a NumPy result.  ``egmm_gpu_fit`` therefore calls ``blend`` with the recursion
metadata it carries in its own ``aux`` dict, which is how a law becomes
available on the GPU path without being re-derived there.

``init_mass`` and the initialization confound
---------------------------------------------
Each law used to treat a warm-start state differently on its very first update:
EMA keeps ``1-a`` of it, RunningMean counts it as one batch, WeightedMean gives
it the first batch's weight, RobbinsMonro and BiasCorrectedEMA discard it, and
AdaptiveEMA nearly discards it (``prior_count=1``).  Early-step differences
between laws therefore partly measured *initialization mass* rather than memory
policy.  ``init_mass`` makes that explicit and equalizable: the first update of a
key is forced to ``a = 1/(1+init_mass)``, i.e. the prior state carries
``init_mass`` batches of weight.  ``0`` discards the warm start, ``1`` counts it
as one batch, and ``(1-a)/a`` reproduces EMA's own ``(1-a) S + a s``.

``None`` (the default) keeps each law's historical first step, because forcing
one value would silently move every recorded result; a cross-law comparison
should set it explicitly and hold it fixed.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ..core import State, UpdateContext
from .base import StateUpdater

__all__ = ["AffineUpdater"]


class AffineUpdater(StateUpdater):
    """A law whose update is a per-key affine blend of the state and statistic."""

    #: Batches of weight credited to an incoming warm-start state on the first
    #: update; ``None`` keeps the law's own first step.  See the module docstring.
    init_mass: float | None = None

    def blend(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """Per-key step sizes, plus the recursion metadata to carry forward.

        ``fresh`` are the keys with no previous value of the right shape: they
        take the reset branch, and a law whose step depends on a per-key counter
        must start that counter at one for them.
        """
        raise NotImplementedError  # pragma: no cover - abstract

    def first_step_alpha(self) -> float:
        """``a`` implied by ``init_mass`` on the first update of a key."""
        mass = 0.0 if self.init_mass is None else float(self.init_mass)
        return 1.0 / (1.0 + max(0.0, mass))

    def alphas_for(
        self,
        ctx: UpdateContext,
        keys: tuple[str, ...],
        meta: dict[str, Any],
        fresh: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """``blend`` plus the ``init_mass`` override on the first update.

        Everything that folds a statistic asks here for its step sizes, so a warm
        start is weighted the same way whether the fold runs in an updater, in a
        ``SufficientStat`` or on the device.  ``init_mass`` keys off ``ctx.step``
        (the caller's own step counter), not off the meta a law happens to keep.
        """
        alphas, meta = self.blend(ctx, keys, meta, fresh=fresh)
        if self.init_mass is not None and int(ctx.step) == 0:
            alphas = {key: self.first_step_alpha() for key in alphas}
        return alphas, meta

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:
        keys = keys or tuple(statistics.values)
        fresh = frozenset(
            key for key in keys
            if (statistic := self._stat(statistics, key)) is not None
            and not self._same_shape(previous, key, statistic)
        )
        alphas, meta = self.alphas_for(ctx, keys, dict(previous.meta), fresh)
        values = dict(previous.values)
        for key in keys:
            statistic = self._stat(statistics, key)
            if statistic is None:
                continue
            if not self._same_shape(previous, key, statistic):
                # The single reset branch: a fresh or reshaped key starts at the
                # batch statistic.  It used to be pasted into eight updaters.
                values[key] = np.array(statistic, copy=True)
                continue
            a = float(alphas.get(key, 1.0))
            values[key] = (1.0 - a) * previous.values[key] + a * statistic
        return State(values, previous.step, meta)

    def config(self) -> dict[str, Any]:
        out = super().config()
        if self.init_mass is not None:
            out["init_mass"] = float(self.init_mass)
        return out
