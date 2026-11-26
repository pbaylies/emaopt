"""Step-size schedules.

A schedule maps :class:`~emaopt.core.UpdateContext` to a step size.  The
context carries progress through the *planned* horizon, which is why a cosine
schedule can be built from shape alone (``n``, ``k``, ``nepochs``) before the
first batch is seen -- the same thing e-means does with its fitted laws.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import UpdateContext


@dataclass
class ConstantSchedule:
    alpha0: float = 0.1

    def alpha(self, ctx: UpdateContext) -> float:
        return float(self.alpha0)


@dataclass
class CosineSchedule:
    """Anneal from ``alpha0`` to ``alpha_end_ratio * alpha0`` along a cosine.

    e-means anneals its step this way (a factor-10 reduction across training),
    the same shape as a learning-rate schedule in deep learning.
    """

    alpha0: float = 0.1
    alpha_end_ratio: float = 0.1
    warmup: int = 0

    def alpha(self, ctx: UpdateContext) -> float:
        if self.warmup and ctx.step < self.warmup:
            return float(self.alpha0) * (ctx.step + 1) / self.warmup
        a0 = float(self.alpha0)
        a1 = a0 * float(self.alpha_end_ratio)
        p = min(1.0, max(0.0, (ctx.step + 1) / max(1, ctx.total_steps)))
        return float(a1 + 0.5 * (a0 - a1) * (1.0 + math.cos(math.pi * p)))

    def alpha_grid(self, total_steps: int, batch_size: int = 0,
                   dataset_size: int = 0) -> np.ndarray:
        """Every step size of the horizon at once, in one NumPy expression.

        Identical to ``[alpha(ctx_t) for t in range(total_steps)]`` -- pinned by
        ``tests/test_schedules_grid.py`` -- but it is what the compiled e-means
        runner needs: asking the scalar form 640 times costs 1.3 ms, which is
        more than the kernel it feeds.  Schedules without a vector form fall
        back to the scalar loop in ``EMeans._alpha_grid``.
        """
        total = max(int(total_steps), 1)
        steps = np.arange(total, dtype=np.float64)
        a0 = float(self.alpha0)
        a1 = a0 * float(self.alpha_end_ratio)
        body = a1 + 0.5 * (a0 - a1) * (
            1.0 + np.cos(np.pi * np.clip((steps + 1.0) / total, 0.0, 1.0)))
        if not self.warmup:
            return body
        warm = max(1, int(self.warmup))
        early = steps < warm
        body[early] = a0 * (steps[early] + 1.0) / warm
        return body


@dataclass
class LinearSchedule:
    alpha0: float = 0.1
    alpha_end_ratio: float = 0.1

    def alpha(self, ctx: UpdateContext) -> float:
        a0 = float(self.alpha0)
        a1 = a0 * float(self.alpha_end_ratio)
        p = min(1.0, max(0.0, (ctx.step + 1) / max(1, ctx.total_steps)))
        return float(a0 + (a1 - a0) * p)


@dataclass
class InverseSqrtSchedule:
    alpha0: float = 0.1
    t0: float = 1.0
    amin: float = 0.0

    def alpha(self, ctx: UpdateContext) -> float:
        return max(float(self.amin), float(self.alpha0) / math.sqrt(self.t0 + ctx.step))


@dataclass
class InversePowerSchedule:
    alpha0: float = 0.1
    gamma: float = 1.0
    t0: float = 1.0
    amin: float = 0.0

    def alpha(self, ctx: UpdateContext) -> float:
        return max(float(self.amin), float(self.alpha0) / (self.t0 + ctx.step) ** self.gamma)


@dataclass
class EffectiveSampleSchedule:
    """``a = b / (b + prior)``: large batches take proportionally larger steps."""

    prior: float = 1.0
    amin: float = 0.0
    amax: float = 1.0

    def alpha(self, ctx: UpdateContext) -> float:
        b = float(ctx.batch_size)
        return float(np.clip(b / (b + self.prior), self.amin, self.amax))


def _anneal_shape(shape: str, alpha0: float, alpha1: float, progress: float) -> float:
    """Interpolate ``alpha0 -> alpha1`` along one of the standard anneal shapes."""
    if shape == "cosine":
        return alpha1 + 0.5 * (alpha0 - alpha1) * (1.0 + math.cos(math.pi * progress))
    if shape == "linear":
        return alpha0 + (alpha1 - alpha0) * progress
    if shape == "geometric":
        return alpha0 * (alpha1 / alpha0) ** progress
    raise ValueError(f"unknown anneal shape {shape!r}; available: cosine, linear, geometric")


@dataclass
class AnnealSchedule:
    """One peak-to-floor anneal: configurable shape and optional warmup.

The three shapes share their endpoints and differ in the middle, so a
comparison across them isolates the *shape* of the anneal rather than its
budget: a cosine sits near the peak for most of the horizon, a geometric decay
sits near the floor, a linear one is between them.  ``warmup_fraction`` spends
the first slice of the horizon rising to the peak, which is the deep-learning
remedy for a state that is still the warm start at step 0 (a full step there
replaces the initialization outright).

This is the family the offline schedule search varies
(measured): every parameter here is knowable from the
planned horizon, so the search is over shape, not over feedback.
"""

    alpha0: float = 0.1
    alpha_end_ratio: float = 0.1
    shape: str = "cosine"
    warmup_fraction: float = 0.0

    def __post_init__(self) -> None:
        _anneal_shape(self.shape, 1.0, 0.5, 0.5)  # validates the name

    def alpha(self, ctx: UpdateContext) -> float:
        a0 = float(self.alpha0)
        a1 = a0 * float(self.alpha_end_ratio)
        total = max(1, int(ctx.total_steps))
        warm = 0
        if self.warmup_fraction > 0.0:
            warm = max(1, int(round(self.warmup_fraction * total)))
            if warm < 2:
                warm = 0
        if warm and ctx.step < warm:
            return float(a0 * (ctx.step + 1) / warm)
        p = min(1.0, max(0.0, (ctx.step - warm) / max(1, total - 1 - warm)))
        return float(_anneal_shape(self.shape, a0, a1, p))


@dataclass
class JitterSchedule:
    """Stochastic annealing: a base schedule times log-normal noise that decays.

    ``sigma_t = sigma * (1 - p)`` so the perturbation is a temperature that
    reaches zero exactly at the end of the horizon -- simulated annealing's
    acceptance noise, applied to the step size instead of to a move.  The draw
    is a function of ``(seed, step)`` only, so a run is reproducible and the
    schedule remains stateless.
    """

    base: Any
    sigma: float = 0.5
    seed: int = 0

    def alpha(self, ctx: UpdateContext) -> float:
        a = float(self.base.alpha(ctx))
        if self.sigma <= 0.0:
            return a
        p = min(1.0, max(0.0, (ctx.step + 1) / max(1, ctx.total_steps)))
        rng = np.random.default_rng([int(self.seed), int(ctx.step)])
        return float(np.clip(a * np.exp(rng.normal(0.0, self.sigma * (1.0 - p))), 0.0, 1.0))


@dataclass
class EMeansStepLaw:
    """The peak step of e-means (arXiv:2611.09501, Eq. 6), cosine-annealed.

    The published law sets the *peak* step from the problem configuration alone::

        alpha0 = min(ac * k^ak * B^aB * n^an * T^aT, 1)

    with the coefficients fitted on 75k DEEP runs (`ac=0.824`, `ak=-0.46`,
    `aB=0.90`, `an=-0.45`, `aT=-0.11`), then anneals it on a cosine to 10% of
    that peak (their Eq. 5).  Two properties matter here beyond the numbers: the
    law depends only on things known before the run -- `k`, batch size, dataset
    size and the epoch budget -- so it closes no feedback loop onto the state it
    damps; and it is *linear in batch size* (`aB` ~ 0.9), which is why a fixed
    alpha cannot serve two batch sizes at once.

    ``k`` has to be passed in because :class:`UpdateContext` carries no
    component count; everything else is read from the context, so the schedule
    works for any fit without extra plumbing.
    """

    k: int
    ac: float = 0.824
    ak: float = -0.46
    aB: float = 0.90
    an: float = -0.45
    aT: float = -0.11
    alpha_end_ratio: float = 0.1
    anneal: str = "cosine"
    warmup_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.anneal not in ("none", "cosine", "linear", "geometric"):
            raise ValueError(
                f"unknown anneal {self.anneal!r}; available: none, cosine, linear, geometric"
            )

    def peak(self, ctx: UpdateContext) -> float:
        b = float(max(ctx.batch_size, 1))
        n = float(max(ctx.dataset_size, 1))
        epochs = max(ctx.total_steps * b / n, 1.0)   # T: updates per epoch
        raw = (
            self.ac
            * float(self.k) ** self.ak
            * b ** self.aB
            * n ** self.an
            * epochs ** self.aT
        )
        return float(min(raw, 1.0))

    def alpha(self, ctx: UpdateContext) -> float:
        a0 = self.peak(ctx)
        total = max(1, int(ctx.total_steps))
        warm = 0
        if self.warmup_fraction > 0.0:
            warm = max(1, int(round(self.warmup_fraction * total)))
            if warm < 2:
                warm = 0
        if warm and ctx.step < warm:
            return float(a0 * (ctx.step + 1) / warm)
        if self.anneal == "none":
            return float(a0)
        a1 = a0 * float(self.alpha_end_ratio)
        p = min(1.0, max(0.0, (ctx.step - warm) / max(1, total - 1 - warm)))
        return float(_anneal_shape(self.anneal, a0, a1, p))


SCHEDULES = {
    "constant": ConstantSchedule,
    "emeans_law": EMeansStepLaw,
    "cosine": CosineSchedule,
    "anneal": AnnealSchedule,
    "jitter": JitterSchedule,
    "linear": LinearSchedule,
    "inverse_sqrt": InverseSqrtSchedule,
    "inverse_power": InversePowerSchedule,
    "effective_sample": EffectiveSampleSchedule,
}


def make_schedule(name: str, **kwargs: Any):
    """Build a schedule by name."""
    if name not in SCHEDULES:
        raise KeyError(f"unknown schedule {name!r}; available: {sorted(SCHEDULES)}")
    return SCHEDULES[name](**kwargs)
