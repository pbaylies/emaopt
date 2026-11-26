"""Precision/speed levels: one knob from reference to performance (T26).

The library's default is the reference path -- NumPy everywhere, float64
throughout.  the measurements measured what a faster setting costs, so the choice can be a
named level instead of a bag of dtype flags:

    level        host E-step   eGMM E-step   eGMM statistics   Gramian
    -----------  ------------  ------------  ----------------  --------
    quality      NumPy         float64       float64           float64
    auto         compiled      float64       float32           float32
    performance  compiled      float32       float32           float32

Acceptance, taken from measurements rather than asserted:

    quality       indistinguishable from the NumPy reference (<=1e-12 nats)
    auto          <=1e-6 nats endpoint drift (measured 2.9e-07);
                  1.1-1.8x per step depending on the process
    performance   <=1e-4 nats -- 100x below the f16 source data's own quantum --
                  (measured 3.9e-05); 2.2-4.5x per step

The drift budgets are the contract: they reproduce exactly across runs because
they come from the arithmetic.  The speedups are indicative, because float64 GEMM
throughput moves with clocks and per-process state -- the same pair of arms
measured 1.15x in one process and 1.69x in another, which is why the levels are
sold on accuracy and only described on speed.

``auto`` is the level a user picks when they want the speed without thinking
about dtypes: it moves only work whose error is a reduction's, never the
algorithm's.  ``performance`` moves the E-step itself, so it is endpoint-grade by
construction: still 100x finer than the quantisation of the data this library is
usually pointed at, but no longer indistinguishable from the reference.

What no level changes: the fold and decode stay float64, the batch stream is the
library's, refusals stay refusals (f16, TF32, non-affine updaters), and nothing
here touches the default path unless a caller asks for a level.

Usage::

    from emaopt import GaussianMixture, levels

    with levels.use("auto"):                 # host fits: compiled E-step on
        result = GaussianMixture(16, covariance="diag").fit(x)

    result = egmm_gpu_fit(model, x, cfg, level="performance")   # device loop

    spec = levels.resolve_level("auto")
    spec.egmm_torch_dtype()      # torch.float64, imported lazily
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

__all__ = ["Level", "LEVELS", "resolve_level", "use", "describe"]

_HOST_KERNEL_ENV = "EMAOPT_HOST_KERNEL"


@dataclass(frozen=True)
class Level:
    """One point on the precision/speed spectrum.

    Dtypes are stored as names so this module imports without torch (the
    library's default path is NumPy-only); :meth:`egmm_torch_dtype` resolves
    them on demand.
    """

    name: str
    host_kernel: bool
    egmm_dtype: str
    egmm_stat_dtype: str
    gramian_dtype: str
    drift_budget: float
    expected_step_speedup: float
    summary: str

    def egmm_torch_dtype(self):
        return _torch_dtype(self.egmm_dtype)

    def stat_torch_dtype(self):
        return _torch_dtype(self.egmm_stat_dtype)


def _torch_dtype(name: str):
    import torch                                   # lazy: NumPy-only default path

    try:
        return {"float64": torch.float64, "float32": torch.float32}[name]
    except KeyError:                               # pragma: no cover - internal
        raise ValueError(f"level dtype {name!r} has no torch equivalent") from None


LEVELS: dict[str, Level] = {
    "quality": Level(
        name="quality",
        host_kernel=False,
        egmm_dtype="float64",
        egmm_stat_dtype="float64",
        gramian_dtype="float64",
        drift_budget=1e-12,
        expected_step_speedup=1.0,
        summary="NumPy reference path: the slowest and the only setting that is "
                "indistinguishable from the host arm (<=1e-12 nats).",
    ),
    "auto": Level(
        name="auto",
        host_kernel=True,
        egmm_dtype="float64",
        egmm_stat_dtype="float32",
        gramian_dtype="float32",
        drift_budget=1e-6,
        expected_step_speedup=1.1,
        summary="Moves only the reductions: f32 statistics and Gramian, f64 "
                "E-step and fold. 1.1-1.8x the step for 2.9e-07 nats.",
    ),
    "performance": Level(
        name="performance",
        host_kernel=True,
        egmm_dtype="float32",
        egmm_stat_dtype="float32",
        gramian_dtype="float32",
        drift_budget=1e-4,
        expected_step_speedup=2.2,
        summary="float32 E-step too: 2.2-4.5x the step for 3.9e-05 nats, 100x "
                "finer than the f16 source data's quantum. Endpoint-grade.",
    ),
}

_ALIASES = {"reference": "quality", "balanced": "auto", "fast": "performance",
            "default": "quality"}


def resolve_level(level: str | None = None) -> Level:
    """The :class:`Level` named ``level`` (``None`` -> ``quality``)."""
    if isinstance(level, Level):
        return level
    name = "quality" if level is None else str(level).strip().lower()
    name = _ALIASES.get(name, name)
    if name not in LEVELS:
        raise ValueError(
            f"unknown level {level!r}; choose from {sorted(LEVELS)} "
            f"(aliases: {sorted(_ALIASES)})")
    return LEVELS[name]


@contextlib.contextmanager
def use(level: str):
    """Run a block at ``level`` for the host-kernel choice, then restore it.

    Only the host E-step is process-global (an environment switch, because it is
    reached from ``GaussianMixture`` internals); the device dtypes travel as
    arguments, so pass ``level=`` to the GPU entry points directly.  The previous
    setting is restored even on an exception.
    """
    spec = resolve_level(level)
    previous = os.environ.get(_HOST_KERNEL_ENV)
    if spec.host_kernel:
        os.environ.pop(_HOST_KERNEL_ENV, None)
    else:
        os.environ[_HOST_KERNEL_ENV] = "numpy"
    try:
        yield spec
    finally:
        if previous is None:
            os.environ.pop(_HOST_KERNEL_ENV, None)
        else:
            os.environ[_HOST_KERNEL_ENV] = previous


def describe() -> list[dict]:
    """The spectrum as rows, for docs, CLIs and evidence payloads."""
    return [
        {
            "level": spec.name,
            "host_kernel": "compiled" if spec.host_kernel else "numpy",
            "egmm_dtype": spec.egmm_dtype,
            "egmm_stat_dtype": spec.egmm_stat_dtype,
            "gramian_dtype": spec.gramian_dtype,
            "drift_budget_nats": spec.drift_budget,
            "expected_step_speedup": spec.expected_step_speedup,
            "summary": spec.summary,
        }
        for spec in LEVELS.values()
    ]
