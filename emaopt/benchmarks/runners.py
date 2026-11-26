"""Shared runners: one loop, one accounting convention, comparable results.

Every comparison in the library goes through :func:`run`, so a variant cannot
win by counting compute differently: the same sampler, the same evaluation
points and the same ``examples_processed`` accounting are used for all of them.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..core import FitConfig, FitResult, StatisticalOptimizer, fit
from . import metrics


def run(
    model: StatisticalOptimizer,
    data: Any,
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
    track_memory: bool = False,
) -> FitResult:
    """Fit ``model`` with the shared driver, tracking peak memory on request."""
    cfg = config or FitConfig()
    if track_memory and not cfg.track_memory:
        cfg = FitConfig(**{**cfg.__dict__, "track_memory": True})
    x = data.x if hasattr(data, "x") else np.asarray(data, dtype=np.float64)
    return fit(model, x, cfg, eval_data=eval_data)


def run_variants(
    build: Callable[[str], StatisticalOptimizer],
    variants: Sequence[str],
    data: Any,
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
    track_memory: bool = False,
) -> dict[str, FitResult]:
    """Fit several named variants of a model on the same data and budget."""
    out: dict[str, FitResult] = {}
    for name in variants:
        out[name] = run(build(name), data, config, eval_data=eval_data,
                        track_memory=track_memory)
    return out


def compare(runs: Mapping[str, FitResult], target: float | None = None) -> list[dict[str, Any]]:
    """Tabulate objective, compute and (optionally) compute-to-target per run."""
    rows: list[dict[str, Any]] = []
    for name, result in runs.items():
        row: dict[str, Any] = {
            "variant": name,
            "objective": float(result.objective),
            "examples": int(result.examples_processed),
            "updates": int(result.updates),
            "wall_s": float(result.wall_time),
            "cpu_s": float(result.cpu_time),
        }
        if target is not None:
            xs, ys = result.curve()
            row["examples_to_target"] = _to_target(xs, ys, target)
        rows.append(row)
    return rows


def _to_target(xs: np.ndarray, ys: np.ndarray, target: float) -> float:
    from ..analysis.efficiency import compute_to_target

    value = compute_to_target(xs, ys, target)
    return float("nan") if value is None else float(value)


def summarize(result: FitResult, **extra: Any) -> dict[str, Any]:
    row = result.summary()
    row.update(extra)
    return row
