"""Parameter sweeps (PLAN stage 3: schedule sweep).

A sweep is not "run everything": it holds data, initialization, budget and
evaluation fixed and varies one axis, so the resulting curve identifies whether
an effect is robust or a single magic hyperparameter.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..core import FitConfig, FitResult
from ..benchmarks.runners import run


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in product(*(grid[k] for k in keys))]


def sweep(
    build: Callable[..., Any],
    grid: Mapping[str, Sequence[Any]],
    data: Any,
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
    track_memory: bool = False,
) -> list[dict[str, Any]]:
    """Fit one model per grid cell; returns ``{"params", "result"}`` rows."""
    rows: list[dict[str, Any]] = []
    for params in expand_grid(grid):
        model = build(**params)
        result = run(model, data, config, eval_data=eval_data, track_memory=track_memory)
        rows.append({"params": params, "result": result})
    return rows


def alpha_sweep(
    build: Callable[[float], Any],
    alphas: Sequence[float],
    data: Any,
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Sweep a constant EMA step (the classic batch/step hyperparameter study)."""
    rows = []
    for alpha in alphas:
        result = run(build(float(alpha)), data, config, eval_data=eval_data)
        rows.append({"params": {"alpha": float(alpha)}, "result": result})
    return rows


def best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Lowest final objective wins; raises on an empty sweep."""
    if not rows:
        raise ValueError("empty sweep")
    return min(rows, key=lambda row: row["result"].objective)


def table(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten sweep rows into a metrics table (sorted by objective)."""
    out = []
    for row in sorted(rows, key=lambda r: r["result"].objective):
        record = dict(row["params"])
        record.update(
            {
                "objective": float(row["result"].objective),
                "examples": int(row["result"].examples_processed),
                "updates": int(row["result"].updates),
                "wall_s": float(row["result"].wall_time),
            }
        )
        out.append(record)
    return out
