"""Scaling studies (PLAN stage 4): does the effect survive realistic scale?

Vary ``n``, ``D``, ``K`` and ``B`` on the same problem family and report the
compute-to-target ratio at every scale.  A result that only holds at one size is
an artifact of that size.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..core import FitConfig
from ..benchmarks import datasets
from ..benchmarks.runners import run
from ..analysis.efficiency import compute_to_target, speedup


def scaling_study(
    build: Callable[..., Any],
    scales: Sequence[dict[str, Any]],
    *,
    reference_build: Callable[..., Any] | None = None,
    dataset_name: str = "blobs",
    nepochs: int = 5,
    seed: int = 0,
    config: FitConfig | None = None,
) -> list[dict[str, Any]]:
    """Run one ``build`` (and optionally a reference) per scale configuration.

    Each scale dict carries its own dataset kwargs plus ``k``:
    ``{"n": 2000, "d": 8, "k": 16, "sep": 6.0}``.
    """
    rows: list[dict[str, Any]] = []
    for scale in scales:
        kwargs = dict(scale)
        k = int(kwargs.pop("k"))
        dataset = datasets.make_dataset(dataset_name, seed=seed, **kwargs)
        x = dataset.x
        cfg = config or FitConfig(nepochs=nepochs, seed=seed)
        result = run(build(k=k, d=dataset.d, n=dataset.n), x, cfg)
        row: dict[str, Any] = {
            "n": dataset.n,
            "d": dataset.d,
            "k": k,
            "objective": float(result.objective),
            "examples": int(result.examples_processed),
            "wall_s": float(result.wall_time),
        }
        if reference_build is not None:
            ref = run(reference_build(k=k, d=dataset.d, n=dataset.n), x, cfg)
            xs_ref, ys_ref = ref.curve()
            xs, ys = result.curve()
            target = float(ref.objective)
            t_ref = compute_to_target(xs_ref, ys_ref, target)
            t_var = compute_to_target(xs, ys, target)
            row["reference_objective"] = float(ref.objective)
            row["examples_to_reference_target"] = (
                float("nan") if t_var is None else float(t_var)
            )
            row["speedup_at_reference"] = (
                float("nan") if (t_ref is None or t_var is None) else float(t_ref / t_var)
            )
        rows.append(row)
    return rows


def summarize_scaling(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a scaling table: median objective ratio and coverage."""
    ratios = [r["objective"] / max(r.get("reference_objective", 1.0), 1e-12)
              for r in rows if "reference_objective" in r]
    speedups = [r["speedup_at_reference"] for r in rows
                if "speedup_at_reference" in r and np.isfinite(r["speedup_at_reference"])]
    return {
        "scales": len(rows),
        "median_objective_ratio": float(np.median(ratios)) if ratios else float("nan"),
        "median_speedup": float(np.median(speedups)) if speedups else float("nan"),
        "fraction_wins": float(np.mean([s > 1.0 for s in speedups])) if speedups else float("nan"),
    }
