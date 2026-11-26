"""Seed studies: a single run is an anecdote.

``run_seeds`` repeats a configuration over independent seeds and reports the
distribution of the final objective.  Reported claims should carry this spread,
because e-means' own headline result is a *reduced rate of bad local minima*, a
statement about a distribution, not about one run.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..core import FitConfig, FitResult
from ..benchmarks.runners import run


def run_seeds(
    build: Callable[[int], Any],
    data: Any,
    seeds: Iterable[int],
    config: FitConfig | None = None,
    *,
    eval_data: np.ndarray | None = None,
) -> list[FitResult]:
    """Fit one model per seed (the seed is passed both to the model and the sampler)."""
    results = []
    for seed in seeds:
        cfg = config or FitConfig(seed=int(seed))
        if config is not None:
            cfg = FitConfig(**{**config.__dict__, "seed": int(seed)})
        results.append(run(build(int(seed)), data, cfg, eval_data=eval_data))
    return results


def seed_statistics(results: Sequence[FitResult], *, reference: float | None = None
                    ) -> dict[str, Any]:
    """Mean/std/quantiles of the final objective, plus a bad-run rate."""
    objectives = np.array([r.objective for r in results], dtype=float)
    out: dict[str, Any] = {
        "runs": int(objectives.size),
        "mean": float(objectives.mean()),
        "median": float(np.median(objectives)),
        "std": float(objectives.std(ddof=1)) if objectives.size > 1 else 0.0,
        "min": float(objectives.min()),
        "max": float(objectives.max()),
        "q10": float(np.quantile(objectives, 0.1)),
        "q90": float(np.quantile(objectives, 0.9)),
    }
    if reference is not None:
        out["reference"] = float(reference)
        out["worse_than_1pct"] = float(np.mean(objectives > 1.01 * reference))
        out["relative_mean"] = float(objectives.mean() / max(reference, 1e-12))
    return out


def compare_seeds(
    base_results: Sequence[FitResult],
    variant_results: Sequence[FitResult],
) -> dict[str, Any]:
    """Paired comparison of two seed ensembles (same seeds, same data)."""
    a = np.array([r.objective for r in base_results], dtype=float)
    b = np.array([r.objective for r in variant_results], dtype=float)
    if a.size != b.size:
        raise ValueError("seed ensembles must have equal length to be paired")
    diff = b - a
    return {
        "n": int(a.size),
        "mean_base": float(a.mean()),
        "mean_variant": float(b.mean()),
        "mean_difference": float(diff.mean()),
        "variant_wins": int(np.sum(diff < 0)),
        "relative_change": float(diff.mean() / max(abs(a.mean()), 1e-12)),
    }
