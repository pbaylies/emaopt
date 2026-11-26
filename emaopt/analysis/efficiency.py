"""Efficiency: the primary metric is compute-to-target-quality.

    speedup(L*) = T_baseline(L*) / T_variant(L*)

where ``T`` is the compute (examples processed, or wall time) needed to first
reach objective ``L*``.  Reporting epochs instead would let a variant win by
taking different-sized steps, which is exactly the confusion the PLAN forbids.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .convergence import best_so_far


def objective_kind(result: Any) -> str | None:
    """What a fit's ``objective`` actually scored (T39).

    ``extras["objective_kind"]`` is what the drivers record; a bare dict is
    accepted so a serialised record can be checked too.
    """
    extras = getattr(result, "extras", None)
    if isinstance(extras, Mapping) and "objective_kind" in extras:
        return extras["objective_kind"]
    if isinstance(result, Mapping):
        return result.get("objective_kind")
    return getattr(result, "objective_kind", None)


def assert_comparable_kinds(*kinds: str | None) -> str:
    """Return the shared kind, or refuse to let two kinds be ratioed.

    ``eval_mode="stream"`` records a prequential score under the same
    ``objective`` key that host ``fit`` uses for a full-data NLL.  A speedup or
    an interval computed across those is a ratio of two different quantities, so
    it fails loudly here instead of looking like evidence.
    """
    present = sorted({str(k) for k in kinds if k is not None})
    if len(present) > 1:
        raise ValueError(
            f"objective kinds {present} are not comparable: a prequential score "
            "and a full-data/held-out NLL measure different things. Compare "
            "arms at one kind (T39)."
        )
    return present[0] if present else "unknown"


def compute_to_target(
    examples: Sequence[float],
    objectives: Sequence[float],
    target: float,
    *,
    mode: str = "examples",
) -> float | None:
    """Compute needed to first reach ``target``, interpolating between evaluations.

    Uses the best-so-far envelope so a noisy mid-run spike cannot make a curve
    look like it regressed (an algorithm's reported loss is its best so far).
    """
    xs = np.asarray(examples, dtype=float)
    ys = np.asarray(objectives, dtype=float)
    if xs.size == 0 or ys.size != xs.size:
        return None
    ys = best_so_far(ys)
    if not np.isfinite(target):
        return None
    if ys[0] <= target:
        return float(xs[0])
    for i in range(1, ys.size):
        if ys[i] <= target:
            y0, y1 = ys[i - 1], ys[i]
            x0, x1 = xs[i - 1], xs[i]
            if y0 <= y1:  # no progress in this segment; take the evaluation point
                return float(x1)
            frac = (y0 - target) / max(y0 - y1, 1e-300)
            return float(x0 + np.clip(frac, 0.0, 1.0) * (x1 - x0))
    return None


def speedup(
    baseline: tuple[Sequence[float], Sequence[float]],
    variant: tuple[Sequence[float], Sequence[float]],
    target: float,
) -> float:
    """``T_baseline / T_variant`` at ``target``; ``nan`` if either never reaches it."""
    t_base = compute_to_target(*baseline, target)
    t_var = compute_to_target(*variant, target)
    if t_base is None or t_var is None or t_var <= 0:
        return float("nan")
    return float(t_base / t_var)


def speedup_curve(
    baseline: tuple[Sequence[float], Sequence[float]],
    variant: tuple[Sequence[float], Sequence[float]],
    targets: Sequence[float],
) -> list[dict[str, Any]]:
    """Speedup at several targets (the compute/quality frontier, not one point)."""
    rows = []
    for target in targets:
        t_base = compute_to_target(*baseline, float(target))
        t_var = compute_to_target(*variant, float(target))
        rows.append(
            {
                "target": float(target),
                "baseline_compute": t_base,
                "variant_compute": t_var,
                "speedup": (
                    float("nan")
                    if (t_base is None or t_var is None or t_var <= 0)
                    else float(t_base / t_var)
                ),
            }
        )
    return rows


def quality_at_compute(
    examples: Sequence[float],
    objectives: Sequence[float],
    budget: float,
) -> float:
    """Interpolated objective at a fixed compute budget (the dual view)."""
    xs = np.asarray(examples, dtype=float)
    ys = best_so_far(objectives)
    if xs.size == 0:
        return float("nan")
    if budget <= xs[0]:
        return float(ys[0])
    if budget >= xs[-1]:
        return float(ys[-1])
    idx = int(np.searchsorted(xs, budget))
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    if x1 == x0:
        return float(y0)
    return float(y0 + (budget - x0) / (x1 - x0) * (y1 - y0))


def efficiency_report(
    baseline: tuple[Sequence[float], Sequence[float]],
    variants: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    targets: Sequence[float],
) -> dict[str, Any]:
    """Per-variant speedup rows plus a count of targets reached at all."""
    out: dict[str, Any] = {"targets": [float(t) for t in targets], "variants": {}}
    for name, curve in variants.items():
        rows = speedup_curve(baseline, curve, targets)
        reached = [r for r in rows if r["variant_compute"] is not None]
        finite = [r["speedup"] for r in reached if np.isfinite(r["speedup"])]
        out["variants"][name] = {
            "rows": rows,
            "targets_reached": len(reached),
            "median_speedup": float(np.median(finite)) if finite else float("nan"),
            "max_speedup": float(np.max(finite)) if finite else float("nan"),
        }
    return out


def compute_ratio(base_compute: float | None, var_compute: float | None) -> float:
    if base_compute is None or var_compute is None or var_compute <= 0:
        return float("nan")
    return float(base_compute / var_compute)
