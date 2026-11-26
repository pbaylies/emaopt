"""Statistics for seed ensembles.

A single run is not evidence.  These helpers report spread and paired
comparisons so a claim like "EMA wins" is a statement about a distribution.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def mean_ci(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Mean with a normal-approximation confidence interval (and the spread)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "lo": float("nan"), "hi": float("nan")}
    mean = float(v.mean())
    if v.size == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "lo": mean, "hi": mean}
    sd = float(v.std(ddof=1))
    se = sd / np.sqrt(v.size)
    z = float(_z_for(confidence))
    return {"n": int(v.size), "mean": mean, "std": sd,
            "lo": mean - z * se, "hi": mean + z * se}


def _z_for(confidence: float) -> float:
    """Inverse normal CDF for the two common confidence levels, no SciPy."""
    table = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.960, 0.99: 2.576}
    return table.get(round(float(confidence), 3), 1.960)


def paired_bootstrap(
    baseline: Sequence[float],
    variant: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap CI for ``mean(variant - baseline)`` on paired seeds.

    Negative values mean the variant has the lower (better) objective.
    """
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(variant, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired comparison requires equal-length arrays")
    if a.size == 0:
        return {"n": 0, "mean_difference": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "p_variant_better": float("nan")}
    diff = b - a
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_boot))
    for i in range(int(n_boot)):
        idx = rng.integers(0, diff.size, size=diff.size)
        means[i] = diff[idx].mean()
    lo, hi = np.quantile(means, [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return {
        "n": int(diff.size),
        "mean_difference": float(diff.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_variant_better": float(np.mean(means < 0.0)),
        "relative_change": float(diff.mean() / max(abs(a.mean()), 1e-12)),
    }


def sign_test(baseline: Sequence[float], variant: Sequence[float]) -> dict[str, Any]:
    """Exact binomial sign test on paired wins (no SciPy)."""
    a = np.asarray(baseline, dtype=float)
    b = np.asarray(variant, dtype=float)
    wins = int(np.sum(b < a))
    losses = int(np.sum(b > a))
    ties = int(a.size - wins - losses)
    n = wins + losses
    if n == 0:
        p = 1.0
    else:
        k = min(wins, losses)
        p = sum(_choose(n, i) for i in range(k + 1)) / (2.0**n)
    return {"wins": wins, "losses": losses, "ties": ties,
            "p_two_sided": float(min(1.0, 2.0 * p))}


def _choose(n: int, k: int) -> float:
    from math import comb

    return float(comb(n, k))


def summarize_objectives(results: Sequence[Any]) -> dict[str, float]:
    """Summary of a list of FitResults (accepts objects with .objective).

    Refuses a mix of objective kinds: an interval over prequential and
    full-data scores together is not an interval of anything (T39).
    """
    from .efficiency import assert_comparable_kinds, objective_kind

    assert_comparable_kinds(*(objective_kind(r) for r in results))
    values = [float(r.objective) for r in results]
    out = mean_ci(values)
    out["min"] = float(np.min(values)) if values else float("nan")
    out["max"] = float(np.max(values)) if values else float("nan")
    return out
