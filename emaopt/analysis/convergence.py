"""Convergence analysis: curves, best-so-far envelopes and target crossing."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def curve_arrays(history: Iterable[dict[str, Any]] | Any) -> tuple[np.ndarray, np.ndarray]:
    """Accept a history list or a FitResult/EmeansResult and return (x, y)."""
    history = history.history if hasattr(history, "history") else list(history)
    xs = np.array([h["examples_processed"] for h in history], dtype=float)
    ys = np.array([h["objective"] for h in history], dtype=float)
    return xs, ys


def best_so_far(ys: Sequence[float]) -> np.ndarray:
    """Running minimum: the loss an algorithm would report if it stopped there."""
    ys = np.asarray(ys, dtype=float)
    if ys.size == 0:
        return ys
    return np.minimum.accumulate(ys)


def smooth(ys: Sequence[float], window: int = 3) -> np.ndarray:
    """Centred moving average (odd window), padded at the edges."""
    ys = np.asarray(ys, dtype=float)
    if window <= 1 or ys.size == 0:
        return ys
    window = int(window) | 1
    pad = window // 2
    padded = np.pad(ys, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: ys.size]


def final_gap(ys: Sequence[float], target: float) -> float:
    ys = np.asarray(ys, dtype=float)
    if ys.size == 0:
        return float("nan")
    return float(ys[-1] - target)


def improvement(ys: Sequence[float]) -> float:
    """Relative improvement from the first to the last evaluation."""
    ys = np.asarray(ys, dtype=float)
    if ys.size < 2:
        return 0.0
    return float((ys[0] - ys[-1]) / max(abs(ys[0]), 1e-12))


def targets_from_reference(
    reference_curve: tuple[Sequence[float], Sequence[float]],
    fractions: Sequence[float] = (0.5, 0.75, 0.9, 1.0),
) -> list[float]:
    """Target objectives derived from a baseline's own trajectory.

    ``fraction=1.0`` is the baseline's final objective; smaller fractions are
    easier targets.  Interpolated between the baseline's first and final loss so
    that the target is reachable by construction.
    """
    _, ys = reference_curve
    ys = np.asarray(ys, dtype=float)
    start, end = float(ys[0]), float(ys[-1])
    return [float(start + f * (end - start)) for f in fractions]


def updates_to_target(xs: Sequence[float], ys: Sequence[float], target: float) -> float | None:
    """First x at which the curve is at or below ``target`` (linear interpolation)."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.size == 0:
        return None
    if ys[0] <= target:
        return float(xs[0])
    for i in range(1, ys.size):
        if ys[i] <= target:
            y0, y1 = ys[i - 1], ys[i]
            x0, x1 = xs[i - 1], xs[i]
            if y0 == y1:
                return float(x1)
            frac = (y0 - target) / (y0 - y1)
            return float(x0 + frac * (x1 - x0))
    return None


def epochs_to_target(xs: Sequence[float], ys: Sequence[float], target: float,
                     n: int) -> float | None:
    examples = updates_to_target(xs, ys, target)
    return None if examples is None else float(examples / max(n, 1))
