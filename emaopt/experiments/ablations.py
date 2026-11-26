"""Ablations: which component of the stateful algorithm is doing the work?

e-means is a *combination* of EMA updates, fitted batch/step laws, cosine
annealing, dead-cluster revival and a final exact-mean pass.  A result that only
reports "the combination wins" cannot say why.  These ablations remove one
component at a time on the same seed and budget.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core import FitConfig
from ..algorithms.kmeans import EMeans
from ..benchmarks.runners import run
from ..state import EMA, Replace


def _variant(name: str, data: np.ndarray, k: int, nepochs: int, seed: int,
             config: FitConfig | None) -> Any:
    if name == "full":
        model = EMeans(k, nepochs=nepochs, seed=seed)
    elif name == "no_revival":
        model = EMeans(k, nepochs=nepochs, seed=seed, revival=False)
    elif name == "no_finalize":
        model = EMeans(k, nepochs=nepochs, seed=seed, finalize=False)
    elif name == "constant_step":
        model = EMeans(k, nepochs=nepochs, seed=seed)
        cfg_alpha = None  # schedule replaced after begin_fit
        model.alpha_end_ratio = 1.0
    elif name == "replace_minibatch":
        model = EMeans(k, nepochs=nepochs, seed=seed, revival=False, finalize=False)
        model.updater = Replace()
        model._force_updater = True
    elif name == "weighted_mean":
        model = EMeans(k, nepochs=nepochs, seed=seed, revival=False, finalize=False)
        from ..state import WeightedMean

        model.updater = WeightedMean("batch_size")
        model._force_updater = True
    else:
        raise KeyError(f"unknown emeans ablation {name!r}")
    return model


def emeans_ablation(
    data: np.ndarray,
    k: int,
    *,
    nepochs: int = 5,
    seed: int = 0,
    config: FitConfig | None = None,
    variants: Sequence[str] = ("full", "no_revival", "no_finalize", "constant_step",
                               "replace_minibatch", "weighted_mean"),
) -> list[dict[str, Any]]:
    """Run each ablation arm and return ``{"variant", "result"}`` rows."""
    data = np.asarray(data, dtype=np.float64)
    cfg = config or FitConfig(nepochs=nepochs, seed=seed)
    rows = []
    for name in variants:
        model = _variant(name, data, k, nepochs, seed, cfg)
        if getattr(model, "_force_updater", False):
            # replace the schedule installed by begin_fit with the requested law
            original_begin = model.begin_fit

            def begin_fit(state, config, n, batch_size, total_steps, _m=model):
                original_begin(state, config, n, batch_size, total_steps)
                if getattr(_m, "_force_updater", False):
                    _m.updater = _m._forced

            model._forced = model.updater
            model.begin_fit = begin_fit  # type: ignore[method-assign]
        result = run(model, data, cfg)
        rows.append({"variant": name, "result": result})
    return rows


def deltas(rows: Sequence[dict[str, Any]], reference: str = "full") -> list[dict[str, Any]]:
    """Relative objective change of each arm against the reference arm."""
    by_name = {row["variant"]: row["result"] for row in rows}
    if reference not in by_name:
        raise KeyError(f"reference arm {reference!r} not in rows")
    base = by_name[reference]
    out = []
    for name, result in by_name.items():
        out.append(
            {
                "variant": name,
                "objective": float(result.objective),
                "relative_to_reference": float(
                    (result.objective - base.objective) / max(abs(base.objective), 1e-12)
                ),
                "examples": int(result.examples_processed),
                "updates": int(result.updates),
            }
        )
    return out


def updater_ablation(
    build,
    updaters: dict[str, Any],
    data: np.ndarray,
    config: FitConfig | None = None,
) -> list[dict[str, Any]]:
    """Same algorithm, different state-update laws (the library's core comparison)."""
    rows = []
    for name, updater in updaters.items():
        model = build(updater)
        result = run(model, data, config)
        rows.append({"variant": name, "result": result})
    return rows
