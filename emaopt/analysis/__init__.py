"""Analysis: convergence, efficiency, statistics and plots."""

from . import convergence, efficiency, plots, statistics
from .convergence import (
    best_so_far,
    curve_arrays,
    epochs_to_target,
    improvement,
    smooth,
    targets_from_reference,
    updates_to_target,
)
from .efficiency import (
    compute_to_target,
    efficiency_report,
    quality_at_compute,
    speedup,
    speedup_curve,
)
from .statistics import mean_ci, paired_bootstrap, sign_test, summarize_objectives

__all__ = [
    "curve_arrays",
    "best_so_far",
    "smooth",
    "improvement",
    "targets_from_reference",
    "updates_to_target",
    "epochs_to_target",
    "compute_to_target",
    "speedup",
    "speedup_curve",
    "quality_at_compute",
    "efficiency_report",
    "mean_ci",
    "paired_bootstrap",
    "sign_test",
    "summarize_objectives",
    "convergence",
    "efficiency",
    "statistics",
    "plots",
]
