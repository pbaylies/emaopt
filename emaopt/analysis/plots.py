"""Plot helpers -- objective versus compute, the canonical figure.

Matplotlib is an optional dependency: it is imported inside the function so the
library stays importable without it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence


def plot_curves(
    curves: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    path: str | Path | None = None,
    *,
    xlabel: str = "examples processed",
    ylabel: str = "objective",
    title: str = "",
    log_y: bool = True,
) -> str:
    """Plot ``{label: (examples, objective)}`` and optionally save to ``path``."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("plot_curves requires matplotlib") from exc

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for label, (xs, ys) in curves.items():
        ax.plot(xs, ys, label=label, linewidth=1.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if path is None:
        path = "curves.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def plot_speedups(rows: Sequence[dict[str, Any]], path: str | Path) -> str:
    """Bar chart of speedup at each target."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("plot_speedups requires matplotlib") from exc

    labels = [f"{row['target']:.4g}" for row in rows]
    values = [row["speedup"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(labels, values)
    ax.axhline(1.0, color="black", linewidth=1.0)
    ax.set_xlabel("target objective")
    ax.set_ylabel("speedup (compute ratio)")
    ax.set_title("compute-to-target speedup")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)
