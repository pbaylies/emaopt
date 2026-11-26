"""Measured memory budget and the streaming/batch resolution it drives.

ONLINE_PLAN phase 2.  ``batch_law="auto"`` resolves a batch size from
``(n, k, d)``; the missing fourth input is *available memory*, and it has to be
measured at call time rather than configured, because it degrades from
neighbours and not from the problem.

    budget = MemoryBudget.probe()          # MemAvailable, cgroup headroom
    plan   = resolve_compute(n, d, k, stat_bytes, budget=budget)
    plan.mode, plan.block_size, plan.act_frequency

Under pressure the same call streams; on an idle box it batches; the resolution
is a plain dict so it lands in ``FitResult.extras["compute_plan"]`` for the
record.  ``MemoryBudget(available_bytes=...)`` and the ``meminfo_path`` /
``cgroup_dir`` arguments exist so tests can simulate pressure without setting a
process-wide rlimit (which would also starve the test runner itself).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-9


def _read_meminfo_available(path: str | os.PathLike[str]) -> int | None:
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _read_cgroup_headroom(directory: str | os.PathLike[str]) -> tuple[int | None, str]:
    """``(headroom_bytes, source)`` from a cgroup v2 or v1 directory, if present."""
    base = Path(directory)
    candidates = [
        ("memory.max", "memory.current", "cgroup-v2"),
        ("memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "cgroup-v1"),
    ]
    for limit_name, usage_name, label in candidates:
        limit_path = base / limit_name
        usage_path = base / usage_name
        try:
            limit_text = limit_path.read_text(encoding="ascii").strip()
            usage_text = usage_path.read_text(encoding="ascii").strip()
        except OSError:
            continue
        if limit_text == "max":
            continue
        try:
            limit = int(limit_text)
            usage = int(usage_text)
        except ValueError:  # pragma: no cover - defensive
            continue
        # cgroup v1 reports "unlimited" as a huge page-aligned number.
        if limit >= 1 << 60:
            continue
        return max(0, limit - usage), label
    return None, ""


@dataclass(frozen=True)
class MemoryBudget:
    """Available working memory, measured and overridable."""

    available_bytes: int | None
    source: str
    fraction: float = 0.5
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def limit_bytes(self) -> int | None:
        if self.available_bytes is None:
            return None
        return max(0, int(self.available_bytes * float(self.fraction)))

    @classmethod
    def probe(
        cls,
        *,
        meminfo_path: str | os.PathLike[str] = "/proc/meminfo",
        cgroup_dir: str | os.PathLike[str] | None = "/sys/fs/cgroup",
        fraction: float = 0.5,
        override: int | None = None,
    ) -> "MemoryBudget":
        """Measure memory available right now.

        ``available = min(MemAvailable, cgroup limit - cgroup usage)``.  A
        missing source is not an error -- it is recorded as ``unavailable`` and
        the resolver then stays in batch mode, which is the pre-plan behavior.
        """
        if override is not None:
            return cls(int(override), "override", float(fraction), {"override": int(override)})
        detail: dict[str, Any] = {}
        available = _read_meminfo_available(meminfo_path)
        sources: list[str] = []
        if available is not None:
            detail["meminfo"] = str(meminfo_path)
            detail["memavailable"] = int(available)
            sources.append("meminfo")
        if cgroup_dir is not None:
            headroom, label = _read_cgroup_headroom(cgroup_dir)
            if headroom is not None:
                detail["cgroup"] = label
                detail["cgroup_headroom"] = int(headroom)
                sources.append(label)
                available = headroom if available is None else min(available, headroom)
        if available is None:
            return cls(None, "unavailable", float(fraction), detail)
        return cls(int(available), "+".join(sources), float(fraction), detail)

    def to_dict(self) -> dict[str, Any]:
        return {
            "available_bytes": None if self.available_bytes is None else int(self.available_bytes),
            "limit_bytes": self.limit_bytes,
            "fraction": float(self.fraction),
            "source": self.source,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class ComputePlan:
    """The resolution: mode, block size, act frequency, and why."""

    mode: str
    block_size: int
    act_frequency: int
    available_bytes: int | None
    limit_bytes: int | None
    resident_estimate_bytes: int
    stat_bytes: int
    source: str
    reason: str
    requested_batch: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "block_size": int(self.block_size),
            "act_frequency": int(self.act_frequency),
            "available_bytes": self.available_bytes,
            "limit_bytes": self.limit_bytes,
            "resident_estimate_bytes": int(self.resident_estimate_bytes),
            "stat_bytes": int(self.stat_bytes),
            "source": self.source,
            "reason": self.reason,
            "requested_batch": None if self.requested_batch is None else int(self.requested_batch),
        }


def _round_down_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << int(np.floor(np.log2(float(value))))


def resolve_compute(
    n: int,
    d: int,
    k: int = 0,
    *,
    stat_bytes: int = 0,
    budget: MemoryBudget | None = None,
    supports_streaming: bool = True,
    requested_batch: int | None = None,
    nepochs: int = 1,
    total_steps: int | None = None,
    working_multiplier: float = 2.0,
) -> ComputePlan:
    """Resolve batch-vs-stream, block size and act frequency from measured bytes.

    The batch-mode resident estimate is the float64 data copy plus the stat plus
    a chunk of ``d``-distance working memory -- the arrays a full-pass call
    actually holds.  When that exceeds the budgeted fraction of available
    memory the plan streams: the block is the largest power of two whose
    per-block working set still fits, and parameters are refreshed roughly 256
    times over the horizon (frequently enough for the online laws; less often
    than every block when the horizon is long).
    """
    n = max(1, int(n))
    d = max(1, int(d))
    k = max(0, int(k))
    stat_bytes = max(0, int(stat_bytes))
    if budget is None or budget.limit_bytes is None:
        mode = "batch"
        reason = "no memory budget available"
        if budget is not None and budget.source == "unavailable":
            reason = "no MemAvailable/cgroup source"
        return ComputePlan(mode, n, 1, None, None,
                           int(n * d * 8 + stat_bytes), stat_bytes,
                           "unavailable" if budget is None else budget.source,
                           reason, requested_batch)
    limit = int(budget.limit_bytes)
    chunk = min(n, 4096)
    resident = int(n * d * 8 + stat_bytes + chunk * d * 8 + chunk * k * 8)
    if not supports_streaming:
        return ComputePlan("batch", n, 1, budget.available_bytes, limit, resident,
                           stat_bytes, budget.source,
                           "optimizer has no streaming form", requested_batch)
    if resident <= limit:
        return ComputePlan("batch", n, 1, budget.available_bytes, limit, resident,
                           stat_bytes, budget.source,
                           f"resident estimate {resident} <= budget {limit}", requested_batch)
    per_row = d * 8 * working_multiplier + k * 8 * 2 + 64
    room = max(0, limit - stat_bytes)
    block = int(room // max(per_row, 1))
    block = max(1, min(block, n))
    block = _round_down_power_of_two(block)
    steps = total_steps
    if steps is None:
        steps = max(1, (n // max(block, 1)) * max(1, int(nepochs)))
    act_frequency = max(1, int(round(steps / 256.0)))
    reason = (
        f"resident estimate {resident} > budget {limit} "
        f"({budget.source}={budget.available_bytes})"
    )
    return ComputePlan("stream", int(block), int(act_frequency), budget.available_bytes,
                       limit, resident, stat_bytes, budget.source, reason, requested_batch)


__all__ = ["MemoryBudget", "ComputePlan", "resolve_compute"]
