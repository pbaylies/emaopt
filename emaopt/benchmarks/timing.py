"""Timing and memory accounting.

Wall clock alone is not evidence: the metric the PLAN asks for is quality per
unit compute with examples processed, CPU time, GPU time (when present) and peak
memory all recorded so that a speedup cannot be manufactured by a different
definition of "one pass".
"""

from __future__ import annotations

import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import numpy as np


@dataclass
class Timing:
    wall_time: float = 0.0
    cpu_time: float = 0.0
    gpu_time: float = 0.0
    peak_memory: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_time": float(self.wall_time),
            "cpu_time": float(self.cpu_time),
            "gpu_time": float(self.gpu_time),
            "peak_memory": int(self.peak_memory),
        }


@contextmanager
def timer(track_memory: bool = False) -> Iterator[Timing]:
    """Context manager measuring wall/cpu time (and optionally peak memory)."""
    result = Timing()
    if track_memory:
        tracemalloc.start()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    try:
        yield result
    finally:
        result.wall_time = time.perf_counter() - wall0
        result.cpu_time = time.process_time() - cpu0
        if track_memory:
            result.peak_memory = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()


def time_call(fn: Callable[..., Any], *args: Any, track_memory: bool = False,
              **kwargs: Any) -> tuple[Any, Timing]:
    with timer(track_memory=track_memory) as t:
        out = fn(*args, **kwargs)
    return out, t


def throughput(examples: int, timing: Timing) -> float:
    """Examples per second."""
    return float(examples / max(timing.wall_time, 1e-12))


def gpu_time_available() -> bool:
    try:  # pragma: no cover - optional dependency
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def cuda_time(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Measure a call with CUDA events when torch+CUDA are present, else wall time."""
    try:  # pragma: no cover - optional dependency
        import torch
    except Exception:
        out, t = time_call(fn, *args, **kwargs)
        return out, t.wall_time
    if not torch.cuda.is_available():
        out, t = time_call(fn, *args, **kwargs)
        return out, t.wall_time
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    out = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return out, float(start.elapsed_time(end)) / 1000.0


def hardware_descriptor() -> dict[str, Any]:
    """Enough hardware/software identity for an experiment record to be reproducible."""
    import platform
    import sys

    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "cpu_count": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover
        pass
    try:  # pragma: no cover - optional
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    return info
