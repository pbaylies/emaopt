"""Thread and work-division policy for stat reductions.

``partitioned_stat`` is a reduction, so the only question is how to cut the
rows and how many threads to spend.  Both are decided here from measured
inputs rather than hard-coded at the call site:

* the number of usable CPUs (affinity), and the number of physical cores
  (SMT siblings do not help a GEMM-bound fold, and on a hybrid CPU they are
  the slower E-core threads);
* a floor of rows per worker below which thread startup plus the merge cost
  more than the parallelism returns;
* a target block size so a worker does not hold a whole partition in float64
  when the partition is large.

The defaults are the ones measured on a
24-logical/16-physical-core box: 16 workers, one BLAS thread for the whole pool
when there is more than one worker, and 8 partitions per worker above 16k rows
per worker (4 below).  Everything is overridable per call (``workers=``,
``max_threads=``, ``n_partitions=``, ``block_size=``, ``blas_threads=``) and
globally through ``EMAOPT_MAX_THREADS``, so a scheduler assignment or a
container CPU quota is respected without editing code.

``blas_threads="auto"`` pins BLAS to one thread *only* when more than one
worker is active: without the pin every worker's GEMM claims all cores and
they contend for them (measured: the centroid fold scaled 1.4x with native
BLAS and 4.4x with BLAS pinned).  A single worker is
left untouched, because a lone GEMM should use the whole machine.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

AUTO = "auto"

#: Environment variable read when ``max_threads`` is left on auto.
ENV_MAX_THREADS = "EMAOPT_MAX_THREADS"

#: Rows per worker below which a thread pool is not worth starting.
MIN_ROWS_PER_WORKER = 2048

#: Block rows a partition is split into when it is large.
TARGET_BLOCK_ROWS = 8192

#: Partitions per worker used to absorb imbalance (work stealing granularity).
OVERSUBSCRIBE_LARGE = 8
OVERSUBSCRIBE_SMALL = 4


def cpu_count() -> int:
    """Usable logical CPUs, respecting the process affinity mask."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return max(1, int(os.cpu_count() or 1))


def physical_cores() -> int:
    """Physical cores if ``/proc/cpuinfo`` exposes them, else logical CPUs."""
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="ignore")
    except OSError:  # pragma: no cover - non-Linux
        return cpu_count()
    pairs: set[tuple[str, str]] = set()
    physical: str | None = None
    for line in text.splitlines():
        if not line.strip():
            physical = None
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical = value.strip()
        elif key == "core id" and physical is not None:
            pairs.add((physical, value.strip()))
    if not pairs:
        return cpu_count()
    return max(1, min(len(pairs), cpu_count()))


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _is_auto(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() == AUTO)


@dataclass(frozen=True)
class ThreadPlan:
    """How a reduction will be cut and threaded, and why."""

    rows: int
    workers: int
    partitions: int
    block_size: int
    blas_threads: int | None
    max_threads: int
    cpu_count: int
    physical_cores: int
    source: str
    reason: str

    @property
    def rows_per_worker(self) -> int:
        return int(math.ceil(self.rows / max(1, self.workers)))

    @property
    def pinned_blas(self) -> bool:
        return self.blas_threads is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": int(self.rows),
            "workers": int(self.workers),
            "partitions": int(self.partitions),
            "block_size": int(self.block_size),
            "rows_per_worker": self.rows_per_worker,
            "blas_threads": None if self.blas_threads is None else int(self.blas_threads),
            "pinned_blas": self.pinned_blas,
            "max_threads": int(self.max_threads),
            "cpu_count": int(self.cpu_count),
            "physical_cores": int(self.physical_cores),
            "source": self.source,
            "reason": self.reason,
        }


def resolve_threads(
    n: int,
    *,
    workers: int | str | None = AUTO,
    max_threads: int | str | None = AUTO,
    partitions: int | str | None = AUTO,
    block_size: int | str | None = AUTO,
    blas_threads: int | str | None = AUTO,
    min_rows_per_worker: int = MIN_ROWS_PER_WORKER,
    target_block_rows: int = TARGET_BLOCK_ROWS,
) -> ThreadPlan:
    """Resolve workers, partitions, block rows and BLAS threads.

    ``max_threads`` is the ceiling requested by the caller (``"auto"`` reads
    ``EMAOPT_MAX_THREADS`` and otherwise uses the physical-core count); the
    number of workers is then the largest count that still gives every worker
    at least ``min_rows_per_worker`` rows.  Partitions are a whole multiple of
    the workers so the pool can absorb a slow partition, and block rows are
    chosen so a large partition is read in ``target_block_rows`` chunks.
    """
    n = max(1, int(n))
    cpus = cpu_count()
    physical = physical_cores()
    explicit: list[str] = []

    if _is_auto(max_threads):
        env_cap = _env_int(ENV_MAX_THREADS)
        if env_cap is not None:
            cap = max(1, min(int(env_cap), cpus))
            explicit.append(f"{ENV_MAX_THREADS}={env_cap}")
            source = "env"
        else:
            cap = max(1, min(physical, cpus))
            source = "auto"
    else:
        cap = max(1, min(int(max_threads), cpus))
        explicit.append(f"max_threads={cap}")
        source = "explicit"

    if _is_auto(workers):
        per_worker = max(1, int(min_rows_per_worker))
        n_workers = min(cap, max(1, n // per_worker))
    else:
        n_workers = max(1, min(int(workers), cap, n))
        explicit.append(f"workers={n_workers}")
        source = "explicit"

    if _is_auto(partitions):
        if n_workers <= 1:
            n_parts = 1
        else:
            rows_per_worker = n // n_workers
            factor = (
                OVERSUBSCRIBE_LARGE
                if rows_per_worker >= 2 * int(target_block_rows)
                else OVERSUBSCRIBE_SMALL
            )
            n_parts = min(n_workers * factor, max(n_workers, n // 256))
    else:
        n_parts = max(1, min(int(partitions), n))
        explicit.append(f"partitions={n_parts}")

    n_workers = max(1, min(n_workers, n_parts))
    rows_per_part = int(math.ceil(n / n_parts))

    if _is_auto(block_size):
        blocks_per_part = max(
            1, int(math.ceil(rows_per_part / max(1, int(target_block_rows))))
        )
        block = int(math.ceil(rows_per_part / blocks_per_part))
    elif block_size is None:  # legacy: one block per partition
        block = rows_per_part
        explicit.append("block=partition")
    else:
        block = max(1, int(block_size))
        explicit.append(f"block={block}")
    block = max(1, min(block, rows_per_part))

    if _is_auto(blas_threads):
        bt = 1 if n_workers > 1 else None
    elif isinstance(blas_threads, str) and blas_threads.strip().lower() in (
        "none", "off", "native", "0"
    ):
        bt = None
        explicit.append("blas=native")
    else:
        bt = max(1, int(blas_threads))
        explicit.append(f"blas={bt}")

    reason = (
        f"{n_workers} worker(s) x {n_parts} partition(s), "
        f"{block} rows/block, BLAS {'native' if bt is None else bt}; "
        f"rows={n}, min_rows_per_worker={min_rows_per_worker}"
        + (f", {', '.join(explicit)}" if explicit else "")
    )
    return ThreadPlan(
        rows=n,
        workers=n_workers,
        partitions=n_parts,
        block_size=block,
        blas_threads=bt,
        max_threads=cap,
        cpu_count=cpus,
        physical_cores=physical,
        source=source,
        reason=reason,
    )


def partition_bounds(n: int, partitions: int) -> list[tuple[int, int]]:
    """Contiguous ``[lo, hi)`` ranges whose sizes differ by at most one row."""
    n = max(0, int(n))
    parts = max(1, min(int(partitions), max(1, n)))
    base, extra = divmod(n, parts)
    bounds: list[tuple[int, int]] = []
    lo = 0
    for i in range(parts):
        hi = lo + base + (1 if i < extra else 0)
        if hi > lo:
            bounds.append((lo, hi))
        lo = hi
    return bounds


def pin_blas_threads(threads: int | None) -> bool:
    """Limit the calling thread's BLAS/OMP pools, without restoring.

    The limit is thread-local for the pools we use, so an outer
    :func:`blas_limit` does not reach pool workers.  A worker must therefore
    set it itself *before* it runs any GEMM, and calls must be serialized:
    concurrent ``openblas_set_num_threads`` calls deadlocked a 20-epoch ICA
    fold.  ``partitioned_reduce`` warms every pool worker through this
    function before submitting fold work, which gives both properties.
    """
    if threads is None:
        return False
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional dependency
        return False
    threadpool_limits(limits=int(threads)).__enter__()
    return True


@contextmanager
def blas_limit(threads: int | None) -> Iterator[bool]:
    """Pin BLAS/OpenMP thread pools for the duration, if threadpoolctl exists.

    Yields whether the limit was actually applied.  Missing ``threadpoolctl``
    is not an error: the call still runs, just without the pin.
    """
    if threads is None:
        yield False
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - optional dependency
        yield False
        return
    with threadpool_limits(limits=int(threads)):
        yield True


__all__ = [
    "AUTO",
    "ENV_MAX_THREADS",
    "MIN_ROWS_PER_WORKER",
    "TARGET_BLOCK_ROWS",
    "ThreadPlan",
    "blas_limit",
    "cpu_count",
    "partition_bounds",
    "physical_cores",
    "pin_blas_threads",
    "resolve_threads",
]
