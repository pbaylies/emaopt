"""Streaming driver: blocks in, stats out, merges for parallelism.

ONLINE_PLAN phases 2, 3 and 6.  Three things live here:

* :func:`iter_blocks` -- one deterministic block iterator over an ndarray, a
  ``.npy`` path (memory-mapped, so an f16 shard is never materialised as float64
  in full), or any object supporting row slicing.
* :func:`stat_fit` -- the driver that folds blocks into a
  :class:`~emaopt.state.sufficient.SufficientStat`, evaluates a user objective
  when it can, and reports examples/compute/memory the way ``FitResult`` does.
* :func:`partitioned_stat` / :func:`merge_stats` -- split a dataset into
  partitions, build one stat per partition (optionally in threads), and merge.
  The acceptance test for the merge algebra is that this equals the
  single-stream stat, which is what makes phase 6 (merge as parallelism) real
  rather than aspirational.
"""

from __future__ import annotations

import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np

from .budget import MemoryBudget, resolve_compute
from .state.sufficient import SufficientStat, merge_stats
from .threads import (
    ThreadPlan,
    blas_limit,
    partition_bounds,
    pin_blas_threads,
    resolve_threads,
)

_EPS = 1e-12

#: Worker-thread marker for the per-thread BLAS limit.
_WORKER_PIN = threading.local()


# --------------------------------------------------------------------------- #
# block iteration
# --------------------------------------------------------------------------- #
def as_block_source(data: Any) -> Any:
    """Open ``data`` for row slicing, memory-mapping ``.npy`` paths."""
    if isinstance(data, (str, os.PathLike, Path)):
        return np.load(Path(data), mmap_mode="r")
    return data


def n_rows(data: Any) -> int:
    data = as_block_source(data)
    if not hasattr(data, "shape") or len(data.shape) != 2:
        raise ValueError("data must be a 2-D array or a path to a .npy array")
    return int(data.shape[0])


def iter_blocks(
    data: Any,
    block_size: int | None = None,
    *,
    nepochs: int = 1,
    shuffle: bool | str = False,
    seed: int = 0,
    drop_last: bool = False,
    max_updates: int | None = None,
    dtype: Any = np.float64,
) -> Iterator[np.ndarray]:
    """Yield row blocks from an array, memmap or path.

    Blocks are converted to ``dtype`` one at a time; the source is never copied
    in full.

    ``shuffle`` has two distinct meanings and they are not interchangeable:

    ``True`` (or ``"blocks"``)
        permute *block indices*; a block's rows stay contiguous, which keeps a
        memmap read sequential.  Every block is a contiguous slice -- a good
        sample of the whole file, a *bad* i.i.d. sample when the file is ordered.
    ``"rows"``
        permute *rows* once per epoch and cut blocks out of the permutation, so
        each block is an exchangeable sample (what EMA / running-mean laws
        assume).  This turns the memmap read into a gather and costs the same
        order of memory bandwidth as the rows it touches -- measure before
        using it on a shard.

    Rows are never permuted by the default, because the shipped LAION path is
    memmap-sequential on purpose.
    """
    source = as_block_source(data)
    n = n_rows(source)
    bs = int(n if block_size is None else block_size)
    bs = max(1, min(bs, n))
    from .plan import plan_batches

    per_epoch = plan_batches(n, bs, nepochs, drop_last=drop_last).per_epoch
    if isinstance(shuffle, str):
        mode = shuffle.strip().lower()
        if mode not in ("rows", "blocks", "true", "false", ""):
            raise ValueError(
                f"shuffle must be bool or 'rows'/'blocks', got {shuffle!r}"
            )
        rows = mode == "rows"
        by_block = mode in ("blocks", "true")
    else:
        rows = False
        by_block = bool(shuffle)
    rng = np.random.default_rng(seed)
    emitted = 0
    for _ in range(max(1, int(nepochs))):
        if rows:
            # Exchangeable blocks: cut the epoch's block *positions* out of one
            # row permutation, so no block is an ordered window of the source.
            order_rows = rng.permutation(n)
            for i in range(per_epoch):
                idx = order_rows[i * bs : (i + 1) * bs]
                if idx.size == 0:
                    continue
                yield np.asarray(source[idx], dtype=dtype)
                emitted += 1
                if max_updates is not None and emitted >= max_updates:
                    return
            continue
        order = rng.permutation(per_epoch) if by_block else np.arange(per_epoch)
        for i in order:
            start = int(i) * bs
            block = source[start : start + bs]
            if len(block) == 0:
                continue
            yield np.asarray(block, dtype=dtype)
            emitted += 1
            if max_updates is not None and emitted >= max_updates:
                return


def block_steps(n: int, block_size: int, *, nepochs: int = 1, drop_last: bool = False) -> int:
    from .plan import plan_batches

    return plan_batches(n, block_size, nepochs, drop_last=drop_last).total_steps


# --------------------------------------------------------------------------- #
# stat driver
# --------------------------------------------------------------------------- #
@dataclass
class StatFitResult:
    """What a stat-driven fit produced, in the shape ``FitResult`` reports."""

    stat: str
    mode: str
    params: Any
    history: list[dict[str, Any]]
    updates: int
    examples_processed: int
    wall_time: float
    config: dict[str, Any]
    nbytes: int
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {
            "stat": self.stat,
            "mode": self.mode,
            "updates": int(self.updates),
            "examples_processed": int(self.examples_processed),
            "wall_time": float(self.wall_time),
            "nbytes": int(self.nbytes),
            "config": self.config,
        }
        out.update(self.extras)
        return out


def stat_fit(
    stat: SufficientStat,
    data: Any,
    *,
    block_size: int | None = None,
    nepochs: int = 1,
    shuffle: bool = False,
    seed: int = 0,
    drop_last: bool = False,
    max_updates: int | None = None,
    objective: Callable[[SufficientStat], float] | None = None,
    eval_every: int | None = None,
    budget: MemoryBudget | None = None,
    prepare: Any = None,
    configure_schedule: bool = True,
) -> StatFitResult:
    """Fold ``data`` into ``stat`` block by block and report what happened.

    ``prepare`` may be a data block used to seed parameters (k-means++,
    optimizer warm start).  When it is ``None`` the stat seeds itself from the
    first block it sees.
    """
    source = as_block_source(data)
    n = n_rows(source)
    d = int(source.shape[1])
    bs = int(n if block_size is None else block_size)
    if budget is not None and budget.limit_bytes is not None:
        plan = resolve_compute(
            n, d, getattr(stat, "k", 0) or 0,
            stat_bytes=stat.nbytes, budget=budget, requested_batch=bs,
            nepochs=nepochs,
        )
        if plan.mode == "stream":
            bs = min(bs, plan.block_size)
    bs = max(1, min(bs, n))
    if prepare is not None:
        # A stat that can seed itself from its first block (CentroidStat,
        # FastICAStat in running-whitening mode) is left to do so; stats that
        # derive from a parameter (NMF basis, LDA beta, dictionary atoms,
        # optimizer warm start) are seeded here.
        stat.prepare(np.asarray(prepare, dtype=np.float64))
    if configure_schedule:
        stat.configure(
            total_steps=block_steps(n, bs, nepochs=nepochs, drop_last=drop_last),
            dataset_size=n,
        )
    wall0 = time.perf_counter()
    # ``mode`` says how blocks were folded; ``objective_kind`` says what the
    # objective measured.  A stat objective is whatever the caller passed
    # (usually a full-data score of ``stat.decode()``), so it is labelled
    # rather than silently ratioed against a prequential number (T39).
    objective_kind = "stat" if objective is not None else None
    history: list[dict[str, Any]] = []
    updates = 0
    examples = 0
    eval_s = 0.0          # objective passes, split out from the fold time
    every = int(eval_every or max(1, block_steps(n, bs, nepochs=nepochs,
                                                  drop_last=drop_last) // 20))
    for block in iter_blocks(source, bs, nepochs=nepochs, shuffle=shuffle, seed=seed,
                             drop_last=drop_last, max_updates=max_updates):
        stat.update(block)
        updates += 1
        examples += int(block.shape[0])
        if updates % every == 0 or updates == max_updates:
            record: dict[str, Any] = {
                "update": updates,
                "examples_processed": examples,
                "weight": float(stat.weight),
                "nbytes": int(stat.nbytes),
                "wall_time": time.perf_counter() - wall0,
            }
            if objective is not None:
                _eval0 = time.perf_counter()
                record["objective"] = float(objective(stat))
                eval_s += time.perf_counter() - _eval0
            history.append(record)
    params = stat.decode()
    wall = time.perf_counter() - wall0
    record = {
        "update": updates,
        "examples_processed": examples,
        "weight": float(stat.weight),
        "nbytes": int(stat.nbytes),
        "wall_time": wall,
    }
    if objective is not None:
        _eval0 = time.perf_counter()
        record["objective"] = float(objective(stat))
        eval_s += time.perf_counter() - _eval0
    if not history or history[-1]["update"] != updates:
        history.append(record)
    for entry in history:
        entry.setdefault("objective_kind", objective_kind)
    return StatFitResult(
        stat=stat.name,
        mode=stat.mode,
        params=params,
        history=history,
        updates=updates,
        examples_processed=examples,
        wall_time=wall,
        config=stat.config(),
        nbytes=int(stat.nbytes),
        extras={
            "n": n,
            "d": d,
            "block_size": bs,
            "nepochs": nepochs,
            "seed": seed,
            "state_keys": list(stat.values),
            "objective_kind": objective_kind,
            "eval_time": float(eval_s),
            "train_time": float(max(0.0, wall - eval_s)),
        },
    )


# --------------------------------------------------------------------------- #
# merge as parallelism
# --------------------------------------------------------------------------- #
def stat_from_blocks(
    make_stat: Callable[[], SufficientStat],
    blocks: Iterable[np.ndarray],
    *,
    weights: Sequence[float] | None = None,
) -> SufficientStat:
    """Build a fresh stat by folding an iterable of blocks."""
    stat = make_stat()
    for i, block in enumerate(blocks):
        stat.update(block, weight=None if weights is None else float(weights[i]))
    return stat


def _pin_worker(threads: int | None) -> None:
    """Limit this worker's BLAS pools once, before it folds anything."""
    if threads is None:
        return
    if getattr(_WORKER_PIN, "threads", None) == threads:
        return
    pin_blas_threads(threads)
    _WORKER_PIN.threads = threads


def _warm_workers(pool: ThreadPoolExecutor, plan: ThreadPlan) -> None:
    """Pin every pool worker before any of them starts a fold.

    The setter is serialized by ``lock`` and no GEMM is running while the
    warm-up tasks wait at ``barrier``.  Calling it from workers during the
    fold instead races concurrent ``openblas_set_num_threads`` calls and
    deadlocked a 20-epoch ICA run; applying it only around the pool misses
    the workers entirely.
    """
    if plan.blas_threads is None or plan.workers <= 1:
        return
    lock = threading.Lock()
    barrier = threading.Barrier(plan.workers)

    def warm(_: int) -> None:
        with lock:
            _pin_worker(plan.blas_threads)
        barrier.wait(timeout=30)

    list(pool.map(warm, range(plan.workers)))


def partitioned_reduce(
    make_stat: Callable[[], SufficientStat],
    data: Any,
    *,
    n_partitions: int | str | None = "auto",
    block_size: int | str | None = "auto",
    workers: int | str | None = "auto",
    max_threads: int | str | None = None,
    blas_threads: int | str | None = "auto",
    shuffle: bool = False,
    seed: int = 0,
    prepare: Any = None,
    configure_schedule: bool = False,
) -> tuple[SufficientStat, ThreadPlan]:
    """Fold contiguous partitions in parallel and merge them, with the plan.

    Sizing comes from :func:`emaopt.threads.resolve_threads`: ``workers="auto"``
    uses the physical-core count capped by ``max_threads`` (or
    ``EMAOPT_MAX_THREADS``) and by a floor of rows per worker, partitions are a
    multiple of the workers so a slow partition can be absorbed, and BLAS is
    limited to one thread for the whole pool only when more than one worker
    runs.  The returned :class:`~emaopt.threads.ThreadPlan` is also stored on
    the merged stat's ``meta`` so a gate can record what it actually did.

    Equivalence is the acceptance gate: for additive (``mode="sum"``) stats
    the merged result equals the single-stream fold up to float summation order.
    """
    source = as_block_source(data)
    n = n_rows(source)
    plan = resolve_threads(
        n,
        workers=workers,
        max_threads=max_threads,
        partitions=n_partitions,
        block_size=block_size,
        blas_threads=blas_threads,
    )
    ranges = partition_bounds(n, plan.partitions)

    def build(bounds_pair: tuple[int, int]) -> SufficientStat:
        lo, hi = bounds_pair
        stat = make_stat()
        if prepare is not None:
            stat.prepare(np.asarray(prepare, dtype=np.float64))
        if configure_schedule:
            stat.configure(
                total_steps=max(1, int(math.ceil((hi - lo) / plan.block_size))),
                dataset_size=n,
            )
        # The BLAS limit is process-global and is applied once by the outer
        # context below, before the pool starts.  Calling it from workers as
        # well raced concurrent openblas_set_num_threads calls and deadlocked
        # a 20-epoch ICA fold, so workers do not touch the pools.
        for block in iter_blocks(source[lo:hi], plan.block_size, shuffle=shuffle,
                                 seed=seed + lo):
            stat.update(block)
        return stat

    pinned = False
    if plan.workers <= 1:
        stats = [build(pair) for pair in ranges]
    else:
        with blas_limit(plan.blas_threads) as pinned:
            with ThreadPoolExecutor(max_workers=plan.workers) as pool:
                _warm_workers(pool, plan)
                stats = list(pool.map(build, ranges))
    merged = merge_stats(stats)
    merged.meta["threads"] = plan.to_dict()
    merged.meta["blas_pinned"] = bool(pinned)
    return merged, plan


def partitioned_stat(
    make_stat: Callable[[], SufficientStat],
    data: Any,
    *,
    n_partitions: int | str | None = "auto",
    block_size: int | str | None = "auto",
    workers: int | str | None = "auto",
    max_threads: int | str | None = None,
    blas_threads: int | str | None = "auto",
    shuffle: bool = False,
    seed: int = 0,
) -> SufficientStat:
    """Build one stat per contiguous partition, then merge them.

    ``workers="auto"`` (the default) picks the worker count from the physical
    cores, the row count and ``max_threads``; pass ``workers=1`` to force a
    sequential reduction or ``blas_threads="native"`` to leave BLAS alone.
    """
    stat, _ = partitioned_reduce(
        make_stat, data, n_partitions=n_partitions, block_size=block_size,
        workers=workers, max_threads=max_threads, blas_threads=blas_threads,
        shuffle=shuffle, seed=seed,
    )
    return stat


def partitioned_fit(
    make_stat: Callable[[], SufficientStat],
    data: Any,
    *,
    prepare: Any = None,
    n_partitions: int | str | None = "auto",
    block_size: int | str | None = "auto",
    workers: int | str | None = "auto",
    max_threads: int | str | None = None,
    blas_threads: int | str | None = "auto",
    seed: int = 0,
) -> StatFitResult:
    """A parallel reduction reported the way :func:`stat_fit` reports a fold."""
    source = as_block_source(data)
    n = n_rows(source)
    d = int(source.shape[1])
    wall0 = time.perf_counter()
    stat, plan = partitioned_reduce(
        make_stat, source, n_partitions=n_partitions, block_size=block_size,
        workers=workers, max_threads=max_threads, blas_threads=blas_threads,
        seed=seed, prepare=prepare, configure_schedule=True,
    )
    params = stat.decode()
    wall = float(time.perf_counter() - wall0)
    return StatFitResult(
        stat=stat.name,
        mode=stat.mode,
        params=params,
        history=[{
            "update": int(stat.steps),
            "examples_processed": int(n),
            "weight": float(stat.weight),
            "nbytes": int(stat.nbytes),
            "wall_time": wall,
        }],
        updates=int(stat.steps),
        examples_processed=int(n),
        wall_time=wall,
        config=stat.config(),
        nbytes=int(stat.nbytes),
        extras={
            "n": int(n),
            "d": int(d),
            "block_size": int(plan.block_size),
            "partitions": int(plan.partitions),
            "workers": int(plan.workers),
            "threads": plan.to_dict(),
            "blas_pinned": bool(stat.meta.get("blas_pinned", False)),
        },
    )


__all__ = [
    "iter_blocks",
    "block_steps",
    "as_block_source",
    "n_rows",
    "stat_fit",
    "StatFitResult",
    "stat_from_blocks",
    "partitioned_reduce",
    "partitioned_stat",
    "partitioned_fit",
    "merge_stats",
]
