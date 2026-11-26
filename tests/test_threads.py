"""Thread policy: auto sizing, work division, BLAS pinning and the parallel fold."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emaopt.state.sufficient import CentroidStat, GramianStat
from emaopt.stream import (
    iter_blocks,
    partitioned_fit,
    partitioned_reduce,
    partitioned_stat,
    stat_from_blocks,
)
from emaopt.threads import (
    ENV_MAX_THREADS,
    cpu_count,
    partition_bounds,
    physical_cores,
    resolve_threads,
)


def test_partition_bounds_are_even_and_cover_every_row():
    for n, p in [(10, 3), (1000, 7), (5, 5), (9, 4), (1, 1), (17, 16)]:
        bounds = partition_bounds(n, p)
        sizes = [hi - lo for lo, hi in bounds]
        assert sum(sizes) == n
        assert max(sizes) - min(sizes) <= 1
        assert bounds[0][0] == 0 and bounds[-1][1] == n


def test_auto_resolution_uses_physical_cores_and_pins_blas_for_workers():
    plan = resolve_threads(400_000)
    assert plan.max_threads == min(physical_cores(), cpu_count())
    assert plan.workers >= 1
    if plan.workers > 1:
        assert plan.blas_threads == 1
        assert plan.pinned_blas
    assert plan.partitions >= plan.workers
    assert 0 < plan.block_size <= int(np.ceil(plan.rows / plan.partitions))
    json.dumps(plan.to_dict())  # serialisable for FitResult.extras


def test_max_threads_caps_every_auto_choice():
    plan = resolve_threads(1_000_000, max_threads=4)
    assert plan.workers == 4
    assert plan.max_threads == 4
    assert plan.partitions >= 4
    assert plan.source == "explicit"


def test_single_worker_leaves_blas_alone():
    plan = resolve_threads(1_500)
    assert plan.workers == 1
    assert plan.partitions == 1
    assert plan.blas_threads is None
    assert not plan.pinned_blas


def test_environment_sets_the_default_ceiling(monkeypatch):
    monkeypatch.setenv(ENV_MAX_THREADS, "3")
    plan = resolve_threads(10_000_000)
    assert plan.max_threads <= 3
    assert plan.workers <= 3
    assert plan.source == "env"
    assert ENV_MAX_THREADS in plan.reason


def test_explicit_workers_are_clamped_to_the_ceiling_and_partitions():
    plan = resolve_threads(10_000, workers=64, max_threads=16, partitions=3)
    assert plan.workers == 3
    assert plan.partitions == 3


def test_auto_reduction_matches_a_single_stream():
    rng = np.random.default_rng(11)
    x = np.ascontiguousarray(rng.normal(size=(60_000, 8)))

    def make():
        return GramianStat(center=True)

    single = stat_from_blocks(make, iter_blocks(x, 4096))
    merged, plan = partitioned_reduce(make, x, max_threads=4)
    assert plan.workers == 4
    assert plan.pinned_blas
    assert merged.n_examples == x.shape[0]
    np.testing.assert_allclose(merged.values["gram"], single.values["gram"],
                               rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(merged.values["sum"], single.values["sum"],
                               rtol=1e-10, atol=1e-8)


def test_parallel_fold_is_deterministic_and_records_its_plan():
    rng = np.random.default_rng(12)
    x = np.ascontiguousarray(rng.normal(size=(40_000, 4)))
    first, plan = partitioned_reduce(lambda: GramianStat(center=True), x, max_threads=4)
    second, _ = partitioned_reduce(lambda: GramianStat(center=True), x, max_threads=4)
    assert np.array_equal(first.values["gram"], second.values["gram"])
    assert first.meta["threads"]["workers"] == plan.workers
    assert first.meta["threads"]["partitions"] == plan.partitions
    assert isinstance(first.meta["blas_pinned"], bool)


def test_centroid_reduction_with_auto_threads_keeps_counts():
    rng = np.random.default_rng(13)
    x = np.ascontiguousarray(rng.normal(size=(30_000, 5)))
    centroids = x[np.linspace(0, x.shape[0] - 1, 8).astype(int)].copy()

    def make():
        return CentroidStat(8, centroids=centroids, mode="sum", prior=0.0,
                            act_every=10 ** 9)

    single = stat_from_blocks(make, iter_blocks(x, 2048))
    merged = partitioned_stat(make, x, max_threads=4)
    assert np.allclose(merged.values["counts"], single.values["counts"], atol=1e-6)
    np.testing.assert_allclose(merged.values["sums"], single.values["sums"],
                               rtol=1e-10, atol=1e-6)


def test_blas_is_limited_for_the_pool_and_restored_afterwards():
    # The limit is per-thread, so each worker must set it itself. The pool
    # warms and pins every worker before any fold starts: applying it around
    # the pool leaves workers at native counts, while applying it concurrently
    # from the fold races openblas_set_num_threads and deadlocked a 20-epoch
    # ICA run.
    threadpoolctl = pytest.importorskip("threadpoolctl")
    if not threadpoolctl.threadpool_info():
        pytest.skip("no BLAS/OpenMP pools to limit")

    seen: list[int] = []

    class Probe(GramianStat):
        def update(self, block, **kwargs):  # noqa: D401 - test probe
            info = threadpoolctl.threadpool_info()
            seen.append(max([int(t.get("num_threads", 1)) for t in info] or [1]))
            return super().update(block, **kwargs)

    before = [int(t["num_threads"]) for t in threadpoolctl.threadpool_info()]
    rng = np.random.default_rng(14)
    x = np.ascontiguousarray(rng.normal(size=(20_000, 6)))
    partitioned_stat(lambda: Probe(center=True), x, workers=2, max_threads=2,
                     n_partitions=4)
    after = [int(t["num_threads"]) for t in threadpoolctl.threadpool_info()]
    assert seen and max(seen) == 1
    assert before == after


def test_partitioned_fit_reports_threads_and_examples():
    rng = np.random.default_rng(15)
    x = np.ascontiguousarray(rng.normal(size=(25_000, 3)))
    result = partitioned_fit(lambda: GramianStat(center=True), x, max_threads=4)
    assert result.examples_processed == 25_000
    assert result.updates > 1
    assert result.extras["threads"]["workers"] == result.extras["workers"]
    assert result.extras["blas_pinned"] is True
    assert result.params["covariance"].shape == (3, 3)
    assert np.isfinite(result.params["covariance"]).all()


def test_native_blas_is_selectable_for_the_parallel_fold():
    plan = resolve_threads(100_000, max_threads=4, blas_threads="native")
    assert plan.workers == 4
    assert plan.blas_threads is None
    assert not plan.pinned_blas
