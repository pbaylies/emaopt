"""T42: BLAS pinning must not leak out of ``partitioned_reduce``.

``pin_blas_threads`` mutates process-global OpenBLAS/OMP state and deliberately
does not restore it (a pool worker must pin itself before its first GEMM).  The
contract is therefore that every *caller* brackets it: ``partitioned_reduce``
uses ``blas_limit`` around the pool, which restores the thread counts captured at
entry.  This test checks the state, not the intent -- on the success path and on
the path where a fold raises.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("threadpoolctl")

from threadpoolctl import threadpool_info  # noqa: E402

from emaopt.state import CentroidStat  # noqa: E402
from emaopt.stream import partitioned_reduce  # noqa: E402


def _thread_counts() -> list[int]:
    return sorted(int(p.get("num_threads", -1)) for p in threadpool_info())


@pytest.fixture()
def data() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(6000, 8))


def test_threads_are_restored_after_a_successful_reduce(data):
    before = _thread_counts()
    # One shared prepare block: each partition would otherwise seed k-means++
    # from its own first block and the merge fingerprint would (correctly)
    # refuse to add statistics measured under different parameters.
    stat, plan = partitioned_reduce(
        lambda: CentroidStat(4, 8), data,
        n_partitions=4, workers=4, blas_threads=1,
        prepare=data[:256],
    )
    assert plan.workers > 1
    assert _thread_counts() == before
    centroids = stat.decode()["centroids"]
    assert np.all(np.isfinite(centroids))


def test_threads_are_restored_when_a_fold_raises(data):
    before = _thread_counts()
    calls = {"n": 0}

    def make_stat():
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("fold failed on purpose")
        return CentroidStat(2, 8)

    with pytest.raises(Exception):
        partitioned_reduce(make_stat, data, n_partitions=4, workers=2,
                           blas_threads=1)
    assert _thread_counts() == before


def test_a_single_worker_does_not_pin_the_process(data):
    before = _thread_counts()
    partitioned_reduce(lambda: CentroidStat(2, 8), data, n_partitions=1,
                       workers=1, blas_threads="auto", prepare=data[:256])
    assert _thread_counts() == before
