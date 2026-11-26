"""Measured memory budget and the batch/stream resolution it drives."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import EMeans, LloydKMeans
from emaopt.benchmarks import datasets
from emaopt.budget import MemoryBudget, resolve_compute
from emaopt.core import FitConfig, fit


def _meminfo(tmp_path, available_kb):
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       32000000 kB\nMemFree:         1000000 kB\n"
        f"MemAvailable:   {available_kb} kB\n",
        encoding="ascii",
    )
    return path


def _cgroup(tmp_path, limit, usage):
    directory = tmp_path / "cgroup"
    directory.mkdir()
    (directory / "memory.max").write_text(str(limit), encoding="ascii")
    (directory / "memory.current").write_text(str(usage), encoding="ascii")
    return directory


def test_probe_reads_memavailable(tmp_path):
    budget = MemoryBudget.probe(meminfo_path=_meminfo(tmp_path, 2048), cgroup_dir=None)
    assert budget.available_bytes == 2048 * 1024
    assert budget.source == "meminfo"
    assert budget.limit_bytes == int(2048 * 1024 * budget.fraction)


def test_probe_reads_cgroup_v2_headroom(tmp_path):
    budget = MemoryBudget.probe(
        meminfo_path=tmp_path / "missing",
        cgroup_dir=_cgroup(tmp_path, 1_000_000, 400_000),
    )
    assert budget.available_bytes == 600_000
    assert budget.source == "cgroup-v2"
    assert budget.detail["cgroup"] == "cgroup-v2"


def test_probe_takes_the_tighter_of_the_two_sources(tmp_path):
    budget = MemoryBudget.probe(
        meminfo_path=_meminfo(tmp_path, 100_000),  # 102.4 MB
        cgroup_dir=_cgroup(tmp_path, 50_000_000, 10_000_000),  # 40 MB
    )
    assert budget.available_bytes == 40_000_000
    assert budget.source == "meminfo+cgroup-v2"


def test_probe_override_and_unavailable_paths(tmp_path):
    override = MemoryBudget.probe(override=1234)
    assert override.available_bytes == 1234 and override.source == "override"
    missing = MemoryBudget.probe(meminfo_path=tmp_path / "nope", cgroup_dir=None)
    assert missing.available_bytes is None and missing.source == "unavailable"


def test_resolve_idle_batches_and_pressure_streams():
    idle = MemoryBudget(available_bytes=10 ** 12, source="test")
    plan = resolve_compute(20_000, 64, 16, stat_bytes=16_384, budget=idle)
    assert plan.mode == "batch"
    assert plan.block_size == 20_000
    assert plan.act_frequency == 1

    tight = MemoryBudget(available_bytes=2_000_000, source="test")  # 1 MB limit
    plan = resolve_compute(20_000, 64, 16, stat_bytes=16_384, budget=tight)
    assert plan.mode == "stream"
    assert 0 < plan.block_size < 20_000
    assert plan.block_size & (plan.block_size - 1) == 0  # power of two
    assert plan.reason.startswith("resident estimate")
    assert plan.to_dict()["limit_bytes"] == tight.limit_bytes


def test_resolver_never_streams_an_optimizer_without_a_streaming_form():
    tight = MemoryBudget(available_bytes=2_000_000, source="test")
    plan = resolve_compute(20_000, 64, 16, budget=tight, supports_streaming=False)
    assert plan.mode == "batch"
    assert "no streaming form" in plan.reason


def test_fit_records_the_resolution_under_simulated_pressure():
    ds = datasets.blobs(n=20_000, k=8, d=32, sep=8.0, seed=0)
    tight = MemoryBudget(available_bytes=200_000, source="test")
    model = EMeans(8, nepochs=2, seed=0)
    result = fit(model, ds.x, FitConfig(nepochs=2, seed=0, budget=tight))
    plan = result.extras["compute_plan"]
    assert plan is not None
    assert plan["mode"] == "stream"
    assert plan["available_bytes"] == 200_000
    assert plan["source"] == "test"
    assert result.extras["batch_size"] == min(result.extras["batch_size_requested"],
                                              plan["block_size"])
    assert result.extras["batch_size"] <= plan["block_size"]
    assert np.isfinite(result.objective)


def test_fit_on_a_roomy_budget_still_batches():
    ds = datasets.blobs(n=800, k=4, d=3, sep=6.0, seed=1)
    roomy = MemoryBudget(available_bytes=10 ** 12, source="test")
    model = EMeans(4, nepochs=2, seed=0)
    result = fit(model, ds.x, FitConfig(nepochs=2, seed=0, budget=roomy))
    plan = result.extras["compute_plan"]
    assert plan["mode"] == "batch"
    assert result.extras["batch_size"] == result.extras["batch_size_requested"]


def test_lloyd_reports_the_pressure_but_does_not_change_semantics():
    ds = datasets.blobs(n=400, k=3, d=2, sep=6.0, seed=2)
    tight = MemoryBudget(available_bytes=10_000, source="test")
    result = fit(LloydKMeans(3, niter=2, seed=0), ds.x,
                 FitConfig(nepochs=2, batch_size=None, seed=0, budget=tight))
    assert result.extras["compute_plan"]["mode"] == "batch"
    assert result.extras["batch_size"] == len(ds.x)


def test_budget_can_be_disabled():
    ds = datasets.blobs(n=100, k=2, d=2, sep=4.0, seed=3)
    result = fit(EMeans(2, nepochs=1, seed=0), ds.x,
                 FitConfig(nepochs=1, seed=0, budget=None))
    assert result.extras["compute_plan"] is None
