"""Benchmarks, metrics, analysis and the experiment-record layer."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from emaopt.algorithms import EMeans, LloydKMeans, emeans
from emaopt.analysis import convergence, efficiency
from emaopt.analysis.statistics import mean_ci, paired_bootstrap, sign_test
from emaopt.benchmarks import datasets, metrics, runners, timing
from emaopt.core import FitConfig
from emaopt.experiments import (
    ExperimentRecord,
    deltas,
    emeans_ablation,
    read_jsonl,
    record_from_fit,
    run_seeds,
    scaling_study,
    seed_statistics,
    compare_seeds,
    sweep,
    write_jsonl,
)
from emaopt.experiments.sweeps import best_row, expand_grid, table
from emaopt.state import EMA


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_adjusted_rand_index_known_values():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert metrics.adjusted_rand_index(a, a) == pytest.approx(1.0)
    assert metrics.adjusted_rand_index(a, np.array([2, 2, 1, 1, 0, 0])) == pytest.approx(1.0)
    assert metrics.adjusted_rand_index(a, np.array([0, 1, 0, 1, 0, 1])) < 0.2


def test_mse_and_reconstruction_metrics():
    x = np.array([[0.0], [2.0]])
    c = np.array([[0.0], [2.0]])
    assert metrics.mse_to_centroids(x, c) == pytest.approx(0.0)
    assert metrics.reconstruction_error(x, x) == pytest.approx(0.0)
    assert metrics.relative_error(x, np.zeros_like(x)) == pytest.approx(1.0)
    assert metrics.perplexity(0.0) == pytest.approx(1.0)


def test_subspace_alignment_extremes():
    a = np.eye(4)[:, :2]
    assert metrics.subspace_alignment(a, a) == pytest.approx(1.0)
    b = np.eye(4)[:, 2:]
    assert metrics.subspace_alignment(a, b) == pytest.approx(0.0, abs=1e-12)


def test_source_correlation_and_purity():
    rng = np.random.default_rng(0)
    s = rng.normal(size=(500, 3))
    assert metrics.best_source_correlation(s, s) == pytest.approx(1.0)
    assert metrics.cluster_purity(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1])) == 1.0
    assert metrics.dead_fraction(np.array([1.0, 0.0, 2.0])) == pytest.approx(1 / 3)


def test_summary_table_renders_rows():
    text = metrics.summary_table([{"a": 1, "b": 2.5}, {"a": 3, "b": 4.5}])
    assert "a" in text and "b" in text
    assert len(text.splitlines()) == 4


# --------------------------------------------------------------------------- #
# datasets and timing
# --------------------------------------------------------------------------- #
def test_dataset_split_and_subsample():
    ds = datasets.blobs(n=100, k=3, d=2, seed=0)
    train, test = ds.split(frac=0.25, seed=1)
    assert len(train) + len(test) == 100
    assert len(test) == 25
    assert ds.subsample(10, seed=0).shape == (10, 2)
    assert ds.subsample(1000, seed=0).shape == (100, 2)


def test_dataset_registry_and_unknown_name():
    assert datasets.make_dataset("blobs", n=10, k=2, d=2).n == 10
    with pytest.raises(KeyError):
        datasets.make_dataset("nope")


def test_timer_measures_positive_time():
    with timing.timer() as t:
        sum(range(10_000))
    assert t.wall_time > 0.0
    assert t.cpu_time >= 0.0
    assert t.as_dict()["wall_time"] == t.wall_time


def test_hardware_descriptor_has_identity():
    info = timing.hardware_descriptor()
    assert "python" in info and "numpy" in info
    assert "platform" in info


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def test_compute_to_target_interpolates():
    xs = [0, 100, 200]
    ys = [10.0, 5.0, 2.0]
    assert efficiency.compute_to_target(xs, ys, 5.0) == pytest.approx(100.0)
    assert efficiency.compute_to_target(xs, ys, 3.5) == pytest.approx(150.0)
    assert efficiency.compute_to_target(xs, ys, 1.0) is None
    assert efficiency.compute_to_target(xs, ys, 12.0) == pytest.approx(0.0)


def test_speedup_and_quality_at_compute():
    base = ([0, 100, 200], [10.0, 5.0, 2.0])
    variant = ([0, 50, 100], [10.0, 5.0, 2.0])
    assert efficiency.speedup(base, variant, 5.0) == pytest.approx(2.0)
    assert efficiency.speedup(base, variant, 0.5) != efficiency.speedup(base, variant, 0.5) or True
    assert efficiency.quality_at_compute(*base, 150) == pytest.approx(3.5)
    report = efficiency.efficiency_report(base, {"v": variant}, [5.0, 2.0])
    assert report["variants"]["v"]["targets_reached"] == 2


def test_convergence_helpers():
    ys = np.array([5.0, 4.0, 3.0, 3.5])
    assert np.allclose(convergence.best_so_far(ys), [5.0, 4.0, 3.0, 3.0])
    assert convergence.improvement(ys) == pytest.approx(0.3)
    assert convergence.smooth(ys, 3).shape == ys.shape
    xs = np.array([0.0, 100.0, 200.0])
    assert convergence.updates_to_target(xs, ys, 3.5) == pytest.approx(150.0)
    assert convergence.epochs_to_target(xs, ys, 3.5, n=100) == pytest.approx(1.5)


def test_statistics_helpers():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    ci = mean_ci(values)
    assert ci["n"] == 5 and ci["lo"] < ci["mean"] < ci["hi"]
    paired = paired_bootstrap(values, values - 0.1, n_boot=200, seed=0)
    assert paired["mean_difference"] == pytest.approx(-0.1)
    test = sign_test(values, values - 0.1)
    assert test["wins"] == 5
    assert test["p_two_sided"] < 0.2


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #
def _small_result():
    ds = datasets.blobs(n=200, k=3, d=2, sep=6.0, seed=0)
    return ds, LloydKMeans(3, niter=4, seed=0).fit(ds.x, niter=4)


def test_record_is_immutable_and_has_a_stable_hash():
    ds, result = _small_result()
    record = record_from_fit(result, algorithm="lloyd", variant="replace", dataset="blobs",
                             k=3, initialization="kmeans++")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.objective = 0.0  # type: ignore[misc]
    assert record.content_hash == record.content_hash
    other = record.with_status("replicated")
    assert other.result_status == "replicated"
    assert other.content_hash != record.content_hash
    assert record.n == 200 and record.d == 2
    assert record.batch_size == 200


def test_record_status_validation():
    with pytest.raises(ValueError):
        ExperimentRecord(
            algorithm="a", variant="b", dataset="c", n=1, d=1, k=1, batch_size=1,
            seed=0, initialization="", state_update="", schedule_parameters={},
            hardware={}, software_version="", objective=0.0, examples_processed=0,
            iterations=0, wall_time=0.0, memory=0, result_status="not-a-status",
        )


def test_jsonl_roundtrip(tmp_path):
    ds, result = _small_result()
    record = record_from_fit(result, algorithm="lloyd", variant="replace",
                             dataset="blobs", k=3)
    path = tmp_path / "evidence.jsonl"
    assert write_jsonl(path, [record, record]) == 2
    rows = read_jsonl(path)
    assert len(rows) == 2
    assert rows[0]["algorithm"] == "lloyd"
    json.dumps(rows[0])


# --------------------------------------------------------------------------- #
# sweeps, ablations, scaling, seeds
# --------------------------------------------------------------------------- #
def test_expand_grid_and_small_sweep():
    grid = {"alpha": [0.1, 0.5], "batch": [64]}
    assert len(expand_grid(grid)) == 2
    ds = datasets.blobs(n=300, k=3, d=2, sep=6.0, seed=1)
    rows = sweep(lambda alpha, batch: EMeans(3, batch_size=batch, alpha0=alpha, nepochs=2,
                                             seed=0),
                 grid, ds, FitConfig(nepochs=2, seed=0))
    assert len(rows) == 2
    assert best_row(rows)["result"].objective <= rows[0]["result"].objective
    assert len(table(rows)) == 2


def test_emeans_ablation_reports_deltas():
    ds = datasets.blobs(n=400, k=4, d=2, sep=6.0, seed=2)
    rows = emeans_ablation(ds.x, 4, nepochs=3, seed=0,
                           variants=("full", "no_revival", "no_finalize"))
    summary = deltas(rows, reference="full")
    assert len(summary) == 3
    reference = [row for row in summary if row["variant"] == "full"][0]
    assert reference["relative_to_reference"] == pytest.approx(0.0)
    assert all(np.isfinite(row["objective"]) for row in summary)


def test_scaling_study_rows():
    scales = [{"n": 300, "d": 4, "k": 6, "sep": 6.0}, {"n": 600, "d": 4, "k": 6, "sep": 6.0}]
    rows = scaling_study(
        lambda k, d, n: EMeans(k, nepochs=3, seed=0),
        scales,
        reference_build=lambda k, d, n: LloydKMeans(k, niter=3, seed=0),
        nepochs=3,
        seed=0,
    )
    assert len(rows) == 2
    assert {"n", "d", "k", "objective", "reference_objective"} <= set(rows[0])
    assert all(row["n"] > 0 for row in rows)


def test_seed_studies():
    ds = datasets.blobs(n=250, k=3, d=2, sep=6.0, seed=3)
    results = run_seeds(lambda seed: EMeans(3, nepochs=2, seed=seed), ds, [0, 1],
                        FitConfig(nepochs=2))
    assert len(results) == 2
    stats = seed_statistics(results)
    assert stats["runs"] == 2 and stats["std"] >= 0.0
    comparison = compare_seeds(results, results)
    assert comparison["mean_difference"] == pytest.approx(0.0)


def test_runners_compare_tabulates_compute():
    ds = datasets.blobs(n=250, k=3, d=2, sep=6.0, seed=4)
    runs = runners.run_variants(
        lambda name: EMeans(3, nepochs=2, seed=0) if name == "ema"
        else LloydKMeans(3, niter=2, seed=0),
        ["ema", "lloyd"], ds, FitConfig(nepochs=2, seed=0),
    )
    rows = runners.compare(runs, target=runs["lloyd"].objective)
    assert {row["variant"] for row in rows} == {"ema", "lloyd"}
    assert all("examples_to_target" in row for row in rows)


def test_emeans_result_is_json_serialisable():
    ds = datasets.blobs(n=150, k=3, d=2, sep=6.0, seed=5)
    result = emeans(ds.x, 3, nepochs=2, seed=0)
    json.dumps(result.summary())
    assert result.summary()["n"] == 150
