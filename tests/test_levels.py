"""The precision/speed spectrum (T26): resolution, aliases, and what a level does."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt import FitConfig, GaussianMixture, egmm_gpu_fit, hostkernels
from emaopt import levels


def _toy(n=400, d=6, k=3, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)), rng.normal(size=(k, d))


def test_resolve_defaults_to_quality():
    spec = levels.resolve_level()
    assert spec.name == "quality"
    assert spec.host_kernel is False
    assert (spec.egmm_dtype, spec.egmm_stat_dtype, spec.gramian_dtype) == (
        "float64", "float64", "float64")


def test_aliases_point_at_the_three_levels():
    assert levels.resolve_level("reference").name == "quality"
    assert levels.resolve_level("balanced").name == "auto"
    assert levels.resolve_level("fast").name == "performance"
    assert levels.resolve_level("AUTO").name == "auto"


def test_unknown_level_lists_the_choices():
    with pytest.raises(ValueError, match="unknown level"):
        levels.resolve_level("turbo")


def test_describe_covers_every_level():
    rows = levels.describe()
    assert [r["level"] for r in rows] == ["quality", "auto", "performance"]
    assert all(r["drift_budget_nats"] > 0 for r in rows)
    # the spectrum must be monotone in speed and in error budget
    budgets = [r["drift_budget_nats"] for r in rows]
    speeds = [r["expected_step_speedup"] for r in rows]
    assert budgets == sorted(budgets)
    assert speeds == sorted(speeds)


def test_use_sets_and_restores_the_host_kernel(monkeypatch):
    monkeypatch.delenv("EMAOPT_HOST_KERNEL", raising=False)
    with levels.use("quality"):
        assert hostkernels.available() is False
    with levels.use("auto"):
        status = hostkernels.status()
        # Portable either way: the artifact may or may not be built on this
        # machine, but the level must never be the reason it is unavailable.
        assert "disabled by EMAOPT_HOST_KERNEL" not in status["error"]
    assert "EMAOPT_HOST_KERNEL" not in __import__("os").environ


def test_use_restores_a_previous_setting(monkeypatch):
    monkeypatch.setenv("EMAOPT_HOST_KERNEL", "numpy")
    with levels.use("performance"):
        assert "EMAOPT_HOST_KERNEL" not in __import__("os").environ
    assert __import__("os").environ["EMAOPT_HOST_KERNEL"] == "numpy"


@pytest.mark.parametrize("level,expected", [
    ("quality", ("float64", "float64")),
    ("auto", ("float64", "float32")),
    ("performance", ("float32", "float32")),
])
def test_egmm_fit_follows_the_level(level, expected):
    x, _ = _toy()
    cfg = FitConfig(batch_size=100, nepochs=1, seed=0)
    result = egmm_gpu_fit(GaussianMixture(3, covariance="diag", seed=0), x, cfg,
                          device="cpu", level=level)
    res = result.extras["residency"]
    assert (res["dtype"], res["stat_dtype"]) == expected
    assert res["level"] == level
    assert np.isfinite(result.objective)


def test_explicit_dtypes_beat_the_level():
    x, _ = _toy()
    cfg = FitConfig(batch_size=100, nepochs=1, seed=0)
    result = egmm_gpu_fit(GaussianMixture(3, covariance="diag", seed=0), x, cfg,
                          device="cpu", level="performance",
                          dtype=__import__("torch").float64)
    res = result.extras["residency"]
    assert res["dtype"] == "float64"          # explicit wins
    assert res["stat_dtype"] == "float32"     # the level fills the rest
    assert res["level"] == "performance"


def test_no_level_keeps_the_reference_contract():
    x, _ = _toy()
    cfg = FitConfig(batch_size=100, nepochs=1, seed=0)
    result = egmm_gpu_fit(GaussianMixture(3, covariance="diag", seed=0), x, cfg,
                          device="cpu")
    res = result.extras["residency"]
    assert (res["dtype"], res["stat_dtype"]) == ("float64", "float64")
    assert res["level"] == "custom"


def test_performance_is_close_to_quality_on_a_toy_fit():
    """The budget is real: f32 endpoints land inside the level's own budget."""
    x, _ = _toy(n=900)
    cfg = FitConfig(batch_size=150, nepochs=2, seed=0)
    base = egmm_gpu_fit(GaussianMixture(3, covariance="diag", seed=0), x, cfg,
                        device="cpu", level="quality")
    fast = egmm_gpu_fit(GaussianMixture(3, covariance="diag", seed=0), x, cfg,
                        device="cpu", level="performance")
    budget = levels.resolve_level("performance").drift_budget
    assert abs(fast.objective - base.objective) <= max(budget, 1e-5)
