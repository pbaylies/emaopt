"""T39: every record says what its objective measured, and ratios refuse mixes."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.analysis.efficiency import assert_comparable_kinds, objective_kind
from emaopt.analysis.statistics import summarize_objectives
from emaopt.core import FitConfig, fit
from emaopt.state import CentroidStat


def test_core_fit_tags_full_data_and_held_out():
    from emaopt.algorithms import GaussianMixture

    x = np.random.default_rng(0).normal(size=(120, 3))
    model = GaussianMixture(2, covariance="diag", seed=0)
    full = fit(model, x, FitConfig(batch_size=30, nepochs=1, budget=None))
    assert full.extras["objective_kind"] == "full_data"
    assert all(h["objective_kind"] == "full_data" for h in full.history)

    held = fit(model, x, FitConfig(batch_size=30, nepochs=1, budget=None),
               eval_data=x[:20])
    assert held.extras["objective_kind"] == "held_out"
    assert all(h["objective_kind"] == "held_out" for h in held.history)


def test_stat_fit_tags_its_objective():
    from emaopt.stream import stat_fit

    x = np.random.default_rng(1).normal(size=(200, 3))
    stat = CentroidStat(2, 3)
    plain = stat_fit(CentroidStat(2, 3), x, block_size=50)
    assert plain.extras["objective_kind"] is None

    scored = stat_fit(stat, x, block_size=50,
                      objective=lambda s: float(np.sum(s.decode()["centroids"] ** 2)))
    assert scored.extras["objective_kind"] == "stat"
    assert all(h["objective_kind"] == "stat" for h in scored.history)


def test_device_prequential_kind():
    torch = pytest.importorskip("torch")
    from emaopt.algorithms import GaussianMixture
    from emaopt.gpu import egmm_gpu_fit

    x = np.random.default_rng(2).normal(size=(100, 3))
    model = GaussianMixture(2, covariance="diag", seed=0)
    result = egmm_gpu_fit(model, x.repeat(1, 1),
                          FitConfig(batch_size=25, nepochs=1, budget=None),
                          device="cpu", eval_mode="stream")
    assert result.extras["objective_kind"] == "prequential"
    assert all(h["objective_kind"] == "prequential" for h in result.history)


def test_mixed_kinds_are_refused():
    with pytest.raises(ValueError, match="not comparable"):
        assert_comparable_kinds("prequential", "full_data")
    assert assert_comparable_kinds("full_data", "full_data") == "full_data"
    assert assert_comparable_kinds(None, None) == "unknown"


class _Result:
    def __init__(self, objective, kind):
        self.objective = objective
        self.extras = {"objective_kind": kind}


def test_summarize_refuses_a_mixed_seed_set():
    same = [_Result(1.0, "full_data"), _Result(1.2, "full_data")]
    assert summarize_objectives(same)["n"] == 2

    mixed = [_Result(1.0, "full_data"), _Result(0.3, "prequential")]
    with pytest.raises(ValueError, match="not comparable"):
        summarize_objectives(mixed)


def test_objective_kind_reads_records_and_results():
    assert objective_kind({"objective_kind": "held_out"}) == "held_out"
    assert objective_kind(_Result(0.0, "stat")) == "stat"
    assert objective_kind(object()) is None
