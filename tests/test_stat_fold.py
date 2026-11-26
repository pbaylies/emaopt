"""T48: a stat's ``mode="ema"`` fold is the law, not a copy of it.

The fold used to apply ``alpha`` directly, so ``SufficientStat(mode="ema")`` and
``EMA(alpha)`` were two transcriptions that could drift.  These tests compare the
stat's fold against the law's own ``update`` on the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.core import State, UpdateContext
from emaopt.state import CentroidStat, EMA, RunningMean
from emaopt.state.schedules import CosineSchedule
from emaopt.state.sufficient import SufficientStat


def ctx(step=0, batch=3, n=3, total=1):
    return UpdateContext(step=step, batch_size=batch, dataset_size=n, total_steps=total)


def test_ema_stat_fold_equals_the_ema_law():
    stat = CentroidStat(1, 1, centroids=np.array([[1.0]]), mode="ema", alpha=0.4,
                        prior=0.0)
    stat.prepare(np.array([[1.0]]))
    stat.values = {"sums": np.array([[1.0]]), "counts": np.array([1.0])}
    stats = {"sums": np.array([[6.0]]), "counts": np.array([3.0])}

    stat._fold(stats, 3.0)

    expected = EMA(alpha=0.4).update(
        State({"sums": np.array([[1.0]]), "counts": np.array([1.0])}),
        State(dict(stats)), ctx(),
    )
    for key in ("sums", "counts"):
        assert np.allclose(stat.values[key], expected[key])
    assert stat.last_alpha == pytest.approx(0.4)


def test_scheduled_ema_stat_fold_uses_the_schedule():
    schedule = CosineSchedule(alpha0=0.4, alpha_end_ratio=0.1)
    stat = CentroidStat(1, 1, centroids=np.array([[1.0]]), mode="ema",
                        alpha=schedule, prior=0.0)
    stat.prepare(np.array([[1.0]])).configure(total_steps=10, dataset_size=3)
    stat.values = {"sums": np.array([[0.0]]), "counts": np.array([0.0])}

    stat._fold({"sums": np.array([[3.0]]), "counts": np.array([1.0])}, 3.0)

    a = schedule.alpha(ctx(total=10))
    assert stat.values["sums"][0, 0] == pytest.approx(a * 3.0)
    assert stat.values["counts"][0] == pytest.approx(a * 1.0)


def test_a_stat_can_hold_the_updater_itself():
    law = RunningMean()
    stat = CentroidStat(1, 1, centroids=np.array([[1.0]]), mode="sum", prior=0.0,
                        updater=law)
    assert stat.mode == "ema"          # an updater implies the blending fold
    stat.prepare(np.array([[1.0]]))
    stat.values = {"sums": np.array([[0.0]]), "counts": np.array([0.0])}
    stats = {"sums": np.array([[2.0]]), "counts": np.array([2.0])}

    stat._fold(stats, 2.0)

    expected = law.update(
        State({"sums": np.array([[0.0]]), "counts": np.array([0.0])}),
        State(dict(stats)), ctx(batch=2, n=2),
    )
    for key in ("sums", "counts"):
        assert np.allclose(stat.values[key], expected[key])
    assert stat.config()["updater"] == "running_mean"


def test_sum_mode_is_still_plain_addition():
    stat = SufficientStat(mode="sum")
    stat._fold({"a": np.array([1.0])}, 4.0)
    stat._fold({"a": np.array([2.0])}, 4.0)
    assert np.allclose(stat.values["a"], [3.0])
    assert stat.weight == 8.0
    assert stat.n_examples == 8


def test_running_mean_stat_matches_the_law_across_blocks():
    """A fresh key must be announced as fresh, or its counter advances twice.

    Found in review: ``_fold`` passed ``fresh=frozenset()``, so a key seen for the
    first time was counted by the reset branch *and* by the counter, and the
    second block came out with ``a = 1/3`` instead of ``1/2``.
    """
    law = RunningMean()
    stat = SufficientStat(mode="sum", updater=law)
    state = State({})
    for step in range(3):
        values = np.array([float(step + 1)])
        stat._fold({"a": values}, 1.0)
        state = law.update(state, State({"a": values}), ctx(step=step, batch=1, n=1))
        assert np.allclose(stat.values["a"], state["a"]), f"block {step}"


def test_a_plain_scalar_alpha_stat_matches_the_ema_law_across_updates():
    stat = SufficientStat(mode="ema", alpha=0.25)
    law = EMA(alpha=0.25)
    state = State({})
    for step in range(4):
        stat._fold({"a": np.array([2.0])}, 1.0)
        state = law.update(state, State({"a": np.array([2.0])}),
                           ctx(step=step, batch=1, n=1))
        assert np.allclose(stat.values["a"], state["a"])
        assert stat.steps == step + 1


def test_init_mass_reaches_the_stat_fold():
    """Review: ``_fold`` ignored ``init_mass``, so a stat and a fit disagreed."""
    law = EMA(alpha=0.5)
    law.init_mass = 2.0
    stat = SufficientStat(mode="sum", updater=law)
    stat.values = {"a": np.array([100.0])}
    stat._fold({"a": np.array([3.0])}, 1.0)
    assert stat.values["a"][0] == pytest.approx((2.0 * 100.0 + 3.0) / 3.0)
    assert stat.last_alpha == pytest.approx(1.0 / 3.0)
