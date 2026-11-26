"""The two update laws the schedule protocol measured: momentum and the e-means step law."""

from __future__ import annotations

import math

import numpy as np
import pytest

from emaopt.core import State, UpdateContext
from emaopt.state import EMA, EMeansStepLaw, MomentumEMA, make_schedule, make_updater


def _ctx(step: int = 0, batch: int = 256, n: int = 8000, total: int = 100) -> UpdateContext:
    return UpdateContext(
        step=step,
        batch_size=batch,
        dataset_size=n,
        total_steps=total,
        epoch=0,
        examples_processed=step * batch,
    )


def test_momentum_with_zero_beta_is_exactly_the_ema():
    ctx = _ctx()
    before = State({"counts": np.ones(4), "sums": np.zeros((4, 3))}, 0)
    stat = State({"counts": np.arange(4.0) + 2.0, "sums": np.ones((4, 3))}, 1)
    ema = EMA(alpha=0.1).update(before, stat, ctx)
    mom = MomentumEMA(alpha=0.1, beta=0.0).update(before, stat, ctx)
    for key in ("counts", "sums"):
        # Same law, different association: (1-a)p + a s versus p + a(s - p).
        # The last bits may differ; the update does not.
        assert np.allclose(ema.values[key], mom.values[key], rtol=0.0, atol=1e-15)


def test_momentum_overshoots_and_is_clamped_back_into_the_feasible_set():
    # A count falling towards zero: the velocity carries it through zero, which is
    # the failure the state-update experiment predicted (parity 1.009, target
    # never reached).  The clamp is what makes that visible instead of silent.
    state = State({"counts": np.full(4, 0.5)}, 0)
    for step in range(6):
        state = MomentumEMA(alpha=0.5, beta=0.9).update(
            state, State({"counts": np.zeros(4)}, step), _ctx(step=step)
        )
    assert np.all(state.values["counts"] >= 0.0)
    assert state.meta["_momentum_clamped"]["counts"] > 0


def test_momentum_rejects_beta_outside_the_unit_interval():
    with pytest.raises(ValueError):
        MomentumEMA(alpha=0.1, beta=1.0)


def test_emeans_law_follows_the_published_power_law():
    law = EMeansStepLaw(k=16)
    ctx = _ctx(batch=1024, n=1_000_000, total=20_000)
    expected = min(
        0.824
        * 16 ** -0.46
        * 1024 ** 0.90
        * 1_000_000 ** -0.45
        * (20_000 * 1024 / 1_000_000) ** -0.11,
        1.0,
    )
    assert law.peak(ctx) == pytest.approx(expected)
    # The peak is a function of the configuration only: the same call twice, and
    # no dependence on the statistic being damped, is the point of the law.
    assert law.peak(ctx) == law.peak(ctx)


def test_emeans_law_directions_and_annealing():
    law = EMeansStepLaw(k=8)
    small_n = law.peak(_ctx(batch=256, n=10_000, total=100))
    big_n = law.peak(_ctx(batch=256, n=10_000_000, total=100))
    small_b = law.peak(_ctx(batch=64, n=10_000, total=100))
    big_b = law.peak(_ctx(batch=4096, n=10_000, total=100))
    assert big_n < small_n, "alpha must shrink with the dataset (a_n < 0)"
    assert big_b > small_b, "alpha must grow with the batch (a_B ~ 0.9)"
    first = law.alpha(_ctx(step=0, batch=256, n=10_000, total=100))
    last = law.alpha(_ctx(step=99, batch=256, n=10_000, total=100))
    assert first > last
    assert last == pytest.approx(0.1 * first, rel=1e-6)
    mid = law.alpha(_ctx(step=50, batch=256, n=10_000, total=100))
    assert last < mid < first
    assert mid != pytest.approx(0.5 * (first + last)), "the anneal is a cosine, not a line"


def test_both_laws_are_reachable_by_name():
    assert isinstance(make_schedule("emeans_law", k=8), EMeansStepLaw)
    assert isinstance(make_updater("momentum", alpha=0.1, beta=0.5), MomentumEMA)
    assert math.isfinite(make_schedule("emeans_law", k=8).alpha(_ctx()))
