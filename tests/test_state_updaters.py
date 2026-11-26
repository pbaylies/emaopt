"""Every state-update law is pinned against its closed form."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emaopt.core import State, UpdateContext
from emaopt.state import (
    UPDATERS,
    AdaptiveEMA,
    BiasCorrectedEMA,
    ConstantSchedule,
    CosineSchedule,
    EMA,
    EffectiveSampleSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    Replace,
    RobbinsMonro,
    RunningMean,
    WeightedMean,
    make_schedule,
    make_updater,
)


def ctx(step=0, batch=10, n=1000, total=100):
    return UpdateContext(step=step, batch_size=batch, dataset_size=n, total_steps=total)


def stat(**values):
    return State({k: np.asarray(v, dtype=float) for k, v in values.items()})


def test_replace_is_the_identity_of_one_pass_rebuild():
    prev = State({"s": np.array([1.0, 2.0])})
    new = Replace().update(prev, stat(s=[5.0, 7.0]), ctx())
    assert np.allclose(new["s"], [5.0, 7.0])


def test_ema_constant_statistic_follows_closed_form():
    updater = EMA(alpha=0.25)
    state = State({"s": np.zeros(1)})
    for step in range(4):
        state = updater.update(state, stat(s=[2.0]), ctx(step=step))
    expected = 2.0 * (1.0 - 0.75**4)
    assert np.isclose(state["s"][0], expected)


def test_ema_first_update_initializes_from_statistic_when_key_absent():
    updater = EMA(alpha=0.1)
    state = updater.update(State({}), stat(s=[3.0, 4.0]), ctx())
    assert np.allclose(state["s"], [3.0, 4.0])


def test_bias_corrected_ema_has_no_zero_init_transient():
    updater = BiasCorrectedEMA(alpha=0.1)
    state = State({})
    state = updater.update(state, stat(s=[2.0]), ctx(step=0))
    assert np.allclose(state["s"], [2.0], atol=1e-9)
    state = updater.update(state, stat(s=[2.0]), ctx(step=1))
    assert np.allclose(state["s"], [2.0], atol=1e-9)


def test_bias_corrected_discards_a_warm_state_on_first_update():
    # Measured 2026-11-14 (run_bias_counts.py): the first update RESETS to the
    # batch statistic when the key is warm in values but absent from meta raw.
    prev = State({"s": np.array([100.0])})
    new = BiasCorrectedEMA(alpha=0.1).update(prev, stat(s=[3.0]), ctx())
    assert np.allclose(new["s"], [3.0])
    # warm start recorded as a seen key; the second full copy of the array
    # (``meta['_raw']``) was removed once the debias factor was shown to pin
    assert new.meta["_seen"] == {"s"}


def test_bias_corrected_exempt_key_of_a_ratio_inflates_the_decode():
    # The centroid = sums/counts hazard: exempting counts keeps the plain EMA
    # recursion (a*stat from a negligible prior) while sums reset to batch
    # scale -- the ratio is inflated by ~1/alpha for ~1/alpha updates.
    eps, a = 2.0**-16, 0.1
    prev = State({"sums": np.array([10.0 * eps]), "counts": np.array([eps])})
    stats = stat(sums=[20.0], counts=[0.5])  # true ratio 40
    both = BiasCorrectedEMA(alpha=a).update(prev, stats, ctx())
    assert np.isclose(both["sums"][0] / both["counts"][0], 40.0)
    ex = BiasCorrectedEMA(alpha=a, exempt=("counts",)).update(prev, stats, ctx())
    assert np.isclose(ex["sums"][0] / ex["counts"][0], 40.0 / a, rtol=1e-2)


def test_bias_corrected_ema_converges_to_plain_ema_on_a_changing_signal():
    bc = BiasCorrectedEMA(alpha=0.5)
    plain = EMA(alpha=0.5)
    s_bc = State({"s": np.zeros(1)})
    s_plain = State({"s": np.zeros(1)})
    for step, value in enumerate([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 5.0, 7.0]):
        s_bc = bc.update(s_bc, stat(s=[value]), ctx(step=step))
        s_plain = plain.update(s_plain, stat(s=[value]), ctx(step=step))
    # the transient vanishes: the two recursions differ only by the 1/p factor
    assert abs(float(s_bc["s"][0] - s_plain["s"][0])) < 0.6


def test_bias_corrected_exempt_key_keeps_plain_ema():
    bc = BiasCorrectedEMA(alpha=0.5, exempt=("counts",))
    state = State({"counts": np.zeros(1)})
    state = bc.update(state, stat(counts=[10.0]), ctx(step=0))
    assert np.isclose(state["counts"][0], 5.0)


def test_running_mean_is_the_arithmetic_mean():
    updater = RunningMean()
    state = State({})
    for step, value in enumerate([1.0, 2.0, 6.0]):
        state = updater.update(state, stat(s=[value]), ctx(step=step))
    assert np.isclose(state["s"][0], 3.0)


def test_weighted_mean_with_batch_weights_matches_macqueen():
    updater = WeightedMean("batch_size")
    state = State({})
    state = updater.update(state, stat(s=[10.0]), ctx(batch=10))
    state = updater.update(state, stat(s=[20.0]), ctx(batch=30))
    assert np.isclose(state["s"][0], (10.0 * 10.0 + 30.0 * 20.0) / 40.0)


def test_weighted_mean_unit_weight_matches_running_mean():
    a, b = WeightedMean("unit"), RunningMean()
    sa, sb = State({}), State({})
    for step, value in enumerate([2.0, 8.0, 4.0]):
        sa = a.update(sa, stat(s=[value]), ctx(step=step))
        sb = b.update(sb, stat(s=[value]), ctx(step=step))
    assert np.isclose(sa["s"][0], sb["s"][0])


def test_robbins_monro_step_decays_as_power_law():
    rm = RobbinsMonro(alpha0=1.0, gamma=1.0, t0=1.0)
    assert np.isclose(rm.alpha(ctx(step=0)), 1.0)
    assert np.isclose(rm.alpha(ctx(step=9)), 0.1)
    rm2 = RobbinsMonro(alpha0=1.0, gamma=0.5, t0=1.0)
    assert np.isclose(rm2.alpha(ctx(step=3)), 0.5)


def test_adaptive_ema_step_shrinks_with_evidence():
    updater = AdaptiveEMA()
    alphas = []
    state = State({})
    for step in range(50):
        state = updater.update(state, stat(s=[1.0]), ctx(step=step, batch=32))
        alphas.append(updater.alpha(ctx(step=step, batch=32)))
    assert alphas[0] > alphas[-1]
    assert all(a > 0.0 for a in alphas)
    # a state with lots of evidence takes small steps
    state = State({"s": np.zeros(1)}, meta={"_ess": {"s": 1e6}})
    small = updater.update(state, stat(s=[1.0]), ctx(step=0, batch=8))
    assert np.isclose(small["s"][0], 8.0 / (8.0 + 1e6), atol=1e-9)


def test_per_key_schedules_are_independent():
    updater = EMA(alpha=0.0, per_key={"counts": 1.0})
    state = State({"sums": np.array([1.0]), "counts": np.array([1.0])})
    new = updater.update(state, stat(sums=[5.0], counts=[5.0]), ctx())
    assert np.allclose(new["sums"], [1.0])  # alpha 0 keeps the old value
    assert np.allclose(new["counts"], [5.0])


def test_cosine_schedule_anneals_and_respects_horizon():
    sched = CosineSchedule(alpha0=0.2, alpha_end_ratio=0.1)
    start = sched.alpha(ctx(step=0, total=100))
    mid = sched.alpha(ctx(step=49, total=100))
    end = sched.alpha(ctx(step=99, total=100))
    assert start > mid > end
    assert np.isclose(end, 0.02, atol=1e-9)
    assert np.isclose(mid, 0.11, atol=2e-2)


def test_other_schedules_are_monotone_or_constant():
    assert LinearSchedule(0.4).alpha(ctx(step=0, total=10)) > LinearSchedule(0.4).alpha(
        ctx(step=9, total=10))
    assert ConstantSchedule(0.3).alpha(ctx()) == 0.3
    assert 0.0 < InverseSqrtSchedule(0.5).alpha(ctx(step=0)) <= 0.5
    assert EffectiveSampleSchedule(prior=100.0).alpha(ctx(batch=100)) < 1.0


def test_schedule_registry_and_errors():
    assert make_schedule("cosine", alpha0=0.1) is not None
    with pytest.raises(KeyError):
        make_schedule("not-a-schedule")


def test_updater_registry_roundtrip_and_config_is_json_serialisable():
    assert set(UPDATERS) == {
        "replace", "running_mean", "weighted_mean", "ema",
        "bias_corrected", "robbins_monro", "adaptive", "momentum",
        "plateau", "portfolio",
    }
    for name in UPDATERS:
        updater = make_updater(name)
        json.dumps(updater.config())
    with pytest.raises(KeyError):
        make_updater("nope")


def test_updaters_do_not_mutate_their_input_state():
    prev = State({"s": np.array([1.0])})
    for name in UPDATERS:
        updater = make_updater(name)
        updater.update(prev, stat(s=[9.0]), ctx())
        assert np.allclose(prev["s"], [1.0]), f"{name} mutated its input"
