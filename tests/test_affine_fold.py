"""The affine fold (T46), ``init_mass`` (T45) and one State per block (T47)."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import GaussianMixture
from emaopt.core import State, UpdateContext
from emaopt.state import (
    AdaptiveEMA,
    BiasCorrectedEMA,
    EMA,
    MomentumEMA,
    OptimizerStat,
    PlateauEMA,
    RobbinsMonro,
    RunningMean,
    WeightedMean,
)
from emaopt.state.affine import AffineUpdater


def ctx(step=0, batch=10, n=1000, total=100):
    return UpdateContext(step=step, batch_size=batch, dataset_size=n, total_steps=total)


def stat(**values):
    return State({k: np.asarray(v, dtype=float) for k, v in values.items()})


def _affine_laws():
    return [EMA(alpha=0.1), RunningMean(), WeightedMean(), AdaptiveEMA(),
            RobbinsMonro(alpha0=0.5), BiasCorrectedEMA(alpha=0.1)]


def test_the_six_blend_laws_share_one_fold():
    for updater in _affine_laws():
        assert isinstance(updater, AffineUpdater), type(updater).__name__
    # the laws that are not a plain blend keep their own update
    for updater in (MomentumEMA(), PlateauEMA()):
        assert not isinstance(updater, AffineUpdater)


def test_the_generic_fold_reproduces_each_laws_closed_form():
    state = State({"s": np.zeros(1)})
    for step in range(4):
        state = EMA(alpha=0.25).update(state, stat(s=[2.0]), ctx(step=step))
    assert np.isclose(state["s"][0], 2.0 * (1.0 - 0.75 ** 4))

    state = State({})
    for step, value in enumerate([1.0, 2.0, 6.0]):
        state = RunningMean().update(state, stat(s=[value]), ctx(step=step))
    assert np.isclose(state["s"][0], 3.0)

    state = State({})
    rm = RobbinsMonro(alpha0=1.0, gamma=1.0, t0=1.0)
    for step, value in enumerate([1.0, 3.0]):
        state = rm.update(state, stat(s=[value]), ctx(step=step))
    # a_0 = 1 replaces (S=1), a_1 = 1/2 blends: 0.5*1 + 0.5*3
    assert np.isclose(state["s"][0], 0.5 * 1.0 + 0.5 * 3.0)

    a, b = WeightedMean(), RunningMean()
    sa, sb = State({}), State({})
    for step, value in enumerate([2.0, 8.0, 4.0]):
        sa = a.update(sa, stat(s=[value]), ctx(step=step, batch=5 + step))
        sb = b.update(sb, stat(s=[value]), ctx(step=step, batch=5 + step))
    assert np.isclose(sa["s"][0], sb["s"][0])


def test_init_mass_forces_the_same_first_step_for_every_law():
    """The initialization confound T45 is about: equal mass on update one."""
    warm = State({"s": np.array([100.0])})
    statistic = stat(s=[3.0])
    expected = (2.0 * 100.0 + 3.0) / 3.0     # a = 1/(1+init_mass) = 1/3
    for updater in _affine_laws():
        updater.init_mass = 2.0
        got = float(updater.update(warm, statistic, ctx(step=0))["s"][0])
        assert got == pytest.approx(expected), type(updater).__name__


def test_init_mass_none_keeps_each_laws_own_first_step():
    warm = State({"s": np.array([100.0])})
    statistic = stat(s=[3.0])
    kept = float(EMA(alpha=0.1).update(warm, statistic, ctx(step=0))["s"][0])
    discarded = float(RobbinsMonro(alpha0=1.0).update(warm, statistic, ctx(step=0))["s"][0])
    assert kept == pytest.approx(0.9 * 100.0 + 0.1 * 3.0)
    assert discarded == pytest.approx(3.0)
    assert kept != pytest.approx(discarded)   # the confound, made visible


def test_device_fold_accepts_the_newly_affine_laws():
    from emaopt.gpu import _fold_alphas

    for updater in (RobbinsMonro(alpha0=0.5), WeightedMean(), BiasCorrectedEMA(alpha=0.1)):
        model = GaussianMixture(3, covariance="diag", updater=updater, seed=0)
        alphas = _fold_alphas(model, ctx(step=0, batch=8), ("counts",), {})
        assert set(alphas) == {"counts"}
        assert 0.0 <= alphas["counts"] <= 1.0
    with pytest.raises(NotImplementedError):
        _fold_alphas(
            GaussianMixture(3, covariance="diag", updater=MomentumEMA(), seed=0),
            ctx(), ("counts",), {},
        )


def test_optimizer_stat_copies_its_state_once_per_block():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(300, 4))
    stat_obj = OptimizerStat(GaussianMixture(3, covariance="diag", seed=0),
                             mode="ema", alpha=1.0)
    stat_obj.prepare(x)

    calls = []
    original = OptimizerStat.to_state

    def counting(self):
        calls.append(1)
        return original(self)

    OptimizerStat.to_state = counting
    try:
        stat_obj.update(x[:50])
    finally:
        OptimizerStat.to_state = original
    assert len(calls) == 1, f"{len(calls)} state copies per block (was 2)"
