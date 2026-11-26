"""The scheduling families the sweep measures: anneal shapes, jitter, plateau, portfolio."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import GaussianMixture
from emaopt.core import FitConfig, State, UpdateContext
from emaopt.state import (
    EMA,
    AnnealSchedule,
    EMeansStepLaw,
    JitterSchedule,
    PlateauEMA,
    PortfolioEMA,
    make_schedule,
)


def ctx(step: int = 0, batch: int = 256, n: int = 8000, total: int = 100) -> UpdateContext:
    return UpdateContext(step=step, batch_size=batch, dataset_size=n, total_steps=total)


# --------------------------------------------------------------------------- #
# anneal shapes
# --------------------------------------------------------------------------- #
def test_anneal_shapes_share_endpoints_and_differ_in_the_middle():
    schedules = {
        shape: AnnealSchedule(alpha0=0.4, alpha_end_ratio=0.1, shape=shape)
        for shape in ("cosine", "linear", "geometric")
    }
    for schedule in schedules.values():
        assert schedule.alpha(ctx(step=0, total=100)) == pytest.approx(0.4)
        assert schedule.alpha(ctx(step=99, total=100)) == pytest.approx(0.04, rel=1e-9)
    middle = {name: s.alpha(ctx(step=49, total=100)) for name, s in schedules.items()}
    # Same endpoints, different time spent near each: that is the whole point of
    # comparing shapes rather than endpoints.
    assert middle["cosine"] > middle["linear"] > middle["geometric"]


def test_anneal_warmup_rises_to_the_peak_then_decays():
    schedule = AnnealSchedule(alpha0=0.4, alpha_end_ratio=0.1, warmup_fraction=0.1)
    assert schedule.alpha(ctx(step=0, total=100)) == pytest.approx(0.04)
    assert schedule.alpha(ctx(step=9, total=100)) == pytest.approx(0.4)
    assert schedule.alpha(ctx(step=50, total=100)) < 0.4
    assert schedule.alpha(ctx(step=99, total=100)) == pytest.approx(0.04, rel=1e-6)


def test_anneal_rejects_an_unknown_shape():
    with pytest.raises(ValueError):
        AnnealSchedule(alpha0=0.4, shape="parabolic")
    assert isinstance(make_schedule("anneal", alpha0=0.2), AnnealSchedule)


def test_emeans_law_anneal_modes_and_warmup():
    constant = EMeansStepLaw(k=8, anneal="none")
    assert constant.alpha(ctx(step=0)) == pytest.approx(constant.peak(ctx(step=0)))
    assert constant.alpha(ctx(step=99, total=100)) == pytest.approx(
        constant.peak(ctx(step=99, total=100))
    )
    cosine = EMeansStepLaw(k=8, anneal="cosine")
    geometric = EMeansStepLaw(k=8, anneal="geometric")
    assert cosine.alpha(ctx(step=99, total=100)) == pytest.approx(
        geometric.alpha(ctx(step=99, total=100)), rel=1e-9
    )
    assert cosine.alpha(ctx(step=50, total=100)) > geometric.alpha(ctx(step=50, total=100))

    warm = EMeansStepLaw(k=8, warmup_fraction=0.1)
    assert warm.alpha(ctx(step=0, total=100)) == pytest.approx(
        warm.peak(ctx(step=0, total=100)) / 10.0
    )
    assert warm.alpha(ctx(step=9, total=100)) == pytest.approx(
        warm.peak(ctx(step=9, total=100))
    )
    assert warm.alpha(ctx(step=99, total=100)) == pytest.approx(
        cosine.alpha(ctx(step=99, total=100)) * 0.1 / 0.1, rel=1e-6
    )
    with pytest.raises(ValueError):
        EMeansStepLaw(k=8, anneal="concave")


# --------------------------------------------------------------------------- #
# stochastic annealing
# --------------------------------------------------------------------------- #
def test_jitter_is_reproducible_and_anneals_to_the_base_schedule():
    base = AnnealSchedule(alpha0=0.4, alpha_end_ratio=0.1)
    schedule = JitterSchedule(base=base, sigma=0.8, seed=3)
    assert schedule.alpha(ctx(step=17, total=100)) == schedule.alpha(ctx(step=17, total=100))
    drawn = {round(schedule.alpha(ctx(step=step, total=100)), 12) for step in range(20)}
    assert len(drawn) > 1, "a zero-sigma schedule would be the base; this one must move"
    # the temperature reaches zero at the horizon, so the last step is the base
    assert schedule.alpha(ctx(step=99, total=100)) == pytest.approx(
        base.alpha(ctx(step=99, total=100))
    )
    flat = JitterSchedule(base=base, sigma=0.0, seed=3)
    assert flat.alpha(ctx(step=5, total=100)) == pytest.approx(base.alpha(ctx(step=5, total=100)))


# --------------------------------------------------------------------------- #
# objective feedback (the H6 counter-example)
# --------------------------------------------------------------------------- #
def test_plateau_halves_after_a_stalled_objective_and_respects_the_floor():
    updater = PlateauEMA(alpha0=0.4, factor=0.5, patience=1, min_alpha=0.05)
    state = State({"s": np.zeros(1)})
    levels = []
    for step in range(12):
        statistics = State({"s": np.ones(1)}, 0, {"nll": 5.0})  # never improves
        state = updater.update(state, statistics, ctx(step=step))
        levels.append(float(state.meta["_plateau_alpha"]))
    assert levels[0] == 0.4
    assert levels[-1] == pytest.approx(0.05)
    assert any(later < earlier for earlier, later in zip(levels, levels[1:]))


def test_plateau_holds_the_step_while_the_objective_improves():
    updater = PlateauEMA(alpha0=0.4, patience=0)
    state = State({"s": np.zeros(1)})
    for step in range(8):
        statistics = State({"s": np.ones(1)}, 0, {"nll": 5.0 - 0.1 * step})
        state = updater.update(state, statistics, ctx(step=step))
    assert state.meta["_plateau_alpha"] == 0.4


def test_plateau_never_invents_a_plateau_without_an_objective():
    updater = PlateauEMA(alpha0=0.3)
    state = State({"s": np.zeros(1)})
    for step in range(1, 12):
        state = updater.update(state, State({"s": np.ones(1)}), ctx(step=step))
    assert state.meta["_plateau_alpha"] == 0.3


# --------------------------------------------------------------------------- #
# online selection
# --------------------------------------------------------------------------- #
def test_portfolio_concentrates_on_the_rate_that_predicts_the_batches():
    updater = PortfolioEMA((0.05, 0.2, 0.6), eta=5.0)
    state = State({"counts": np.zeros(2)})
    for step in range(40):
        state = updater.update(state, State({"counts": np.array([4.0, 4.0])}), ctx(step=step))
    weights = np.asarray(state.meta["_portfolio_weights"])
    assert weights.sum() == pytest.approx(1.0)
    # a constant statistic is predicted soonest by the largest rate
    assert int(np.argmax(weights)) == 2
    assert float(np.dot(weights, updater.alphas)) > 0.4


def test_portfolio_returns_a_convex_combination_and_leaves_the_input_state():
    before = State({"s": np.array([1.0])})
    updater = PortfolioEMA((0.1, 0.5))
    new = updater.update(before, State({"s": np.array([9.0])}), ctx())
    assert np.allclose(before["s"], [1.0]), "the input state must not be mutated"
    assert 1.0 < new["s"][0] < 9.0
    assert np.isclose(float(np.sum(new.meta["_portfolio_weights"])), 1.0)


# --------------------------------------------------------------------------- #
# e-means' extras, ported to the mixture
# --------------------------------------------------------------------------- #
def test_gmm_revival_splits_a_donor_and_preserves_its_mean():
    model = GaussianMixture(2, covariance="diag", revival=True, seed=0)
    counts = np.array([60.0, 1e-9])
    sums = np.array([[6.0, 0.0], [0.0, 0.0]])
    sq = np.array([[60.0 + 60.0, 60.0], [0.0, 0.0]])  # donor variance 1 in dim 0
    state = State({"counts": counts, "sums": sums, "sq": sq})
    new = model.update_state(state, state, ctx(batch=120))

    assert np.allclose(new["counts"], [30.0, 30.0])
    means = new["sums"] / new["counts"][:, None]
    assert np.linalg.norm(means[0] - means[1]) > 0.0, "the halves must be separated"
    assert np.allclose(means.mean(axis=0), sums[0] / 60.0, atol=1e-9)
    variance = new["sq"] / new["counts"][:, None] - means**2
    assert np.all(variance > -1e-9), "a split must leave both halves feasible"
    assert new.meta["_revivals"] == 1


def test_gmm_finalize_replaces_the_ema_state_with_one_exact_pass():
    rng = np.random.default_rng(0)
    x = np.vstack(
        [rng.normal(-2.0, 0.5, size=(200, 2)), rng.normal(2.0, 0.5, size=(200, 2))]
    )
    model = GaussianMixture(2, covariance="diag", updater=EMA(alpha=0.1), finalize=True, seed=0)
    config = FitConfig(batch_size=50, nepochs=2, seed=0)
    per_epoch = len(x) // 50  # 8 batches of 50 over 400 rows
    state = model.initialize(x, config)
    model.begin_fit(state, config, len(x), 50, total_steps=2 * per_epoch)
    for epoch in range(2):
        for i in range(per_epoch):
            batch = x[i * 50 : (i + 1) * 50]
            frame = UpdateContext(
                step=epoch * per_epoch + i, batch_size=50, dataset_size=len(x),
                total_steps=2 * per_epoch, epoch=epoch,
            )
            state = model.update_state(state, model.measure(batch, state), frame)
            state.step = epoch * per_epoch + i + 1
            model.on_batch(state, batch, frame)

    assert state.accumulators["final_complete"]
    counts = model.decode(state)["counts"]
    # one exact pass over the last epoch's 400 examples, not the EMA's ~one batch
    assert counts.sum() == pytest.approx(float(len(x)), rel=0.02)
    assert counts.sum() > 3 * 50
