"""Regressions for the updater-layer review (2026-11-14).

Each test pins a bug that was reproduced against the shipped code, so the fix
cannot silently revert.  Names state the failure, not the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.core import State, UpdateContext
from emaopt.state import (
    AdaptiveEMA,
    BiasCorrectedEMA,
    CentroidStat,
    MomentumEMA,
    PlateauEMA,
    RunningMean,
    WeightedMean,
    make_schedule,
    merge_stats,
)


def ctx(step=0, batch=10, n=1000, total=100):
    return UpdateContext(step=step, batch_size=batch, dataset_size=n, total_steps=total)


def stat(**values):
    return State({k: np.asarray(v, dtype=float) for k, v in values.items()})


def test_plateau_halves_on_a_flat_negative_nll():
    """``best * (1 - tol)`` never triggered for best < 0, so alpha stayed fixed."""
    updater = PlateauEMA(alpha0=0.4, factor=0.5, patience=1, min_alpha=0.05)
    state = State({"s": np.zeros(1)})
    levels = []
    for step in range(12):
        statistics = State({"s": np.ones(1)}, 0, {"nll": -2000.0})  # flat, negative
        state = updater.update(state, statistics, ctx(step=step))
        levels.append(float(state.meta["_plateau_alpha"]))
    assert levels[0] == 0.4
    assert levels[-1] == pytest.approx(0.05)
    assert any(later < earlier for earlier, later in zip(levels, levels[1:]))


def test_plateau_still_halves_on_a_flat_positive_nll():
    updater = PlateauEMA(alpha0=0.4, factor=0.5, patience=1, min_alpha=0.05)
    state = State({"s": np.zeros(1)})
    for step in range(12):
        state = updater.update(state, State({"s": np.ones(1)}, 0, {"nll": 5.0}),
                               ctx(step=step))
    assert float(state.meta["_plateau_alpha"]) == pytest.approx(0.05)


def test_adaptive_alpha_matches_the_recursion_it_actually_runs():
    """``alpha()`` reported 1/(t+1); the recursion is inverse-sqrt (~1/sqrt(2t))."""
    updater = AdaptiveEMA()
    n_eff, b = updater.prior_count, 32.0
    realised = []
    for _ in range(60):
        a = float(np.clip(b / (b + n_eff), updater.amin, updater.amax))
        realised.append(a)
        n_eff = (1.0 - a) * n_eff + b
    estimated = updater.alpha(ctx(step=59, batch=32))
    assert estimated == pytest.approx(realised[-1], rel=0.10)
    # and the decay really is inverse-sqrt, not 1/t: 1/t would predict 1/60
    assert estimated > 2.0 / 60.0


def test_weighted_mean_defaults_to_unit_weight_for_sum_statistics():
    updater = WeightedMean()
    assert updater.batch_weight(ctx(batch=100)) == 1.0
    # the default law must equal the equal-weight mean even with ragged batches
    a, b = WeightedMean(), RunningMean()
    sa, sb = State({}), State({})
    for step, value in enumerate([2.0, 8.0, 4.0]):
        sa = a.update(sa, stat(s=[value]), ctx(step=step, batch=7 + step))
        sb = b.update(sb, stat(s=[value]), ctx(step=step, batch=7 + step))
    assert np.isclose(sa["s"][0], sb["s"][0])


def test_weighted_mean_batch_size_still_matches_macqueen_for_means():
    updater = WeightedMean("batch_size")
    state = State({})
    state = updater.update(state, stat(s=[10.0]), ctx(batch=10))
    state = updater.update(state, stat(s=[20.0]), ctx(batch=30))
    assert np.isclose(state["s"][0], (10.0 * 10.0 + 30.0 * 20.0) / 40.0)


def test_every_named_schedule_is_constructible_by_name():
    """Five of the six advertised names raised TypeError without alpha0."""
    for name in ("constant", "cosine", "linear", "inverse_sqrt",
                 "inverse_power", "effective_sample", "anneal"):
        schedule = make_schedule(name)
        value = schedule.alpha(ctx(step=3))
        assert np.isfinite(value) and 0.0 <= value <= 1.0


def test_momentum_does_not_shadow_the_alpha_method():
    """``self.alpha`` (float) shadowed ``alpha(ctx)``: updater.alpha(ctx) raised."""
    updater = MomentumEMA(alpha=0.3)
    assert updater.alpha(ctx()) == pytest.approx(0.3)
    assert updater.alpha_at(ctx()) == pytest.approx(0.3)


def test_bias_corrected_keeps_one_copy_and_still_warm_starts():
    updater = BiasCorrectedEMA(alpha=0.1)
    previous = State({"s": np.array([100.0])})
    new = updater.update(previous, stat(s=[3.0]), ctx())
    assert np.allclose(new["s"], [3.0])          # warm state discarded, as before
    assert "_raw" not in new.meta and "_p" not in new.meta
    assert new.meta["_seen"] == {"s"}
    # no second full copy: the state holds the value once, in ``values``
    assert set(new.meta) == {"_seen"}


def test_merge_refuses_stats_measured_under_different_parameters():
    rng = np.random.default_rng(0)
    x = np.vstack([
        rng.normal(-6.0, 1.0, size=(200, 2)),
        rng.normal(0.0, 1.0, size=(200, 2)),
        rng.normal(6.0, 1.0, size=(200, 2)),
    ])
    left = CentroidStat(3, 2, seed=0).prepare(x[:300])
    right = CentroidStat(3, 2, seed=1).prepare(x[300:])
    with pytest.raises(ValueError, match="different parameters"):
        merge_stats([left, right])

    # the same seeding is still mergeable, and the merge is the exact reduction
    a = CentroidStat(3, 2, seed=0).prepare(x[:300])
    b = CentroidStat(3, 2, seed=0).prepare(x[:300])
    merged = merge_stats([a, b])
    assert np.allclose(merged.centroids(), a.centroids())


def test_merge_of_a_parameter_free_stat_is_unaffected():
    from emaopt.state import GramianStat

    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 4))
    a = GramianStat(4).update(x[:100])
    b = GramianStat(4).update(x[100:])
    merged = merge_stats([a, b])
    single = GramianStat(4).update(x)
    assert np.allclose(merged.covariance(), single.covariance(), rtol=1e-12, atol=1e-12)
