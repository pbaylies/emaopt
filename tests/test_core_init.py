"""Cheap coreset seeding for the two core algorithms, pinned where it matters.

The e-means/GMM seed is a cost, not part of the model, so a cheaper seed is only
allowed where it is never worse on held-out quality.

* **e-means** -- ``init="auto"`` stays opt-in.  It keeps k-means++ where the
  coreset seed *loses* (3000 x 8: 1.55 ms against 0.83 and 75.3 held-out MSE
  against 73.5) and takes the coreset once the data is at least eight coresets
  wide.  The 15-cell sweep measured median 1.000x the best arm's MSE against
  1.024x for pinned k-means++, but the pre-committed five-draw study failed D1
  (zipf 1.0167) and D2 (worst 1.5201), so the reference default stays put.
* **eGMM** -- the seed sample is now the coreset rule with one Lloyd restart,
  because that arm was never worse on held-out nll in 15 of 15 seed-shape cells
  (4 strictly better) at 7-12x less init time.
"""

from __future__ import annotations

import numpy as np

from emaopt.algorithms.gmm import GaussianMixture
from emaopt.algorithms.kmeans import EMeans, LloydKMeans
from emaopt.algorithms.smeans import coreset_size


def blobs(n: int, k: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 3.0
    return centres[rng.integers(0, k, size=n)] + rng.normal(size=(n, d))


def test_reference_kmeans_family_defaults_stay_put() -> None:
    """``auto`` is opt-in: the reference arms keep the init they published."""
    assert EMeans(8).init == "kmeans++"
    assert LloydKMeans(8).init == "kmeans++"


def test_auto_init_picks_coreset_only_on_wide_data() -> None:
    """The rule is the measured crossover, not a preference."""
    k = 8
    # 3000 rows is not 8 coresets wide at k=8 (512 rows), and there the coreset
    # seed measured 27% slower *and* 2.5% worse held out.
    small = blobs(3000, k, 4)
    assert EMeans(k, init="auto").initialize(small).meta["init"] == "kmeans++"
    # 20000 rows is 39 coresets wide, where it measured 4.7x cheaper to build
    # with held-out MSE equal within noise.  ``auto`` keeps the plain draw: five
    # guard variants were measured and none bounded its +52% tail, while all of
    # them cost more init (see `init="coreset-guard"`, kept for experiments).
    wide = blobs(20000, k, 4)
    assert EMeans(k, init="auto").initialize(wide).meta["init"] == "coreset"


def test_auto_init_refuses_the_coreset_when_feature_scales_spread() -> None:
    """The cost rule is bounded by a distortion check: aniso is not free."""
    k = 8
    x = blobs(20000, k, 8)
    scaled = x * np.exp(np.random.default_rng(0).normal(scale=1.0, size=8))
    state = EMeans(k, init="auto").initialize(scaled)
    # 20000 rows is 39 coresets wide, so the cost rule alone would choose the
    # coreset; on the aniso sweep cell that measured 3.4% worse held out.
    assert state.meta["init"] == "kmeans++"
    assert state.meta["init_spread"] > 2.0


def test_feature_scale_spread_is_scale_free_and_sample_stable() -> None:
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(20000, 8, 8)
    assert sm.feature_scale_spread(x) < 2.0
    # units cancel: an isotropic set stays isotropic at any scale
    assert abs(sm.feature_scale_spread(x * 1000.0)
               - sm.feature_scale_spread(x)) < 1e-9
    scaled = x * np.exp(np.random.default_rng(2).normal(scale=1.0, size=8))
    full = sm.feature_scale_spread(scaled, sample=10**9)
    sampled = sm.feature_scale_spread(scaled)
    assert full > 2.0
    assert abs(sampled - full) < 0.25 * full


def test_auto_init_respects_an_explicit_coreset_size() -> None:
    """A caller's ``init_coreset`` moves the crossover with it."""
    k = 8
    small = blobs(3000, k, 4)
    # 3000 >= 8 * 256, so a 256-row coreset is wide enough to be chosen.
    model = EMeans(k, init="auto", init_coreset=256)
    assert model.initialize(small).meta["init"] == "coreset"


def test_auto_init_never_changes_the_centroid_count() -> None:
    model = EMeans(8, init="auto")
    state = model.initialize(blobs(20000, 8, 4))
    assert state["sums"].shape == (8, 4)


def test_multi_draw_seed_keeps_the_best_candidate(monkeypatch) -> None:
    """The draw is the noisy part of a coreset seed, so judge several on the data."""
    import importlib

    # the package re-exports a ``smeans`` function that shadows the module
    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(4000, 4, 2, seed=3)
    rng = np.random.default_rng(1)
    good = x[rng.choice(x.shape[0], 4, replace=False)]
    draws = iter([good + 25.0, good])
    monkeypatch.setattr(sm, "coreset_lloyd_seed", lambda *a, **k: next(draws))
    out = sm.multi_coreset_seed(x, 4, np.random.default_rng(0), draws=2)
    assert np.array_equal(out, good)


def test_guarded_seed_keeps_the_candidate_that_fits_the_check_slice(monkeypatch) -> None:
    """The guard is a held-out-slice comparison, not a preference for draw 0."""
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(4000, 4, 2, seed=3)
    rng = np.random.default_rng(1)
    good = x[rng.choice(x.shape[0], 4, replace=False)]
    draws = iter([good + 40.0, good])
    monkeypatch.setattr(sm, "coreset_lloyd_seed", lambda *a, **k: next(draws))
    out = sm.guarded_coreset_seed(x, 4, np.random.default_rng(0),
                                  draws=2, check_rows=512)
    assert np.allclose(out, good)


def test_guard_rejects_a_seed_with_starved_components(monkeypatch) -> None:
    """The fallback is what bounds the tail: a merged-cluster seed is refused."""
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(4000, 4, 2, seed=3)
    monkeypatch.setattr(sm, "coreset_lloyd_seed",
                        lambda *a, **k: np.zeros((4, 2)))
    # Four identical centres collapse every row onto component 0, so the other
    # three take none of their equal share.
    _, ok = sm.guarded_coreset_seed(
        x, 4, np.random.default_rng(0), draws=1, check_rows=512,
        refine_niter=0, verify=True,
    )
    assert ok is False


def test_auto_init_falls_back_when_the_guard_rejects(monkeypatch) -> None:
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    monkeypatch.setattr(
        sm, "guarded_coreset_seed",
        lambda data, k, rng, **kw: (np.zeros((k, data.shape[1])), False),
    )
    state = EMeans(8, init="coreset-guard").initialize(blobs(20000, 8, 4))
    assert state.meta["init"] == "kmeans++"


def test_auto_init_uses_the_guarded_draw(monkeypatch) -> None:
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    seen: dict = {}
    real = sm.guarded_coreset_seed

    def spy(data, k, rng, **kwargs):
        seen.update(kwargs)
        return real(data, k, rng, **kwargs)

    monkeypatch.setattr(sm, "guarded_coreset_seed", spy)
    state = EMeans(8, init="coreset-guard").initialize(blobs(20000, 8, 4))
    assert state.meta["init"] == "coreset-guard"
    # Two candidates by default: one uniform draw, one stratified.
    assert seen.get("draws", 0) >= 2


def test_init_draws_reaches_the_multi_draw_seed(monkeypatch) -> None:
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    seen: dict = {}
    real = sm.multi_coreset_seed

    def spy(data, k, rng, **kwargs):
        seen.update(kwargs)
        return real(data, k, rng, **kwargs)

    monkeypatch.setattr(sm, "multi_coreset_seed", spy)
    model = EMeans(8, init="coreset", init_draws=3)
    assert model.initialize(blobs(4000, 8, 4)).meta is not None
    assert seen.get("draws") == 3


def test_init_draws_one_keeps_the_single_draw_path(monkeypatch) -> None:
    """The default must stay bit-for-bit the shipped coreset seed."""
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    called = {"multi": 0}
    monkeypatch.setattr(
        sm, "multi_coreset_seed",
        lambda *a, **k: called.__setitem__("multi", called["multi"] + 1),
    )
    model = EMeans(8, init="coreset")
    model.initialize(blobs(4000, 8, 4))
    assert called["multi"] == 0
    assert model.init_draws == 1


def test_gmm_seed_uses_the_coreset_size_rule() -> None:
    k = 16
    x = blobs(20000, k, 8)
    model = GaussianMixture(k, covariance="diag", seed=0)
    state = model.initialize(x)
    assert state.meta["seed_rows"] == coreset_size(20000, k) < 20000
    # One restart: extra restarts re-optimise the *seed*, which measured never
    # better on held-out nll for 7-12x the init cost.
    assert model.n_init == 1


def test_gmm_seed_rows_are_capped_by_explicit_init_sample() -> None:
    k = 8
    x = blobs(20000, k, 8)
    model = GaussianMixture(k, covariance="diag", seed=0, init_sample=1000)
    assert model.initialize(x).meta["seed_rows"] == 1000


def test_gmm_seed_still_sees_small_data_whole() -> None:
    """``coreset_size`` clamps at n, so a tiny fit keeps every row."""
    k = 4
    x = blobs(300, k, 4)
    model = GaussianMixture(k, covariance="diag", seed=0)
    assert model.initialize(x).meta["seed_rows"] == 300
