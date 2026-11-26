"""The s-means recipe pieces: mechanics, wiring, and one quality check."""

import numpy as np
import pytest

from emaopt.algorithms.kmeans import EMeans
from emaopt.algorithms.smeans import (
    CORESET_MIN,
    PowerCosineSchedule,
    SMeans,
    _d15_weights,
    anisotropic_split,
    coreset_size,
    d15_seed,
    shard_refine_seed,
    shrink_final,
    stratified_sample,
)
from emaopt.core import UpdateContext


def blobs(n=3000, k=8, d=8, sep=4.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=sep, size=(k, d))
    labels = rng.integers(k, size=n)
    return centres[labels] + rng.normal(size=(n, d))


def test_d15_seeding_weights_are_the_distance_to_the_1_5_power():
    squared = np.array([4.0, 9.0])
    weights = _d15_weights(squared)
    # d^1.5 on distances == (d^2)^0.75 on the squared distances we hold
    assert np.allclose(weights, np.array([2.0, 3.0]) ** 1.5)


def test_d15_seed_draws_in_batched_rounds(monkeypatch):
    import importlib

    # ``emaopt.algorithms.smeans`` is the convenience function, so reach the
    # module the same way Python does.
    sm = importlib.import_module("emaopt.algorithms.smeans")

    calls = {"n": 0}
    real = sm.squared_distances

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sm, "squared_distances", counted)
    x = blobs(n=4000, k=16, d=8)
    centres = d15_seed(x, 16, np.random.default_rng(0), rounds=4, greedy=False)
    assert centres.shape == (16, 8)
    # one distance pass for the first centre, then one batched refresh per round
    assert calls["n"] == 5


def test_d15_seed_greedy_tests_candidates_per_centre(monkeypatch):
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    calls = {"n": 0}
    real = sm.squared_distances

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(sm, "squared_distances", counted)
    x = blobs(n=4000, k=16, d=8)
    centres = d15_seed(x, 16, np.random.default_rng(0), greedy=True)
    assert centres.shape == (16, 8)
    # first centre plus one candidate-distance pass per remaining centre
    assert calls["n"] == 16


def test_d15_seed_uses_the_16k_1024_coreset_floor(monkeypatch):
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")

    seen = {}
    real = sm.squared_distances

    def spy(sample, centres, **kwargs):
        seen["rows"] = np.asarray(sample).shape[0]
        return real(sample, centres, **kwargs)

    monkeypatch.setattr(sm, "squared_distances", spy)
    x = blobs(n=6000, k=8, d=4)  # 16k = 128, so the 1,024 floor is the coreset size
    d15_seed(x, 8, np.random.default_rng(0))
    assert seen["rows"] == 1024


def test_shrinkage_moves_low_count_clusters_toward_the_smoothed_centroid():
    mean = np.array([[10.0, 10.0], [1.0, 1.0]])
    smoothed = np.zeros((2, 2))
    counts = np.array([1.0, 100.0])
    out = shrink_final(mean, smoothed, counts, tau=0.20)
    assert np.allclose(out[0], mean[0] * (1.0 / 1.2))
    assert np.allclose(out[1], mean[1] * (100.0 / 100.2))
    # the retained fraction, not the raw coordinate, is what shrinkage controls
    assert out[0, 0] / mean[0, 0] < out[1, 0] / mean[1, 0]


def test_anisotropic_split_is_mean_preserving_and_follows_the_feature_scale():
    rng = np.random.default_rng(3)
    mean = np.array([5.0, 5.0, 5.0])
    scale = np.array([2.0, 2.0, 0.002])
    a, b = anisotropic_split(mean, 100.0, 4.0, scale, rng)
    assert np.allclose((a + b) / 2.0, mean)
    delta = np.abs(a - b)
    # the axis follows the data's anisotropy, not an isotropic direction
    assert delta[0] > 20.0 * delta[2]


def test_anisotropic_split_keeps_the_isotropic_magnitude_convention():
    # The published per-feature form has norm sqrt(d) * radius; this port
    # normalizes to sqrt(distortion / (count * d)) like the shipped split.
    mean = np.zeros(8)
    count, distortion = 100.0, 800.0
    a, b = anisotropic_split(mean, count, distortion, np.ones(8),
                            np.random.default_rng(0))
    assert np.isclose(np.linalg.norm(a - mean),
                      np.sqrt(distortion / (count * 8)), rtol=1e-12)


def test_power_cosine_decays_from_alpha0_toward_its_floor():
    sched = PowerCosineSchedule(0.4)

    def at(step):
        return sched.alpha(
            UpdateContext(step=step, batch_size=8, dataset_size=64, total_steps=10)
        )

    first, last = at(0), at(9)
    assert first <= 0.4 and first > 0.38
    assert last < first
    assert last >= 0.4 * sched.floor
    grid = sched.alpha_grid(10)
    assert np.allclose(grid, [at(step) for step in range(10)])


def test_smeans_wires_the_recipe_and_the_reference_defaults_stay_put():
    ours, ref = SMeans(8), EMeans(8)
    assert ours.init == "d1.5"
    assert ours.revival_geometry == "anisotropic"
    assert ours.finalize_shrink == pytest.approx(0.20)
    assert ref.init == "kmeans++"
    assert ref.revival_geometry == "isotropic"
    assert ref.finalize_shrink == pytest.approx(0.0)
    # Anisotropic revival is measured-safe only at small k.
    assert SMeans(128).revival_geometry == "isotropic"


def test_coreset_lloyd_seed_clusters_only_the_coreset(monkeypatch):
    import importlib

    import emaopt.algorithms.kmeans as km

    seen = {"max_rows": 0}
    real = km.squared_distances

    def spy(sample, centres, **kwargs):
        seen["max_rows"] = max(seen["max_rows"], np.asarray(sample).shape[0])
        return real(sample, centres, **kwargs)

    monkeypatch.setattr(km, "squared_distances", spy)
    import emaopt.algorithms.smeans as sm

    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(n=20000, k=64, d=8)
    centres = sm.coreset_lloyd_seed(x, 64, np.random.default_rng(0))
    assert centres.shape == (64, 8)
    # Lloyd runs on the coreset the size rule prescribes, never on the 20k rows
    assert 0 < seen["max_rows"] <= coreset_size(20000, 64)
    assert seen["max_rows"] < x.shape[0]


def test_shard_union_seed_combines_disjoint_shards_into_k_centres():
    import importlib

    sm = importlib.import_module("emaopt.algorithms.smeans")
    x = blobs(n=20000, k=32, d=8)
    centres = sm.shard_union_seed(x, 32, np.random.default_rng(0), shards=4)
    assert centres.shape == (32, 8)
    again = sm.shard_union_seed(x, 32, np.random.default_rng(0), shards=4)
    assert np.allclose(centres, again)


def test_coreset_size_is_rows_per_cluster_with_a_floor_and_a_cap():
    assert coreset_size(20000, 64) == 2048     # 32 rows per cluster
    assert coreset_size(20000, 8) == CORESET_MIN  # floor wins below 16 rows/cluster
    assert coreset_size(300, 64) == 300        # never more rows than exist


def test_stratified_sample_is_deterministic_and_keeps_the_requested_size():
    x = blobs(n=5000, k=4, d=4)
    a = stratified_sample(x, 512, np.random.default_rng(0))
    b = stratified_sample(x, 512, np.random.default_rng(0))
    assert a.shape == (512, 4)
    assert np.allclose(a, b)


def test_shard_refine_seed_returns_k_centres():
    x = blobs(n=20000, k=32, d=8)
    centres = shard_refine_seed(x, 32, np.random.default_rng(0), shards=4,
                               coreset=max(4 * 32, CORESET_MIN), combine="fps")
    assert centres.shape == (32, 8)


def test_coreset_size_is_selectable():
    # A 4x coreset measured slightly better on average than the default m;
    # the knob exists so that result is reachable without a private call.
    model = EMeans(8, init="coreset", init_coreset=2048)
    assert model.init_coreset == 2048


def test_smeans_preset_beats_the_reference_recipe_at_small_n():
    # Small-n pin.  The old 20k x 32 failure was two separate bugs: a 512-row
    # correlated D^1.5 draw (now greedy, 16 rows/centre, 1,024-row floor) and
    # anisotropic revival above k=64 (now falls back to isotropic).  The
    # corrected family sweep is future work.
    x = blobs(n=3000, k=16, d=8)
    ours = SMeans(16, nepochs=5, seed=0).fit(x, nepochs=5, seed=0)
    ref = EMeans(16, nepochs=5, seed=0).fit(x, nepochs=5, seed=0)
    assert ours.objective <= ref.objective * 0.90
