"""SufficientStat protocol, merge algebra, and the optimizer retrofit surface."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emaopt.algorithms import GaussianMixture, KMeans, VectorQuantizer
from emaopt.benchmarks import datasets
from emaopt.state.sufficient import (
    CentroidStat,
    GramianStat,
    OptimizerStat,
    merge_stats,
)
from emaopt.stream import partitioned_stat, stat_fit, stat_from_blocks


def test_centroid_stat_merge_is_exact_for_additive_accumulation():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(size=(200, 4)) + 4.0,
                        rng.normal(size=(200, 4)) - 4.0])
    centroids = np.array([[4.0, 4.0, 4.0, 4.0], [-4.0, -4.0, -4.0, -4.0]])

    single = CentroidStat(2, centroids=centroids, mode="sum", prior=0.0)
    for block in np.array_split(x, 7):
        single.update(block)

    parts = []
    for partition in np.array_split(x, 4):
        stat = CentroidStat(2, centroids=centroids, mode="sum", prior=0.0)
        for block in np.array_split(partition, 3):
            stat.update(block)
        parts.append(stat)
    merged = merge_stats(parts)

    assert np.allclose(single.values["sums"], merged.values["sums"], atol=1e-10)
    assert np.allclose(single.values["counts"], merged.values["counts"], atol=1e-10)
    assert np.allclose(single.decode()["centroids"], merged.decode()["centroids"],
                       atol=1e-12)


def test_partitioned_stat_equals_single_stream_for_centroids_and_gramian():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 5))
    centroids = x[rng.choice(600, size=3, replace=False)]

    frozen = 10 ** 9  # no acts: every partition measures against the same centroids
    make_centroid = lambda: CentroidStat(3, centroids=centroids, mode="sum", prior=0.0,
                                         act_every=frozen)
    single = make_centroid()
    for block in np.array_split(x, 6):
        single.update(block)
    merged = partitioned_stat(make_centroid, x, n_partitions=3, block_size=50)
    assert np.allclose(single.values["sums"], merged.values["sums"], atol=1e-10)

    make_gram = lambda: GramianStat(center=True)
    single_g = make_gram()
    for block in np.array_split(x, 6):
        single_g.update(block)
    merged_g = partitioned_stat(make_gram, x, n_partitions=3, block_size=50)
    assert np.allclose(single_g.values["gram"], merged_g.values["gram"], atol=1e-9)
    assert merged_g.n_examples == single_g.n_examples == x.shape[0]
    assert np.allclose(single_g.decode()["components"], merged_g.decode()["components"],
                       atol=1e-10)


def test_partitioned_stat_with_threads_matches_the_sequential_reduction():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(800, 4))
    make = lambda: GramianStat(center=True)
    sequential = partitioned_stat(make, x, n_partitions=4, block_size=100, workers=1)
    threaded = partitioned_stat(make, x, n_partitions=4, block_size=100, workers=2)
    assert np.allclose(sequential.values["gram"], threaded.values["gram"], atol=1e-9)
    assert threaded.n_examples == x.shape[0]


def test_gramian_stat_rejects_a_block_of_the_wrong_width():
    with pytest.raises(ValueError, match="block width 5"):
        GramianStat(d=4).update(np.ones((10, 5)))


def test_permutation_invariance_of_a_sum_stat():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 6))
    order = rng.permutation(500)
    a = GramianStat(center=True)
    b = GramianStat(center=True)
    for block in np.array_split(x, 5):
        a.update(block)
    for block in np.array_split(x[order], 5):
        b.update(block)
    assert np.allclose(a.values["gram"], b.values["gram"], rtol=1e-12, atol=1e-12)


def test_ema_fold_is_the_decayed_average_and_merges_as_a_weighted_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 3))
    stat = GramianStat(center=False, mode="ema", alpha=0.25)
    blocks = np.array_split(x, 4)
    expected = None
    for block in blocks:
        gram = block.T @ block
        expected = gram if expected is None else 0.75 * expected + 0.25 * gram
        stat.update(block)
    assert np.allclose(stat.values["gram"], expected, rtol=1e-12, atol=1e-12)

    left = GramianStat(center=False, mode="ema", alpha=0.25)
    left.update(blocks[0])
    left.update(blocks[1])
    right = GramianStat(center=False, mode="ema", alpha=0.25)
    right.update(blocks[2])
    right.update(blocks[3])
    merged = left.merge(right)
    # decayed sums and their mass add: the merged decode is the mass-weighted
    # combination, which is the decay correction for an additive stat
    assert np.allclose(merged.values["gram"], left.values["gram"] + right.values["gram"],
                       rtol=1e-12, atol=1e-12)
    assert np.isclose(merged.weight, left.weight + right.weight, rtol=1e-12)
    assert merged.steps == 4


def test_stat_config_is_json_serialisable_and_reports_bytes():
    stat = CentroidStat(4, centroids=np.zeros((4, 3)), prior=0.0)
    stat.update(np.ones((10, 3)))
    payload = json.dumps(stat.config())
    assert "centroid_stat" in payload
    assert stat.config()["nbytes"] == stat.nbytes
    assert stat.nbytes >= 4 * 3 * 8 + 4 * 8
    stats = stat.memory_report()
    assert stats["total_bytes"] == stat.nbytes


def test_reducing_a_stat_matches_sequential_partitions():
    """`merge` is associative up to float summation order, not order-dependent."""
    rng = np.random.default_rng(8)
    x = rng.normal(size=(600, 5))
    make = lambda: GramianStat(center=False)
    left = partitioned_stat(make, x, n_partitions=2, block_size=100)
    first = stat_from_blocks(make, np.array_split(x, 4)[:2])
    second = stat_from_blocks(make, np.array_split(x, 4)[2:])
    merged = first.merge(second)
    assert np.allclose(left.values["gram"], merged.values["gram"], atol=1e-9)


def test_stats_reject_incompatible_merges():
    a = CentroidStat(2, centroids=np.zeros((2, 3)), prior=0.0)
    b = GramianStat()
    with pytest.raises(TypeError):
        a.merge(b)


# --------------------------------------------------------------------------- #
# phase 4 retrofit surface: an existing optimizer as a stat
# --------------------------------------------------------------------------- #
def test_optimizer_stat_pins_kmeans_replace():
    ds = datasets.blobs(n=250, k=3, d=2, sep=6.0, seed=3)
    reference = KMeans(3, seed=0).fit(ds.x, batch_size=None, nepochs=1)
    stat = OptimizerStat(KMeans(3, seed=0), mode="ema", alpha=1.0)
    result = stat_fit(stat, ds.x, block_size=None, prepare=ds.x)
    assert np.allclose(result.params["centroids"], reference.params["centroids"],
                       rtol=1e-12, atol=1e-12)
    assert np.array_equal(result.params["counts"], reference.params["counts"])


def test_optimizer_stat_pins_diag_mixture():
    ds = datasets.gaussian_mixture(n=400, k=3, d=3, covariance="diag", seed=4)
    reference = GaussianMixture(3, covariance="diag", seed=0).fit(
        ds.x, batch_size=None, nepochs=1
    )
    stat = OptimizerStat(GaussianMixture(3, covariance="diag", seed=0),
                         mode="ema", alpha=1.0)
    result = stat_fit(stat, ds.x, block_size=None, prepare=ds.x)
    assert np.allclose(result.params["means"], reference.params["means"], atol=1e-12)
    assert np.allclose(result.params["covariances"], reference.params["covariances"],
                       atol=1e-12)
    assert np.allclose(result.params["weights"], reference.params["weights"], atol=1e-12)


def test_optimizer_stat_pins_the_quantizer():
    ds = datasets.blobs(n=200, k=4, d=3, sep=5.0, seed=5)
    reference = VectorQuantizer(4, seed=0).fit(ds.x, batch_size=None, nepochs=1)
    stat = OptimizerStat(VectorQuantizer(4, seed=0), mode="ema", alpha=1.0)
    result = stat_fit(stat, ds.x, block_size=None, prepare=ds.x)
    assert np.allclose(result.params["centroids"], reference.params["centroids"],
                       atol=1e-12)


def test_reset_accumulation_keeps_parameters_and_clears_moments():
    stat = CentroidStat(2, centroids=np.zeros((2, 3)), prior=0.0)
    stat.update(np.ones((10, 3)))
    before = stat.decode()["centroids"].copy()
    stat.reset_accumulation()
    assert stat.values == {} and stat.steps == 0 and stat.n_examples == 0
    np.testing.assert_array_equal(stat.decode()["centroids"], before)


def test_optimizer_stat_frozen_params_takes_one_decode_per_epoch():
    ds = datasets.gaussian_mixture(n=600, k=3, d=3, covariance="diag", seed=7)
    model = GaussianMixture(3, covariance="diag", seed=0)
    stat = OptimizerStat(model, mode="sum", act_every=10 ** 9, freeze_params=True)
    stat.prepare(ds.x[:200])
    calls = {"decode": 0, "statistics_from": 0}
    original_decode = model.decode
    original_stats = model.statistics_from

    def counted_decode(state):
        calls["decode"] += 1
        return original_decode(state)

    def counted_stats(x, params):
        calls["statistics_from"] += 1
        return original_stats(x, params)

    model.decode = counted_decode
    model.statistics_from = counted_stats
    before = stat.decode()["means"].copy()
    calls["decode"] = 0
    stat.reset_accumulation()
    for block in np.array_split(ds.x, 3):
        stat.update(block)
    assert calls["decode"] == 0  # frozen: no M-step while accumulating
    stat.act()
    assert calls["statistics_from"] == 3
    assert calls["decode"] == 1  # the act, not once per block
    assert stat.steps == 3
    assert np.isfinite(stat.decode()["means"]).all()
    assert before.shape == stat.decode()["means"].shape


def test_optimizer_stat_epoch_mode_matches_one_full_batch_step():
    """A whole pass of additive statistics must equal one full-data E-step."""
    ds = datasets.gaussian_mixture(n=600, k=3, d=3, covariance="tied", seed=8)
    model = GaussianMixture(3, covariance="tied", seed=0)
    blocks = np.array_split(ds.x, 4)
    stat = OptimizerStat(model, mode="sum", act_every=len(blocks),
                         freeze_params=True)
    stat.prepare(ds.x)
    initial = stat.decode()
    stat.reset_accumulation()
    for block in blocks:
        stat.update(block)

    expected = model.statistics_from(ds.x, initial)
    for key, value in expected.values.items():
        np.testing.assert_allclose(stat.values[key], value, rtol=1e-12, atol=1e-12)
    got = stat.decode()
    want = model.decode(expected)
    np.testing.assert_allclose(got["means"], want["means"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["covariances"], want["covariances"],
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got["weights"], want["weights"], rtol=1e-12, atol=1e-12)
    assert stat.steps == len(blocks)


def test_optimizer_stat_streams_with_an_ema_law():
    ds = datasets.blobs(n=600, k=3, d=2, sep=7.0, seed=6)
    stat = OptimizerStat(KMeans(3, seed=0), mode="ema", alpha=0.2)
    result = stat_fit(stat, ds.x, block_size=64, prepare=ds.x)
    assert result.updates == 10  # ceil(600/64) blocks in one epoch
    assert result.examples_processed == 600
    assert np.isfinite(result.params["centroids"]).all()
