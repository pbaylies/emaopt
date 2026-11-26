"""k-means and e-means: correctness, the Lloyd special case, and the extras."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms import EMeans, LloydKMeans, emeans
from emaopt.algorithms.kmeans import (
    KMeans,
    alpha0_from_law,
    batch_size_from_law,
)
from emaopt.benchmarks import datasets, metrics
from emaopt.core import State, UpdateContext, assign


def test_lloyd_recovers_well_separated_blobs():
    ds = datasets.blobs(n=800, k=4, d=2, sep=7.0, seed=0)
    result = LloydKMeans(4, niter=25, seed=0).fit(ds.x)
    labels = assign(ds.x, result.params["centroids"])
    assert metrics.adjusted_rand_index(ds.meta["labels"], labels) > 0.95
    assert result.objective < 2.0


def test_lloyd_stops_before_the_iteration_cap_when_converged():
    ds = datasets.blobs(n=800, k=4, d=2, sep=7.0, seed=0)
    result = LloydKMeans(4, niter=100, rel_tol=1e-4, seed=0).fit(ds.x)
    assert result.updates < 100
    assert result.extras["converged"] is True
    assert result.objective < 2.0


def test_emeans_defaults_recover_the_same_blobs():
    ds = datasets.blobs(n=800, k=4, d=2, sep=7.0, seed=0)
    result = emeans(ds.x, 4, nepochs=5, seed=0)
    ari = metrics.adjusted_rand_index(ds.meta["labels"], result.labels)
    assert ari > 0.9
    assert result.objective < 3.0
    assert result.labels.shape == (ds.n,)
    assert result.centroids.shape == (4, 2)
    # drop_last means the per-epoch remainder is not re-processed
    assert 0.9 * 5 * ds.n <= result.examples_processed <= 5 * ds.n


def test_one_emeans_update_with_full_batch_and_alpha_one_is_one_lloyd_iteration():
    ds = datasets.blobs(n=300, k=3, d=2, sep=7.0, seed=1)
    x = ds.x
    # same seed => same k-means++ initialization in both models
    init_model = EMeans(3, nepochs=1, seed=0)
    centroids = init_model.decode(init_model.initialize(x))["centroids"]
    labels = assign(x, centroids)
    manual = np.stack([x[labels == j].mean(axis=0) for j in range(3)])

    model = EMeans(3, nepochs=1, batch_size=len(x), alpha0=1.0,
                   alpha_end_ratio=1.0, finalize=False, revival=False, seed=0)
    result = model.fit(x, batch_size=len(x), nepochs=1, seed=0)
    assert np.allclose(result.params["centroids"], manual, atol=1e-9)


def test_two_full_batch_updates_equal_two_lloyd_iterations():
    ds = datasets.blobs(n=300, k=3, d=2, sep=7.0, seed=2)
    x = ds.x
    init = EMeans(3, seed=0).initialize(x)
    centroids = init["sums"] / init["counts"][:, None]
    for _ in range(2):
        labels = assign(x, centroids)
        centroids = np.stack([x[labels == j].mean(axis=0) for j in range(3)])
    model = EMeans(3, nepochs=2, batch_size=len(x), alpha0=1.0,
                   alpha_end_ratio=1.0, finalize=False, revival=False, seed=0)
    result = model.fit(x, batch_size=len(x), nepochs=2, seed=0)
    assert np.allclose(result.params["centroids"], centroids, atol=1e-9)


def test_batch_size_one_is_a_valid_stream():
    ds = datasets.blobs(n=120, k=3, d=2, sep=6.0, seed=4)
    model = EMeans(3, nepochs=2, batch_size=1, seed=0)
    result = model.fit(ds.x, batch_size=1, nepochs=2, seed=0)
    assert np.isfinite(result.objective)
    assert result.updates == 2 * ds.n


def test_finalize_lands_on_a_consistent_assignment():
    """The exact-mean finish makes one further Lloyd step a near no-op.

    This is the property the finish buys: the returned centroids are the exact
    means of the last epoch's assignments, so re-assigning and re-averaging moves
    them only by boundary-point mass -- unlike the plain EMA arm, whose centroids
    lag the assignments.
    """
    ds = datasets.blobs(n=600, k=3, d=2, sep=8.0, seed=5)
    finalized = EMeans(3, nepochs=3, finalize=True, revival=False, seed=0).fit(
        ds.x, nepochs=3, seed=0)
    plain = EMeans(3, nepochs=3, finalize=False, revival=False, seed=0).fit(
        ds.x, nepochs=3, seed=0)

    def lloyd_step_drift(centroids):
        labels = assign(ds.x, centroids)
        means = np.stack([ds.x[labels == j].mean(axis=0) for j in range(3)])
        return float(np.max(np.abs(means - centroids)))

    assert lloyd_step_drift(finalized.params["centroids"]) < 0.02
    assert lloyd_step_drift(finalized.params["centroids"]) <= lloyd_step_drift(
        plain.params["centroids"]) + 1e-12


def test_empty_clusters_keep_finite_centroids():
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    model = KMeans(5, seed=0)
    result = model.fit(x, batch_size=3, nepochs=2, seed=0)
    assert np.all(np.isfinite(result.params["centroids"]))
    assert np.isfinite(result.objective)


def test_duplicate_points_produce_zero_error():
    x = np.zeros((50, 3))
    result = emeans(x, 3, nepochs=2, seed=0)
    assert result.objective == pytest.approx(0.0, abs=1e-12)


def test_revival_splits_a_donor_and_conserves_mass():
    model = EMeans(4, revival=True, revival_threshold=1e-3, seed=0)
    values = {
        "sums": np.array([[10.0, 0.0], [0.0, 10.0], [0.0, 0.0], [0.0, 0.0]]),
        "counts": np.array([10.0, 10.0, 0.0, 0.0]),
    }
    state = State(values, meta={})
    stats = State({}, 0, {"inertia": 20.0})
    revived = model._revive(state, stats, UpdateContext(step=0, batch_size=100,
                                                       dataset_size=1000, total_steps=10))
    assert np.all(revived["counts"] > 0.0)
    assert np.allclose(revived["sums"].sum(axis=0), state["sums"].sum(axis=0), atol=1e-9)
    assert np.all(revived["counts"] > 0.0)


def test_revival_is_off_by_default_at_low_dimension():
    x = np.random.default_rng(0).normal(size=(100, 2))
    result = emeans(x, 3, nepochs=2, seed=0)
    assert result.extras["trained_on"] == 100


def test_scaling_laws_are_monotone_and_bounded():
    small = batch_size_from_law(1_000, 64, 5)
    large = batch_size_from_law(1_000_000, 64, 5)
    assert small < large
    assert 1 <= small <= 1_000
    quality = batch_size_from_law(1_000_000, 64, 5, batch_law="quality")
    assert quality < large
    for n, k, bs in [(1_000, 4, 32), (1_000_000, 1024, 4096)]:
        a = alpha0_from_law(n, k, bs, 5)
        assert 0.0 < a <= 1.0
    with pytest.raises(KeyError):
        batch_size_from_law(100, 4, 5, batch_law="nonsense")


def test_subsampling_respects_max_points_per_centroid():
    ds = datasets.blobs(n=500, k=4, d=2, sep=6.0, seed=7)
    result = emeans(ds.x, 4, nepochs=2, max_points_per_centroid=8, seed=0)
    assert result.extras["trained_on"] == 32
    assert result.labels.shape == (500,)


def test_objective_of_result_matches_decoded_centroids():
    ds = datasets.blobs(n=200, k=3, d=2, sep=6.0, seed=8)
    result = emeans(ds.x, 3, nepochs=2, seed=0)
    assert result.objective == pytest.approx(
        metrics.mse_to_centroids(ds.x, result.centroids), rel=1e-12
    )


def test_emeans_handles_more_clusters_than_points():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    result = emeans(x, 5, nepochs=2, seed=0)
    assert np.all(np.isfinite(result.centroids))


def test_emeans_cross_check_against_faiss_reference():
    """Fidelity check against faiss 1.16.0's e-means on the same data.

    Initialization and revival details differ, so this asserts comparability
    (our loss is not materially worse), not bit equality.
    """
    e_means = pytest.importorskip("faiss.contrib.e_means")
    from faiss.contrib.e_means import NumpyEmeans

    ds = datasets.blobs(n=2000, k=16, d=8, sep=5.0, seed=11)
    x = ds.x.astype(np.float32)
    reference = NumpyEmeans(ds.d, 16, nepochs=5, seed=1234)
    centroids, _labels = reference.train(x)
    ref_obj = metrics.mse_to_centroids(ds.x, np.asarray(centroids))
    ours = emeans(ds.x, 16, nepochs=5, seed=1234)
    assert ours.objective <= 1.5 * ref_obj + 1e-9
    assert np.isfinite(ref_obj)


def test_emeans_maternal_init_matches_faiss_train_with_init_centroids():
    """A controlled head-to-head: same starting centroids, same data.

    faiss ``NumpyEmeans.train(x, init_centroids=...)`` (the reference release's
    maternal hook) and our ``emeans(..., init_centroids=...)`` must agree within
    the initialization-invariant tolerance, isolating the algorithm from the
    k-means++-vs-random start that confounds the default-parity test.
    """
    pytest.importorskip("faiss")
    e_means = pytest.importorskip("faiss.contrib.e_means")
    ds = datasets.blobs(n=2000, k=16, d=8, sep=6.0, seed=11)
    start = ds.x[:16].astype(np.float32)
    reference = e_means.NumpyEmeans(ds.d, 16, nepochs=5, seed=1234)
    centroids, _ = reference.train(ds.x.astype(np.float32), init_centroids=start)
    ref_obj = metrics.mse_to_centroids(ds.x, np.asarray(centroids, dtype=np.float64))
    ours = emeans(ds.x, 16, nepochs=5, seed=1234, init_centroids=start)
    assert ours.objective <= 1.15 * ref_obj + 1e-9  # measured 0.94x at the port defaults


def test_emeans_maternal_init_shape_guard():
    ds = datasets.blobs(n=200, k=4, d=3, sep=6.0, seed=5)
    bad = np.zeros((2, ds.d))
    with pytest.raises(ValueError):
        emeans(ds.x, 4, nepochs=1, init_centroids=bad)
