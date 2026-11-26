"""PDDP/kM-PDDP seed behaviour and opt-in integration."""

from __future__ import annotations

import numpy as np

from emaopt.algorithms.kmeans import EMeans, KMeans
from emaopt.algorithms.pddp import _one_dimensional_split, pddp_seed
from emaopt.benchmarks.datasets import blobs


def test_pddp_seed_is_deterministic_and_has_requested_shape():
    data = blobs(n=800, k=8, d=4, seed=2).x
    first = pddp_seed(data, 8, seed=3)
    second = pddp_seed(data, 8, seed=3)
    assert first.shape == (8, 4)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)


def test_pddp_seed_k_one_is_the_mean():
    data = blobs(n=200, k=3, d=5, seed=1).x
    got = pddp_seed(data, 1)
    np.testing.assert_allclose(got, data.mean(axis=0, keepdims=True))


def test_one_dimensional_split_separates_two_modes():
    values = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    threshold = _one_dimensional_split(values)
    assert -1.0 < threshold < 1.0
    assert np.array_equal(values <= threshold, [True, True, True, False, False, False])


def test_pddp_seed_pads_degenerate_data():
    data = np.zeros((3, 2), dtype=np.float64)
    got = pddp_seed(data, 5, seed=0)
    assert got.shape == (5, 2)
    assert np.isfinite(got).all()
    np.testing.assert_array_equal(got, np.zeros((5, 2)))

    one = np.array([[2.0, 3.0]])
    padded = pddp_seed(one, 3)
    assert padded.shape == (3, 2)
    np.testing.assert_array_equal(padded, np.repeat(one, 3, axis=0))


def test_pddp_seed_power_path_runs():
    data = blobs(n=400, k=6, d=6, seed=4).x
    got = pddp_seed(data, 6, direction="power", power_iters=4, seed=0)
    assert got.shape == (6, 6)
    assert np.isfinite(got).all()


def test_kmeans_pddp_init_is_opt_in():
    data = blobs(n=500, k=6, d=4, seed=5).x
    model = KMeans(6, init="pddp", seed=7)
    state = model.initialize(data)
    assert state.meta["init"] == "pddp"
    assert state["sums"].shape == (6, 4)

    result = EMeans(6, init="pddp", nepochs=2, seed=7).fit(
        data, nepochs=2, seed=7
    )
    assert result.params["centroids"].shape == (6, 4)
    assert np.isfinite(result.objective)
