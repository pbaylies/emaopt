"""Core utilities and the fit driver."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.core import (
    DataSpec,
    FitConfig,
    State,
    UpdateContext,
    assign,
    logsumexp_rows,
    softmax_rows,
    squared_distances,
)

from conftest import brute_force_sq_dist


def test_squared_distances_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(37, 5))
    c = rng.normal(size=(6, 5))
    got = squared_distances(x, c)
    assert np.allclose(got, brute_force_sq_dist(x, c), atol=1e-10)


def test_squared_distances_non_negative_and_chunked_consistent():
    rng = np.random.default_rng(1)
    x = rng.normal(scale=100.0, size=(501, 3))
    c = rng.normal(scale=100.0, size=(4, 3))
    full = squared_distances(x, c, chunk=10000)
    chunked = squared_distances(x, c, chunk=17)
    assert np.all(full >= 0.0)
    assert np.allclose(full, chunked, atol=1e-9)


def test_assign_returns_nearest():
    x = np.array([[0.0, 0.0], [10.0, 0.0], [5.1, 0.0]])
    c = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert assign(x, c).tolist() == [0, 1, 1]


def test_softmax_and_logsumexp_agree():
    rng = np.random.default_rng(2)
    logits = rng.normal(scale=30.0, size=(13, 7))
    prob = softmax_rows(logits)
    assert np.allclose(prob.sum(axis=1), 1.0)
    assert np.allclose(np.log(prob), logits - logsumexp_rows(logits)[:, None], atol=1e-10)


def test_softmax_handles_extreme_logits():
    logits = np.array([[1e4, -1e4], [-np.inf, 0.0]])
    prob = softmax_rows(logits)
    assert np.all(np.isfinite(prob))
    assert np.allclose(prob.sum(axis=1), 1.0)


def test_state_replace_values_keeps_meta_and_step():
    s = State({"a": np.zeros(2)}, step=4, meta={"k": 1})
    t = s.replace_values(a=np.ones(2))
    assert t.step == 4
    assert t.meta == {"k": 1}
    assert np.allclose(s["a"], 0.0) and np.allclose(t["a"], 1.0)
    assert "a" in s and "b" not in s


def test_update_context_progress_is_bounded():
    ctx = UpdateContext(step=0, batch_size=10, dataset_size=100, total_steps=10)
    assert 0.0 < ctx.progress <= 1.0
    ctx = UpdateContext(step=99, batch_size=10, dataset_size=100, total_steps=10)
    assert ctx.progress == 1.0


def test_dataspec_purely_descriptive():
    spec = DataSpec(n=10, d=3, k=2, seed=1)
    assert spec.n == 10 and spec.k == 2


def test_fit_records_examples_and_objective(small_blobs):
    from emaopt.algorithms.kmeans import LloydKMeans

    model = LloydKMeans(3, niter=5, seed=0)
    result = model.fit(small_blobs.x, niter=5)
    assert result.updates == 5
    assert result.examples_processed == 5 * len(small_blobs.x)
    assert result.extras["batch_size"] == len(small_blobs.x)
    assert result.extras["n"] == len(small_blobs.x)
    assert result.extras["d"] == small_blobs.x.shape[1]
    xs, ys = result.curve()
    assert xs.shape == ys.shape
    assert np.isfinite(result.objective)


def test_fit_config_batch_clamping(small_blobs):
    from emaopt.algorithms.kmeans import KMeans, LloydKMeans

    model = LloydKMeans(3, niter=1, seed=0)
    result = model.fit(small_blobs.x, niter=1)
    assert result.extras["batch_size"] == small_blobs.x.shape[0]
    assert model.resolve_batch_size(FitConfig(batch_size=10_000), 50) == 50
    # a generic k-means clamps to [1, n]; Lloyd always takes the full pass
    generic = KMeans(3)
    assert generic.resolve_batch_size(FitConfig(batch_size=10_000), 50) == 50
    assert generic.resolve_batch_size(FitConfig(batch_size=0), 50) == 1
    assert generic.resolve_batch_size(FitConfig(batch_size=None), 50) == 50


def test_fit_rejects_non_2d_data():
    from emaopt.algorithms.kmeans import LloydKMeans

    with pytest.raises(ValueError):
        LloydKMeans(2).fit(np.arange(10.0), niter=1)
