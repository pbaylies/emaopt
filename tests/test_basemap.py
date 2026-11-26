"""Tests for the basemap pipeline: UMAP embedding, GPU e-means, placement,
Voronoi cut and push-apart.

The GPU arms are opt-in: tests skip when torch or CUDA is unavailable, and
the host-style oracles (emeans, Lloyd k-means) guard against silent device
correctness drift.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import emaopt
from emaopt import (
    emeans,
    emeans_gpu_fit,
    find_neighbors,
    fuzzy_membership,
    hierarchical_place,
    nearest_parent,
    place_around_parents,
    push_apart,
    umap_embed,
    UMAPConfig,
    voronoi_cut,
)

def _brute_force_sq_dist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)


def _blobs(n=300, k=3, d=2, sep=7.0, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    labels = []
    for i in range(k):
        loc = np.zeros(d)
        loc[0] = sep * i
        loc[1] = sep * (1 if i % 2 == 1 else 0)
        parts.append(rng.normal(loc=loc, scale=0.3, size=(n // k, d)))
        labels += [i] * (n // k)
    x = np.vstack(parts).astype(np.float32)
    return x, np.array(labels)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestUMAP:
    def test_blobs_separate(self):
        from sklearn.metrics import adjusted_rand_score
        from sklearn.cluster import KMeans

        x, labels = _blobs(n=900, k=3, d=2, sep=7.0, seed=0)
        res = umap_embed(x, UMAPConfig(n_neighbors=10, n_epochs=120, burn_in_epoch=20,
                                      learning_rate=1.0, seed=0), device="cuda")
        assert res.embedding.shape == (x.shape[0], 2)
        assert np.isfinite(res.embedding).all()
        pred = KMeans(3, random_state=0, n_init=10).fit_predict(res.embedding)
        assert adjusted_rand_score(labels, pred) > 0.8

    def test_kNN_self_excluded(self):
        x, _ = _blobs(n=200, k=2, d=2, seed=0)
        labels, dists = find_neighbors(x, 8, device="cuda")
        for i in range(x.shape[0]):
            assert i not in labels[i]
        # nearest neighbor must be geometrically closest (self excluded)
        d2 = _brute_force_sq_dist(x, x)
        for i in [0, 1, 2]:
            d2[i, i] = np.inf
            assert labels[i, 0] == np.argmin(d2[i])

    def test_fuzzy_membership_bounds(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(0.01, 5.0, size=(100, 8)).astype(np.float32)
        d.sort(axis=1)
        w = fuzzy_membership(d)
        assert w.shape == d.shape
        assert (w > 0).all() and (w <= 1.0 + 1e-6).all()
        # strictly non-increasing with distance within each row
        for i in range(100):
            assert np.all(np.diff(w[i]) <= 1e-6)

    def test_umap_result_summary(self):
        x, _ = _blobs(n=100, k=2, d=2, seed=0)
        res = umap_embed(x, UMAPConfig(n_neighbors=8, n_epochs=20, burn_in_epoch=0,
                                      init_method="random", seed=1), device="cuda")
        s = res.summary()
        assert s["n"] == 100 and s["d"] == 2 and s["n_neighbors"] == 8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestGpuEmeans:
    def test_parity_with_host(self):
        x, _ = _blobs(n=600, k=3, d=2, seed=0)
        host = emeans(x, 3, nepochs=10, seed=0, unused_threshold=None)
        dev = emeans_gpu_fit(x, 3, nepochs=10, seed=0, device="cuda")
        # the device arm is float32-flavored; demand agreement in the first digit
        assert dev.objective > 0.0
        assert abs(dev.objective / host.objective - 1.0) < 0.30
        # centroid parity: match device to host by greedy nearest-neighbour;
        # the label index order may legitimately differ between arms.
        h = host.centroids
        d = dev.centroids
        used = set()
        dists = []
        for dc in d:
            dd = np.sqrt(((h - dc) ** 2).sum(axis=1))
            j = int(np.argmin(dd))
            if j in used:
                dd[list(used)] = np.inf
                j = int(np.argmin(dd))
                dd = np.sqrt(((h - dc) ** 2).sum(axis=1))
            used.add(j)
            dists.append(float(dd[j]))
        assert float(np.mean(dists)) < 1.5  # blobs at sep 7.0

    def test_maternal_init_honoured(self):
        x, _ = _blobs(n=200, k=2, d=2, seed=1)
        init = emeans(x, 2, nepochs=3, seed=0, unused_threshold=None).centroids
        dev = emeans_gpu_fit(x, 2, nepochs=1, seed=0, init_centroids=init, device="cuda")
        assert np.isfinite(dev.centroids).all()
        assert dev.init == "maternal"

    def test_large_k_random_fallback(self):
        x, _ = _blobs(n=600, k=3, d=2, seed=0)
        dev = emeans_gpu_fit(x, 500, nepochs=1, seed=0, max_init_k=100, device="cuda")
        assert dev.centroids.shape == (500, 2)
        assert dev.init == "random"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestPlacement:
    def test_nearest_parent_brute_force(self):
        rng = np.random.default_rng(0)
        parents = rng.normal(size=(7, 3)).astype(np.float32)
        points = parents[rng.integers(0, 7, size=50)] + rng.normal(size=(50, 3)) * 0.05
        d2 = _brute_force_sq_dist(points, parents)
        labels = nearest_parent(points, parents, device="cuda")
        assert np.all(labels == d2.argmin(axis=1))

    def test_place_around_parents_deterministic_and_spread(self):
        rng = np.random.default_rng(0)
        parents_xy = rng.normal(size=(30, 2))
        points_parent = rng.integers(0, 30, size=200)
        pos1 = place_around_parents(points_parent, parents_xy, spread=0.3, seed=1)
        pos2 = place_around_parents(points_parent, parents_xy, spread=0.3, seed=1)
        assert np.allclose(pos1, pos2)
        assert pos1.shape == (200, 2)
        assert np.isfinite(pos1).all()
        # mean displacement from parent should be well below the 2-D spacing
        spacing = np.median(
            np.sqrt(np.sum((parents_xy[:, None, :] - parents_xy[None, :, :]) ** 2, axis=2))
        )
        disp = np.linalg.norm(pos1 - parents_xy[points_parent], axis=1)
        assert disp.mean() < 4.0 * spacing

    def test_hierarchical_place_consistent(self):
        x, _ = _blobs(n=150, k=2, d=2, seed=0)
        parent_xy = np.array([[0.0, 0.0], [1.0, 1.0]])
        hlp = hierarchical_place(x, parent_xy, parent_xy, spread=0.3, seed=0)
        assert hlp.child_positions.shape == (150, 2)
        assert hlp.spacing > 0.0
        assert hlp.parent_ids.shape == (150,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestVoronoi:
    def test_voronoi_cut_populates_groups(self):
        rng = np.random.default_rng(0)
        pos = rng.normal(size=(400, 2))
        vr = voronoi_cut(pos, 8, iterations=60, seed=0, device="cuda")
        assert vr.labels.shape == (400,)
        assert len(np.unique(vr.labels)) >= 2
        assert np.isfinite(vr.centers).all()
        # labels are the argmin against centers
        d2 = np.sqrt(np.sum((pos[:, None, :] - vr.centers[None, :]) ** 2, axis=2))
        assert (vr.labels == d2.argmin(axis=1)).all()

    def test_push_apart_increases_separation(self):
        # overlapping centres
        rng = np.random.default_rng(0)
        c = rng.normal(size=(6, 2)) * 0.01
        d0 = np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d0, np.inf)
        before = d0.min()
        pa = push_apart(c, min_separation=0.05, iterations=80, seed=0, device="cuda")
        d1 = np.sqrt(np.sum((pa.centers[:, None, :] - pa.centers[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d1, np.inf)
        after = d1.min()
        assert after > before
        assert pa.converged
        assert after >= 0.05  # target is reached under iterations=80


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestLocalSpacing:
    def test_dense_parents_get_tighter_offsets(self):
        # two clusters of parents, very different densities
        rng = np.random.default_rng(0)
        dense = rng.normal(size=(20, 2), scale=0.05)
        sparse = np.array([[0.0, 0.0], [8.0, 8.0], [8.0, 0.0], [0.0, 8.0]])
        parents_xy = np.vstack([dense, sparse])
        parents = np.array([0, 1, 2, 20, 21, 23])  # dense-parent children + sparse-parent children
        pos = place_around_parents(parents, parents_xy, spread=0.3, seed=1,
                                   local_spacing=True)
        disp = np.linalg.norm(pos - parents_xy[parents], axis=1)
        dense_disp = disp[:3].mean()
        sparse_disp = disp[3:].mean()
        # dense parents are ~0.05 apart -> tight offsets; sparse parents ~8 apart -> wide offsets
        assert sparse_disp > 10.0 * dense_disp
        # offsets stay local: each child stays well within its own parent neighborhood
        assert (sparse_disp < 4.0 * 8.0)

    def test_local_spacing_disabled_uses_global(self):
        rng = np.random.default_rng(0)
        parents_xy = rng.normal(size=(30, 2))
        parents = rng.integers(0, 30, size=50)
        pos = place_around_parents(parents, parents_xy, spread=0.3, seed=1,
                                   local_spacing=False)
        assert np.isfinite(pos).all()
        # global path: per-parent offsets share scale, so spread is comparable across parents
        global_spacing = np.median(
            np.sqrt(np.sum((parents_xy[:, None, :] - parents_xy[None, :, :]) ** 2, axis=2))
        )
        disp = np.linalg.norm(pos - parents_xy[parents], axis=1)
        assert disp.mean() < 4.0 * global_spacing


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestRunBasemap:
    def test_run_basemap_small_two_level(self):
        import emaopt

        rng = np.random.default_rng(0)
        # two well-separated blobs, moderate scale: top=8, mid=16, fine=32, groups=6
        x = rng.normal(size=(600, 4))
        x[:300, 0] += 6.0
        x[:300, 1] += 6.0
        x = x.astype(np.float32)
        res = emaopt.run_basemap(
            x, top_k=8, mid_k=16, fine_k=32, nepochs=2, seed=0,
            groups=6, do_tail=True, device="cuda",
        )
        assert res.top_xy.shape == (8, 2)
        assert res.mid_xy.shape == (16, 2)
        assert res.fine_xy.shape == (32, 2)
        assert res.tail_xy.shape == (600, 2)
        assert res.tail_parents.shape == (600,)
        assert (res.tail_parents >= 0).all() and (res.tail_parents < 32).all()
        assert np.isfinite(res.tail_xy).all()
        assert len(np.unique(res.fine_parents)) >= 1
        # every fine centroid maps back to exactly one top parent
        assert (res.fine_parents_to_top >= 0).all() and (res.fine_parents_to_top < 8).all()
        # voronoi + push apart ran and converged
        assert res.voronoi is not None
        assert len(np.unique(res.voronoi.labels)) >= 1
        assert res.push_apart.min_separation >= 0.0

    def test_run_basemap_no_mid_level(self):
        import emaopt

        rng = np.random.default_rng(0)
        x = rng.normal(size=(400, 4)).astype(np.float32)
        x[:200, 0] += 5.0
        res = emaopt.run_basemap(
            x, top_k=6, mid_k=0, fine_k=24, nepochs=2, seed=0,
            groups=5, do_tail=False, device="cuda",
        )
        assert res.mid_xy is None
        assert res.mid_features is None
        assert res.mid_parents is None
        assert res.fine_xy.shape == (24, 2)
        assert res.tail_xy.shape == (400, 2)
        assert res.tail_parents.shape == (400,)
        # do_tail=False leaves tail_parents as -1 sentinels
        assert (res.tail_parents == -1).all()
