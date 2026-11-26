"""Tests for the latent-scope integration bridge (guarded, no hard dep)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
emaopt_bridge = pytest.importorskip("emaopt.integrations")


class TestAvailability:
    def test_is_available_never_raises(self):
        assert isinstance(emaopt_bridge.is_available(), bool)

    def test_probe_emaopt_is_bool(self):
        assert isinstance(emaopt_bridge.probe_emaopt(), bool)

    def test_probe_false_without_cuda(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert emaopt_bridge.probe_emaopt() is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
class TestBackends:
    def _blobs(self, n=270, k=3, d=8, seed=0):
        rng = np.random.default_rng(seed)
        parts = []
        for i in range(k):
            loc = np.zeros(d)
            loc[0] = 10.0 * i
            parts.append(rng.normal(loc=loc, scale=0.5, size=(n, d)))
        return np.vstack(parts).astype(np.float32)

    def test_run_umap_shape_and_finite(self):
        x = self._blobs(n=100)
        emb, backend = emaopt_bridge.run_umap(x, 10, 0.075, 0)
        assert emb.shape == (x.shape[0], 2)
        assert backend == "emaopt"
        assert np.isfinite(emb).all()

    def test_run_umap_3d_rejected(self):
        x = self._blobs(n=100)
        with pytest.raises(ValueError):
            emaopt_bridge.run_umap(x, 10, 0.075, 0, dimensions=3)

    def test_run_emeans_recovers_blobs(self):
        x = self._blobs(n=120)
        labels, centroids = emaopt_bridge.run_emeans(x, 3, nepochs=5, seed=0)
        assert labels.shape == (x.shape[0],)
        assert centroids.shape == (3, x.shape[1])
        from sklearn.metrics import adjusted_rand_score

        true = np.repeat([0, 1, 2], x.shape[0] // 3)
        assert adjusted_rand_score(true, labels) > 0.9

    def test_run_emeans_shapes(self):
        x = self._blobs(n=90)
        labels, centroids = emaopt_bridge.run_emeans(x, 2, nepochs=3, seed=1)
        assert labels.shape == (x.shape[0],)
        assert centroids.shape == (2, x.shape[1])
        assert np.isfinite(centroids).all()

    def test_install_latentscope_is_idempotent(self):
        # without latent-scope installed this returns False, never raises
        res = emaopt_bridge.install_latentscope()
        assert isinstance(res, bool)
        if res:
            assert emaopt_bridge.install_latentscope() is True
