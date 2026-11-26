"""Phase-5 composed pipeline: four stages, bounded memory, finite results."""

from __future__ import annotations

import numpy as np

from emaopt.pipeline import PipelineConfig, StreamingPipeline


def _write_shard(tmp_path, n=6000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(n, 3))
    mixing = rng.normal(size=(3, d))
    x = sources @ mixing + 0.1 * rng.normal(size=(n, d))
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "emb.npy"
    np.save(path, x.astype(np.float16))
    return str(path), n, d


def test_pipeline_runs_four_stages_on_an_f16_memmap(tmp_path):
    path, n, d = _write_shard(tmp_path)
    config = PipelineConfig(n_components=6, n_clusters=5, gmm_components=3,
                            batch_size=512, reservoir=2000, seed=0)
    result = StreamingPipeline(config).fit(
        StreamingPipeline.memmap_factory(path, 512)
    )

    assert result.whitening["components"].shape == (d, d)
    assert result.ica["unmixing"].shape == (6, d)
    assert result.clusters["centroids"].shape == (5, 6)
    assert result.mixture["means"].shape == (3, 6)
    assert all(np.isfinite(v) for v in result.objective.values())
    assert result.objective["gmm_nll"] < 50.0
    assert result.objective["kmeans_inertia"] > 0.0
    assert result.passes == 4
    assert result.examples_processed == 4 * n
    assert len(result.stages) == 4
    assert all("config" in stage for stage in result.stages.values())


def test_pipeline_state_is_independent_of_the_shard_length(tmp_path):
    short, _, _ = _write_shard(tmp_path / "a", n=2000, seed=1)
    long, _, _ = _write_shard(tmp_path / "b", n=20_000, seed=2)
    config = PipelineConfig(n_components=5, n_clusters=4, gmm_components=2,
                            batch_size=512, reservoir=2000, seed=0)
    a = StreamingPipeline(config).fit(StreamingPipeline.memmap_factory(short, 512))
    b = StreamingPipeline(config).fit(StreamingPipeline.memmap_factory(long, 512))
    assert a.memory["state_bytes"] == b.memory["state_bytes"]
    # only the reservoir (bounded) may differ modestly, not the live state
    assert b.memory["state_bytes"] < 1_000_000


def test_pipeline_fits_the_phase5_one_gigabyte_budget(tmp_path):
    path, _, _ = _write_shard(tmp_path, n=8000, d=64, seed=3)
    config = PipelineConfig(n_components=16, n_clusters=8, gmm_components=4,
                            batch_size=1024, reservoir=4000, seed=0)
    result = StreamingPipeline(config).fit(
        StreamingPipeline.memmap_factory(path, 1024)
    )
    assert result.within_budget(1024 ** 3)
    assert result.memory["peak_bytes"] < 100 * 1024 ** 2
    summary = result.summary()
    assert "memory" in summary and "objective" in summary
