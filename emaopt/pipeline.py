"""Composed streaming pipeline: whiten -> ICA -> e-means -> mixture.

ONLINE_PLAN phase 5.  The LAION chain, expressed entirely in sufficient
statistics:

1. **whiten** -- a centred :class:`~emaopt.state.sufficient.GramianStat` over
   the shard yields the mean and covariance; the retained ``k`` directions give
   a ``(k, d)`` whitening matrix.
2. **ICA** -- :class:`~emaopt.algorithms.ica.FastICAStat` with the whitening
   fixed by stage 1, one TD pass over the blocks.
3. **e-means** -- :class:`~emaopt.state.sufficient.CentroidStat` with the e-means
   EMA law, initialised k-means++ on a uniform reservoir of ICA features.
4. **mixture** -- :class:`~emaopt.core.StatisticalOptimizer` (diagonal mixture)
   wrapped as an :class:`~emaopt.state.sufficient.OptimizerStat` and streamed
   over the same features.

One pass per stage, state is ``O(d^2 + k d)`` plus a bounded reservoir, and no
stage materialises a float64 copy of the shard.  ``fit`` takes a *block
factory* (a callable returning a fresh iterable of blocks) so each stage can
re-read a memory-mapped file without the pipeline owning the data.

The memory report is part of the result, because the point of the phase is the
budget, not only the quality: ``result.memory["peak_bytes"]`` is the largest
working set the chain actually needs (blocks + state + reservoir).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from .algorithms.gmm import GaussianMixture
from .algorithms.ica import FastICAStat, whitening_matrix
from .core import squared_distances
from .state.sufficient import CentroidStat, GramianStat, OptimizerStat

_EPS = 1e-12
GB = 1024 ** 3


@dataclass(frozen=True)
class PipelineConfig:
    """Shape and budget of the chain (all defaults are deliberately small)."""

    n_components: int = 64
    n_clusters: int = 64
    gmm_components: int = 16
    covariance: str = "diag"
    batch_size: int = 4096
    ica_decay: float = 0.02
    ica_act_every: int = 1
    ema_alpha: float = 0.1
    gmm_alpha: float = 0.05
    nepochs: int = 1
    reservoir: int = 20_000
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_components": int(self.n_components),
            "n_clusters": int(self.n_clusters),
            "gmm_components": int(self.gmm_components),
            "covariance": self.covariance,
            "batch_size": int(self.batch_size),
            "ica_decay": float(self.ica_decay),
            "ica_act_every": int(self.ica_act_every),
            "ema_alpha": float(self.ema_alpha),
            "gmm_alpha": float(self.gmm_alpha),
            "nepochs": int(self.nepochs),
            "reservoir": int(self.reservoir),
            "seed": int(self.seed),
        }


@dataclass
class PipelineResult:
    whitening: dict[str, Any]
    ica: dict[str, Any]
    clusters: dict[str, Any]
    mixture: dict[str, Any]
    stages: dict[str, dict[str, Any]]
    objective: dict[str, float]
    memory: dict[str, Any]
    wall_time: float
    examples_processed: int
    passes: int
    config: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        out = {
            "objective": dict(self.objective),
            "memory": dict(self.memory),
            "wall_time": float(self.wall_time),
            "passes": int(self.passes),
            "examples_processed": int(self.examples_processed),
            "stages": {k: v.get("config", {}) for k, v in self.stages.items()},
        }
        out.update(self.config)
        return out

    def within_budget(self, limit_bytes: int) -> bool:
        return int(self.memory["peak_bytes"]) <= int(limit_bytes)


def _kmpp(block: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding on a reservoir (the same greedy variant as KMeans)."""
    n, d = block.shape
    k = min(int(k), n)
    centroids = np.empty((k, d), dtype=np.float64)
    centroids[0] = block[rng.integers(n)]
    closest = squared_distances(block, centroids[:1]).ravel()
    trials = min(n, 2 + int(np.log(max(k, 2))))
    for j in range(1, k):
        total = float(closest.sum())
        if total <= 0.0 or not np.isfinite(total):
            centroids[j] = block[rng.integers(n)]
            continue
        candidates = np.unique(rng.choice(n, size=trials, p=closest / total))
        d2 = squared_distances(block, block[candidates])
        potentials = np.minimum(closest[:, None], d2).sum(axis=0)
        best = int(np.argmin(potentials))
        centroids[j] = block[candidates[best]]
        closest = np.minimum(closest, d2[:, best])
    return centroids


class StreamingPipeline:
    """The four-stage chain over an f16 resident shard, O(kd) state per stage."""

    def __init__(self, config: PipelineConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = PipelineConfig(**kwargs)
        self.config = config

    # -- helpers ------------------------------------------------------------ #
    @staticmethod
    def memmap_factory(path: str, block_size: int, *, nepochs: int = 1, seed: int = 0,
                       shuffle: bool = False) -> Callable[[], Iterable[np.ndarray]]:
        """A block factory re-reading ``path`` for every stage."""
        from .stream import iter_blocks

        def factory() -> Iterable[np.ndarray]:
            return iter_blocks(path, block_size, nepochs=nepochs, shuffle=shuffle, seed=seed)

        return factory

    # -- the chain ---------------------------------------------------------- #
    def fit(
        self,
        blocks_factory: Callable[[], Iterable[np.ndarray]],
        *,
        eval_blocks_factory: Callable[[], Iterable[np.ndarray]] | None = None,
    ) -> PipelineResult:
        cfg = self.config
        wall0 = time.perf_counter()
        d = None
        n = 0
        # ---- stage 1: whiten -------------------------------------------- #
        gram = GramianStat(center=True)
        for block in blocks_factory():
            block = np.asarray(block, dtype=np.float64)
            d = int(block.shape[1])
            n += int(block.shape[0])
            gram.update(block)
        if d is None:
            raise ValueError("pipeline received no blocks")
        whiten_params = gram.decode()
        cov = whiten_params["covariance"]
        mean = whiten_params["mean"]
        k = int(min(cfg.n_components, d))
        wz = whitening_matrix(cov, k)
        # ---- stage 2: ICA ------------------------------------------------ #
        ica = FastICAStat(
            k, mode="ema", alpha=cfg.ica_decay, act_every=cfg.ica_act_every,
            seed=cfg.seed, whitening="fixed",
        )
        ica.prepare(d=d, covariance=cov)
        stride = max(1, n // max(1, cfg.reservoir))
        seen = 0
        pool: list[np.ndarray] = []
        kept = 0
        for block in blocks_factory():
            block = np.asarray(block, dtype=np.float64)
            xw = (block - mean) @ wz.T
            ica.update_whitened(xw)
            start = (-seen) % stride
            take = xw[start::stride]
            if take.size and kept < cfg.reservoir:
                take = take[: cfg.reservoir - kept]
                pool.append(take)
                kept += take.shape[0]
            seen += block.shape[0]
        if not pool:  # pragma: no cover - defensive
            raise ValueError("pipeline reservoir is empty")
        whitened_sample = np.concatenate(pool, axis=0)
        ica_params = ica.decode()
        unmixing = ica_params["unmixing"]
        features = whitened_sample @ ica_params["components"].T
        # ---- stage 3: e-means -------------------------------------------- #
        rng = np.random.default_rng(cfg.seed)
        centroids0 = _kmpp(features, cfg.n_clusters, rng)
        clusters = CentroidStat(
            min(cfg.n_clusters, features.shape[0]), d=k, centroids=centroids0,
            mode="ema", alpha=cfg.ema_alpha, act_every=1, seed=cfg.seed,
        )
        for block in blocks_factory():
            block = np.asarray(block, dtype=np.float64)
            clusters.update(((block - mean) @ wz.T) @ ica_params["components"].T)
        cluster_params = clusters.decode()
        # ---- stage 4: mixture -------------------------------------------- #
        gmm = GaussianMixture(cfg.gmm_components, covariance=cfg.covariance, seed=cfg.seed)
        gmm_stat = OptimizerStat(gmm, mode="ema", alpha=cfg.gmm_alpha)
        gmm_stat.prepare(features)
        for block in blocks_factory():
            block = np.asarray(block, dtype=np.float64)
            gmm_stat.update(((block - mean) @ wz.T) @ ica_params["components"].T)
        mixture_params = gmm_stat.decode()
        # ---- objectives -------------------------------------------------- #
        ica_unmixed = whitened_sample @ np.asarray(ica.params["W"]).T
        objective = {
            "ica_negentropy": float(-np.mean(np.log(np.cosh(ica_unmixed)))),
            "kmeans_inertia": float(np.mean(
                squared_distances(features, cluster_params["centroids"]).min(axis=1)
            )),
            "gmm_nll": float(gmm_stat.optimizer.objective(features, mixture_params)),
        }
        n_blocks = int(np.ceil(n / max(1, cfg.batch_size)))
        state_bytes = int(gram.nbytes + ica.nbytes + clusters.nbytes + gmm_stat.nbytes)
        reservoir_bytes = int(features.nbytes * 2 + whitened_sample.nbytes)
        block_bytes = int(cfg.batch_size * d * 8)
        memory = {
            "state_bytes": state_bytes,
            "reservoir_bytes": reservoir_bytes,
            "block_bytes": block_bytes,
            "peak_bytes": int(state_bytes + reservoir_bytes + block_bytes),
            "source_rows": int(n),
            "source_d": int(d),
            "blocks_per_stage": n_blocks,
            "stages": {
                "whiten": gram.memory_report(),
                "ica": ica.memory_report(),
                "clusters": clusters.memory_report(),
                "mixture": gmm_stat.memory_report(),
            },
        }
        passes = 4
        return PipelineResult(
            whitening=whiten_params,
            ica=ica_params,
            clusters=cluster_params,
            mixture=mixture_params,
            stages={
                "whiten": {"config": _jsonable(gram.config()), "nbytes": gram.nbytes},
                "ica": {"config": _jsonable(ica.config()), "nbytes": ica.nbytes},
                "clusters": {"config": _jsonable(clusters.config()), "nbytes": clusters.nbytes},
                "mixture": {"config": _jsonable(gmm_stat.config()), "nbytes": gmm_stat.nbytes},
            },
            objective=objective,
            memory=memory,
            wall_time=float(time.perf_counter() - wall0),
            examples_processed=int(n * passes),
            passes=passes,
            config={"pipeline": cfg.to_dict(), "stride": int(stride)},
        )


def _jsonable(value: Any) -> Any:
    """Best-effort JSON conversion for stage configs (numpy scalars, arrays)."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


__all__ = ["PipelineConfig", "PipelineResult", "StreamingPipeline"]
