"""Device-resident e-means: the streaming e-means loop with the device as the
destination of the work.

The host :func:`emaopt.algorithms.emeans.emeans` is the reference: it folds
EMA sufficient statistics (sums/counts) on every batch and decodes centroids
per step.  This module keeps the same *math* on the GPU -- the batch is
uploaded once per step, the E-step (distance + fold) runs on the device, and
only the final centroids, labels and objective cross PCIe -- so the device
arm stays fast at the LAION / MONET shapes.

Design contract:

* Init is a device k-means++ greedy pass over a subsample (``init_sample``).
  For very large ``k`` this is capped and the fit runs with a random-subsample
  seed instead (maternal-style): the fold still refines it, and the alternative
  -- an O(n k) greedy pass at k = 1e6 -- never pays.
* The fold is the host's EMA law: ``S <- (1-a) S + a s``, ``N <- (1-a) N + a n``
  with ``a`` from the same cosine schedule and fitted laws as the host
  (:func:`batch_size_from_law`, :func:`alpha0_from_law`).  ``revival`` is not
  ported (it is a host RNG walk); pass ``revival=False``.
* Distances are computed in row-chunks x centroid-chunks so an (n, k) distance
  matrix is never materialised.  ``dtype`` is the precision of the E-step;
  accumulators follow the host convention (float64 in the device path, float32
  optional).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .algorithms.kmeans import batch_size_from_law, alpha0_from_law
from .core import squared_distances

__all__ = ["GpuEmeansConfig", "GpuEmeansResult", "emeans_gpu_fit"]

_FLOOR = 1e-12


@dataclass(frozen=True)
class GpuEmeansConfig:
    k: int
    nepochs: int = 5
    batch_size: int | None = None     # None -> the fitted law
    alpha0: float | None = None       # None -> the fitted law
    alpha_end_ratio: float = 0.1
    batch_law: str = "auto"
    init_sample: int = 100_000
    max_init_k: int = 25_000          # greedy k-means++ cap for large k
    seed: int = 0
    dtype: str = "float32"            # E-step precision
    stat_dtype: str = "float64"       # fold precision
    chunk_rows: int = 4096            # device row chunks
    cluster_chunk: int = 65536        # device centroid chunks
    revival: bool = False
    device: str | None = None

    def __post_init__(self) -> None:
        if int(self.k) < 1:
            raise ValueError("k must be >= 1")
        if int(self.nepochs) < 1:
            raise ValueError("nepochs must be >= 1")
        if self.revival:
            raise NotImplementedError(
                "revival is not ported to the device loop (host RNG walk); use the host emeans()"
            )


@dataclass
class GpuEmeansResult:
    centroids: np.ndarray       # (k, d) float64
    labels: np.ndarray          # (n,) int64
    objective: float            # mean min squared distance
    updates: int
    wall_time: float
    seed: int
    nepochs: int
    batch_size: int
    alpha0: float
    init: str
    method: str
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": "emeans_gpu",
            "k": int(self.centroids.shape[0]),
            "d": int(self.centroids.shape[1]),
            "n": int(self.labels.shape[0]),
            "objective": float(self.objective),
            "batch_size": int(self.batch_size),
            "alpha0": float(self.alpha0),
            "nepochs": int(self.nepochs),
            "updates": int(self.updates),
            "wall_time": float(self.wall_time),
            "seed": int(self.seed),
            "init": self.init,
            "method": self.method,
        }


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError("emeans_gpu_fit needs torch") from e
    return torch


def _device_of(device: str | None):
    torch = _torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _alpha_grid(alpha0: float, end_ratio: float, total_steps: int):
    """Cosine schedule matching :class:`CosineSchedule` exactly."""
    total = max(int(total_steps), 1)
    a0 = float(alpha0)
    a1 = a0 * float(end_ratio)
    steps = np.arange(total, dtype=np.float64)
    p = np.minimum(1.0, np.maximum(0.0, (steps + 1.0) / total))
    return a1 + 0.5 * (a0 - a1) * (1.0 + np.cos(np.pi * p))


def _kmeans_pp_device(x_t, k: int, rng, torch, chunk_rows: int = 8192,
                      max_centroids: int = 25_000):
    """Greedy k-means++ on a device subsample, chunked over rows."""
    n, d = x_t.shape
    k = min(int(k), n)
    centroids = torch.empty((k, d), device=x_t.device, dtype=x_t.dtype)
    first = int(torch.randint(0, n, (1,), generator=rng, device=x_t.device).item())
    centroids[0] = x_t[first]
    closest = torch.full((n,), float("inf"), device=x_t.device, dtype=torch.float32)
    closest.scatter_(0, torch.arange(n, device=x_t.device),
                     torch.sum((x_t - centroids[0:1]) ** 2, dim=1))
    for j in range(1, k):
        total = closest.sum()
        if not torch.isfinite(total) or total <= 0.0:
            centroids[j] = x_t[torch.randint(0, n, (1,), generator=rng, device=x_t.device).item()]
            continue
        probs = (closest / total).clamp(min=0.0)
        # sample the next point
        cum = torch.cumsum(probs, dim=0)
        draw = torch.rand(1, generator=rng, device=x_t.device).item()
        new_idx = int(torch.searchsorted(cum, draw).item())
        new_idx = min(new_idx, n - 1)
        centroids[j] = x_t[new_idx]
        # update running closest distance
        d2 = torch.sum((x_t - centroids[j:j + 1]) ** 2, dim=1)
        closest = torch.minimum(closest, d2)
    return centroids


def _assign_labels(x_block, centroids, cluster_chunk: int, torch):
    """Chunked top-1 assignment of ``x_block`` to ``centroids``."""
    n, d = x_block.shape
    k = centroids.shape[0]
    best_d2 = torch.full((n,), float("inf"), device=x_block.device, dtype=torch.float32)
    best_label = torch.zeros((n,), device=x_block.device, dtype=torch.int64)
    x_norm = torch.sum(x_block * x_block, dim=1)
    for c_start in range(0, k, int(cluster_chunk)):
        c_end = min(c_start + int(cluster_chunk), k)
        c_block = centroids[c_start:c_end]
        c_norm = torch.sum(c_block * c_block, dim=1)
        d2 = x_norm[:, None] + c_norm[None, :] - 2.0 * (x_block @ c_block.T)
        d2 = torch.clamp(d2, min=0.0)
        local_min, local_arg = torch.min(d2, dim=1)
        improved = local_min < best_d2
        best_label = torch.where(improved, local_arg + c_start, best_label)
        best_d2 = torch.where(improved, local_min, best_d2)
    return best_label, best_d2


def emeans_gpu_fit(
    x: np.ndarray,
    k: int,
    *,
    nepochs: int = 5,
    batch_size: int | None = None,
    alpha0: float | None = None,
    alpha_end_ratio: float = 0.1,
    batch_law: str = "auto",
    init_sample: int = 100_000,
    init_centroids: np.ndarray | None = None,
    max_init_k: int = 25_000,
    seed: int = 0,
    dtype: str = "float32",
    stat_dtype: str = "float64",
    chunk_rows: int = 4096,
    cluster_chunk: int = 65536,
    device: str | None = None,
    status: Callable[[int, int, float], None] | None = None,
) -> GpuEmeansResult:
    """Fit e-means entirely on the device.

    ``x`` is the full array; it is *not* uploaded a full copy of to the
    device -- each batch is uploaded once.  ``init_centroids`` (k, d) starts
    from given centroids (maternal).  The device keeps the EMA sufficient
    statistics in ``stat_dtype`` and folds them per step.
    """
    t0 = time.perf_counter()
    torch = _torch()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x must be (n, d)")
    n, d = x.shape
    if n < 1:
        raise ValueError("x is empty")
    k = int(k)
    if k < 1:
        raise ValueError("k must be >= 1")
    if n < k and init_centroids is None:
        k = n  # degenerate; centroids == rows
    dev = _device_of(device)
    dtype_t = torch.float64 if dtype == "float64" else torch.float32
    stat_t = torch.float64 if stat_dtype == "float64" else torch.float32
    stat_dtype_name = "float64" if stat_t is torch.float64 else "float32"

    rng = np.random.default_rng(int(seed))
    bs = int(batch_size) if batch_size is not None else batch_size_from_law(
        n, k, nepochs, batch_law=batch_law, d=d
    )
    bs = int(np.clip(bs, 1, n))
    total_steps = int(math.ceil(n / bs)) * int(nepochs)
    a0 = float(alpha0) if alpha0 is not None else alpha0_from_law(n, k, bs, nepochs)
    a0 = float(np.clip(a0, 1e-4, 1.0))
    alphas = _alpha_grid(a0, alpha_end_ratio, total_steps)

    # ---- init ----------------------------------------------------------- #
    init_method = "maternal" if init_centroids is not None else "kmeans++"
    sub_n = min(n, int(init_sample))
    sub = rng.choice(n, size=sub_n, replace=False)
    x_sub = np.ascontiguousarray(x[sub])
    if init_centroids is not None:
        given = np.asarray(init_centroids, dtype=np.float64)
        if given.shape != (k, d):
            raise ValueError(f"init_centroids shape {given.shape} != {(k, d)}")
        centroids = torch.as_tensor(given, device=dev, dtype=stat_t).clone()
        init_method = "maternal"
    else:
        x_sub_t = torch.as_tensor(x_sub, device=dev, dtype=dtype_t)
        if k <= int(max_init_k):
            rng_t = torch.Generator(device=dev)
            rng_t.manual_seed(int(seed))
            centroids = _kmeans_pp_device(x_sub_t, k, rng_t, torch, chunk_rows, int(max_init_k))
            centroids = centroids.to(stat_t)
        else:
            # large k: maternal-style random seed from the subsample
            init_method = "random"
            idx = torch.as_tensor(rng.choice(sub_n, size=k, replace=False), device=dev)
            centroids = x_sub_t[idx].to(stat_t).clone()
    # Warm start: give each centroid ``1e-6`` units of mass so the first
    # decode is finite and the maternal seed survives the first fold.
    sums = centroids * 1e-6
    counts = torch.full((k,), 1e-6, device=dev, dtype=stat_t)

    # ---- streaming fit ---------------------------------------------------- #
    per_epoch = int(math.ceil(n / bs))
    order = np.arange(n)
    updates = 0
    step = 0
    status_interval = max(1, total_steps // 20)
    for epoch in range(int(nepochs)):
        perm = rng.permutation(n) if epoch > 0 else order
        for start in range(0, n, bs):
            end = min(start + bs, n)
            idx = perm[start:end]
            rows = torch.as_tensor(x[idx], device=dev, dtype=dtype_t)
            labels, d2min = _assign_labels(
                rows, centroids.to(dtype_t), int(cluster_chunk), torch
            )
            block_sums = torch.zeros((k, d), device=dev, dtype=dtype_t)
            block_sums.index_add_(0, labels, rows)
            block_counts = torch.zeros((k,), device=dev, dtype=dtype_t)
            block_counts.index_add_(0, labels, torch.ones_like(labels, dtype=dtype_t))
            alpha = float(alphas[step]) if step < len(alphas) else float(alphas[-1])
            sums = (1.0 - alpha) * sums + alpha * block_sums.to(stat_t)
            counts = (1.0 - alpha) * counts + alpha * block_counts.to(stat_t)
            updates += 1
            step += 1
            if status is not None and (updates % status_interval == 0):
                status(updates, total_steps, float(d2min.mean().detach()))
            # decode once per batch (the assignment above was against the
            # previous decode); cheap O(k d) op, no sync.
            centroids = sums / torch.clamp(counts, min=_FLOOR)[:, None]
        # end of epoch

    # ---- final state ------------------------------------------------------- #
    centroids = sums / torch.clamp(counts, min=_FLOOR)[:, None]
    final_counts = torch.clamp(counts, min=_FLOOR)
    dead = counts <= 1e-9

    # ---- labels + objective ----------------------------------------------- #
    labels_all = np.empty((n,), dtype=np.int64)
    obj_sum = 0.0
    for start in range(0, n, int(chunk_rows)):
        end = min(start + int(chunk_rows), n)
        rows = torch.as_tensor(x[start:end], device=dev, dtype=dtype_t)
        lab, d2min = _assign_labels(rows, centroids.to(dtype_t), int(cluster_chunk), torch)
        labels_all[start:end] = lab.to("cpu", dtype=torch.int64).numpy()
        obj_sum += float(d2min.sum().detach())
    objective = float(obj_sum / max(1, n))
    total = time.perf_counter() - t0
    return GpuEmeansResult(
        centroids=centroids.to("cpu", dtype=torch.float64).numpy(),
        labels=labels_all,
        objective=objective,
        updates=int(updates),
        wall_time=float(total),
        seed=int(seed),
        nepochs=int(nepochs),
        batch_size=int(bs),
        alpha0=float(a0),
        init=init_method,
        method="device",
        extras={
            "dtype": dtype,
            "stat_dtype": stat_dtype_name,
            "init_sample": int(sub_n),
            "per_epoch": int(per_epoch),
            "dead_clusters": int(dead.sum().item()),
            "n": int(n),
            "d": int(d),
            "k": int(k),
            "device": str(dev),
        },
    )
