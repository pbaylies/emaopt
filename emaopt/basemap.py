"""Hierarchical 2-D placement on top of UMAP-style embeddings.

The basemap pipeline (toy-scale version of the MONET-style scheme):

1. embed a sample with a UMAP-style embedding (:mod:`emaopt.algorithms.umap`)
2. train e-means centroids on real data (:mod:`emaopt.gpu_emeans`)
3. embed the top level (e.g. 32k centroids -> 2D)
4. place 260k/1m mid-level points around their parent centroids
5. place all rows around the 1m centroids
6. cut the 1m centroids into ~2k clusters (Voronoi) and push them apart

This module is steps 4-6:

* :func:`nearest_parent` -- chunked top-1 assignment of points to parents
  (feature space), the same math as :func:`emaopt.gpu_emeans._assign_labels`.
* :func:`place_around_parents` -- put a child at its parent's 2-D position plus
  an offset scaled to the local 2-D neighbour spacing (``spread`` of it, in
  class space).  The fill is deterministic (seed).
* :func:`voronoi_cut` -- cut the 1m centroids into ``groups`` Voronoi cells
  with 2-D Lloyd k-means.
* :func:`push_apart` -- uniform repulsion between the ``groups`` cluster
  centres until the minimum separation is reached (a bounded number of steps).

All functions are device-first (torch on the GPU when available) and keep the
(n, k) distance matrices chunked.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = [
    "place_around_parents",
    "nearest_parent",
    "voronoi_cut",
    "push_apart",
    "run_basemap",
    "HierarchicalPlacement",
    "BasemapResult",
    "VoronoiResult",
    "PushApartResult",
]


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError("basemap placement needs torch") from e
    return torch


def _device_of(device: str | None):
    torch = _torch()
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_float32(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32)


def nearest_parent(
    points: np.ndarray,
    parents: np.ndarray,
    *,
    device: str | None = None,
    chunk_rows: int = 4096,
    cluster_chunk: int = 65536,
) -> np.ndarray:
    """Nearest parent index for each point (row-chunked, never n x k)."""
    torch = _torch()
    dev = _device_of(device)
    pts = torch.as_tensor(_as_float32(points), device=dev)
    par = torch.as_tensor(_as_float32(parents), device=dev)
    n, d = pts.shape
    k = par.shape[0]
    out = np.empty((n,), dtype=np.int64)
    par_norm = torch.sum(par * par, dim=1)
    for start in range(0, n, int(chunk_rows)):
        end = min(start + int(chunk_rows), n)
        block = pts[start:end]
        best = torch.full((end - start,), float("inf"), device=dev)
        best_idx = torch.zeros((end - start,), device=dev, dtype=torch.int64)
        block_norm = torch.sum(block * block, dim=1)
        for c_start in range(0, k, int(cluster_chunk)):
            c_end = min(c_start + int(cluster_chunk), k)
            c_block = par[c_start:c_end]
            c_norm = par_norm[c_start:c_end]
            d2 = block_norm[:, None] + c_norm[None, :] - 2.0 * (block @ c_block.T)
            d2 = torch.clamp(d2, min=0.0)
            local = torch.min(d2, dim=1)
            improved = local.values < best
            best = torch.where(improved, local.values, best)
            best_idx = torch.where(improved, local.indices + c_start, best_idx)
        out[start:end] = best_idx.to("cpu", dtype=torch.int64).numpy()
    return out


def place_around_parents(
    child_parents: np.ndarray,
    parent_xy: np.ndarray,
    *,
    parents_2d_neighbour_spacing: float | None = None,
    parent_spacing: np.ndarray | None = None,
    spread: float = 0.3,
    seed: int = 0,
    local_spacing: bool = True,
) -> np.ndarray:
    """Place each child at its parent's 2-D position plus an offset.

    ``child_parents`` (n,) parent index per child; ``parent_xy`` (P, 2) 2-D
    positions of the parents (the UMAP'd top level).  The offset direction is a
    deterministic pseudo-random unit vector, scaled to ``spread`` x the parent's
    2-D neighbour spacing.  With ``local_spacing=True`` (default, and what the
    real MONET scheme needs) each parent is scaled by its own local 2-D
    nearest-neighbour distance, so dense parents get tight offsets and sparse
    parents get wide ones; with ``local_spacing=False`` the offsets use the
    global median spacing of the parent cloud.

    ``parents_2d_neighbour_spacing`` overrides the scale entirely (pass the
    global median or a fixed absolute value).  ``spread`` is a fraction:
    ``spread * spacing`` is the mean offset scale.  Empty parents get zero
    offset.
    """
    parents = np.asarray(child_parents, dtype=np.int64)
    parent_xy = np.asarray(parent_xy, dtype=np.float32)
    if parent_xy.ndim != 2 or parent_xy.shape[1] != 2:
        raise ValueError("parent_xy must be (P, 2)")
    if parents.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if parents.min() < 0 or parents.max() >= parent_xy.shape[0]:
        raise ValueError("child_parents out of range")

    # local 2-D neighbour spacing (per-parent nearest 2-D neighbour)
    if parent_spacing is not None:
        sp_scale = np.asarray(parent_spacing, dtype=np.float64)
        if sp_scale.shape != (parent_xy.shape[0],):
            raise ValueError("parent_spacing must be (P,)")
    elif parents_2d_neighbour_spacing is not None:
        sp_scale = np.full(parent_xy.shape[0], float(parents_2d_neighbour_spacing), dtype=np.float64)
    elif local_spacing:
        sp_scale = _local_spacing(parent_xy)
    else:
        spacing = _neighbour_spacing(parent_xy)
        sp_scale = np.full(parent_xy.shape[0], spacing, dtype=np.float64)
    sp_scale = np.maximum(sp_scale, 1e-12)
    if not np.isfinite(sp_scale).all():
        sp_scale = np.where(np.isfinite(sp_scale), sp_scale, 1.0)

    rng = np.random.default_rng(int(seed))
    n = int(parents.shape[0])
    offsets = rng.normal(size=(n, 2)).astype(np.float64)
    norms = np.linalg.norm(offsets, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    offsets = offsets / norms
    offsets *= spread * sp_scale[parents][:, None] * rng.uniform(0.3, 1.0, size=(n, 1))
    return parent_xy[parents].astype(np.float64) + offsets


def _local_spacing(parent_xy: np.ndarray, chunk: int = 128) -> np.ndarray:
    """Per-parent 2-D nearest-neighbour distance (row-chunked).

    Returns an array of shape (P,).  Groups with a missing neighbour (single
    parent, or all-duplicated positions) fall back to the global median.
    """
    p = np.asarray(parent_xy, dtype=np.float64)
    P = p.shape[0]
    if P <= 1:
        return np.full(P, 1.0 if P == 1 else 0.0, dtype=np.float64)
    sp = np.full(P, np.inf, dtype=np.float64)
    for start in range(0, P, int(chunk)):
        end = min(start + int(chunk), P)
        block = p[start:end]
        d = np.sqrt(np.sum((block[:, None, :] - p[None, :, :]) ** 2, axis=2))
        self_rows = np.arange(start, end)[:, None]
        d = np.where(np.arange(P)[None, :] == self_rows, np.inf, d)
        sp[start:end] = d.min(axis=1)
    fallback = float(np.median(sp[np.isfinite(sp)])) if np.isfinite(sp).any() else 1.0
    fallback = fallback if fallback > 0.0 else 1.0
    return np.where(np.isfinite(sp) & (sp > 0), sp, fallback)


def _neighbour_spacing(parent_xy: np.ndarray, chunk: int = 128) -> float:
    """Median distance to the nearest other parent, computed row-chunked."""
    p = np.asarray(parent_xy, dtype=np.float64)
    P = p.shape[0]
    if P <= 1:
        return 1.0
    sp = np.full(P, np.inf)
    for start in range(0, P, int(chunk)):
        end = min(start + int(chunk), P)
        block = p[start:end]
        # (chunk, P) nearest-neighbour distance within the full set
        d = np.sqrt(np.sum((block[:, None, :] - p[None, :, :]) ** 2, axis=2))
        # mask the row's own column (self)
        self_rows = np.arange(start, end)[:, None]
        d = np.where(np.arange(P)[None, :] == self_rows, np.inf, d)
        sp[start:end] = d.min(axis=1)
    valid = np.isfinite(sp)
    if not valid.any():
        return 1.0
    spacing = float(np.median(sp[valid]))
    return spacing if spacing > 0.0 else 1.0


@dataclass
class VoronoiResult:
    labels: np.ndarray            # (m,) group index
    centers: np.ndarray           # (G, 2) Lloyd centres
    converged: bool
    iterations: int
    wall_time: float
    objective: float              # mean squared distance to assigned centre
    extras: dict[str, Any] = field(default_factory=dict)


def _kmeanspp_init(pos, G, device, torch, rng_gen):
    """Deterministic greedy k-means++ init over a subsample (mean-k++ )."""
    m, d = pos.shape
    sub_n = min(m, 5000)
    sub_idx = torch.randperm(m, device=device, generator=rng_gen)[:sub_n]
    sub = pos[sub_idx]
    s = sub.shape[0]
    centres = torch.empty((G, 2), device=device, dtype=torch.float32)
    centres[0] = sub[torch.randint(0, s, (1,), device=device, generator=rng_gen).item()]
    closest = torch.sum((sub - centres[0]) ** 2, dim=1)
    for j in range(1, G):
        total = closest.sum().clamp(min=1e-12)
        # deterministic: argmax distance to nearest chosen centre
        pick = int(torch.argmax(closest).item())
        centres[j] = sub[pick]
        d2 = torch.sum((sub - centres[j]) ** 2, dim=1)
        closest = torch.minimum(closest, d2)
    return centres.to(torch.float64)


def voronoi_cut(
    positions: np.ndarray,
    groups: int,
    *,
    iterations: int = 50,
    seed: int = 0,
    device: str | None = None,
    tolerance: float = 1e-4,
    init: str = "kmeans++",
    status: Callable[[int, int, float], None] | None = None,
) -> VoronoiResult:
    """Cut ``positions`` (m, 2) into ``groups`` Voronoi cells (Lloyd k-means).

    ``init="kmeans++"`` (default) seeds the centres with a deterministic greedy
    k-means++ pass over a subsample (cheap, helps Lloyd actually converge);
    ``init="bbox"`` keeps the uniform bounding-box seed (faster, same Lloyd).
    """
    t0 = time.perf_counter()
    torch = _torch()
    dev = _device_of(device)
    pos = torch.as_tensor(_as_float32(positions), device=dev)
    m, d = pos.shape
    G = int(min(groups, m))
    if d != 2:
        raise ValueError("voronoi_cut needs (m, 2) positions")
    if G < 1:
        raise ValueError("groups must be >= 1")
    rng_t = torch.Generator(device=dev)
    rng_t.manual_seed(int(seed))
    if init == "kmeans++":
        centres = _kmeanspp_init(pos, G, dev, torch, rng_t)
    elif init == "bbox":
        xmin = pos.min(dim=0).values
        xmax = pos.max(dim=0).values
        centres = (xmin + (xmax - xmin) * torch.rand((G, 2), device=dev, generator=rng_t)).to(torch.float64)
    else:
        raise ValueError("init must be 'kmeans++' or 'bbox'")
    labels = torch.full((m,), -1, device=dev, dtype=torch.int64)
    converged = False
    it = -1
    chunk = 65_536

    def _assign_and_reduce( centres, labels_out=None):
        # returns (labels, new_centres, counts) computed in row chunks
        new_labels = torch.empty((m,), device=dev, dtype=torch.int64)
        new_centres = torch.zeros((G, 2), device=dev, dtype=torch.float64)
        counts = torch.zeros((G,), device=dev, dtype=torch.float64)
        pos64 = pos.to(torch.float64)
        for start in range(0, m, chunk):
            end = min(start + chunk, m)
            block = pos64[start:end]
            d2 = (torch.sum(block * block, dim=1, keepdim=True)
                  + torch.sum(centres * centres, dim=1)[None, :]
                  - 2.0 * (block @ centres.T))
            d2 = torch.clamp(d2, min=0.0)
            new_labels[start:end] = torch.argmin(d2, dim=1)
            new_centres.index_add_(0, new_labels[start:end], block)
            counts.index_add_(0, new_labels[start:end], torch.ones(end - start, device=dev, dtype=torch.float64))
        return new_labels, new_centres, counts

    for it in range(int(iterations)):
        new_labels, new_centres, counts = _assign_and_reduce(centres)
        if torch.equal(new_labels, labels):
            labels = new_labels
            converged = True
            break
        labels = new_labels
        valid = counts > 0
        new_centres = torch.where(valid[:, None], new_centres / counts[:, None], centres)
        # per-centre shift: mean L2 change over centres, so the tolerance is
        # comparable across group counts (a global L2 would grow with sqrt(G))
        shift = float(torch.sqrt(torch.mean(torch.sum((new_centres - centres) ** 2, dim=1))).detach())
        centres = new_centres
        if shift < tolerance:
            converged = True
            break
        if status is not None and (it % 10 == 0):
            status(it + 1, iterations, float(shift))
    # final objective in row chunks
    obj_sum = 0.0
    pos64 = pos.to(torch.float64)
    for start in range(0, m, chunk):
        end = min(start + chunk, m)
        block = pos64[start:end]
        d2 = (torch.sum(block * block, dim=1, keepdim=True)
              + torch.sum(centres * centres, dim=1)[None, :]
              - 2.0 * (block @ centres.T))
        d2 = torch.clamp(d2, min=0.0)
        obj_sum += float(d2.min(dim=1).values.sum().detach())
    objective = float(obj_sum / max(1, m))
    return VoronoiResult(
        labels=labels.to("cpu", dtype=torch.int64).numpy(),
        centers=centres.to("cpu", dtype=torch.float64).numpy(),
        converged=bool(converged),
        iterations=int(it + 1),
        wall_time=float(time.perf_counter() - t0),
        objective=objective,
        extras={"groups": int(G), "m": int(m), "device": str(dev)},
    )


@dataclass
class PushApartResult:
    centers: np.ndarray       # (G, 2) after repulsion
    iterations: int
    min_separation: float
    converged: bool
    wall_time: float
    extras: dict[str, Any] = field(default_factory=dict)


def push_apart(
    centers: np.ndarray,
    *,
    min_separation: float = 0.05,
    iterations: int = 50,
    seed: int = 0,
    device: str | None = None,
    status: Callable[[int, int, float], None] | None = None,
) -> PushApartResult:
    """Uniform repulsion: scale every centre away from the centroid until the
    minimum pairwise distance reaches ``min_separation`` (or the iteration
    budget expires)."""
    t0 = time.perf_counter()
    torch = _torch()
    dev = _device_of(device)
    c = torch.as_tensor(_as_float32(centers), device=dev).to(torch.float64)
    G = int(c.shape[0])
    if G < 2:
        return PushApartResult(
            centers=c.to("cpu", dtype=torch.float64).numpy(),
            iterations=0,
            min_separation=0.0,
            converged=True,
            wall_time=0.0,
            extras={"groups": G},
        )
    # Uniform repulsion: scale every centre away from the centroid so that the
    # minimum pairwise distance reaches ``min_separation`` (bounded per step so
    # intermediate states remain well-behaved).  Per-pair forces would get
    # clipped by a per-centre cap and never actually separate the closest pair;
    # rescaling grows every distance by the same factor and is monotone.
    target = float(np.maximum(min_separation, 1e-4))
    converged = False
    it = -1
    for it in range(int(iterations)):
        d2 = torch.sum((c[:, None, :] - c[None, :, :]) ** 2, dim=2)
        d2.fill_diagonal_(0.0)
        d = torch.sqrt(torch.clamp(d2, min=0.0))
        off = d > 0
        min_d = float(torch.where(off, d, torch.full_like(d, float("inf"))).min().detach())
        if min_d >= target or min_d == float("inf"):
            converged = True
            break
        # cap the growth per step at 1.5x so the sequence stays smooth
        scale = min(target / min_d, 1.5)
        centre = c.mean(dim=0)
        c = centre + (c - centre) * scale
        if status is not None and (it % 10 == 0):
            status(it + 1, iterations, min_d)
    d2 = torch.sum((c[:, None, :] - c[None, :, :]) ** 2, dim=2)
    d2.fill_diagonal_(0.0)
    d = torch.sqrt(torch.clamp(d2, min=0.0))
    min_d = float(torch.where(d > 0, d, torch.full_like(d, float("inf"))).min().detach())
    return PushApartResult(
        centers=c.to("cpu", dtype=torch.float64).numpy(),
        iterations=int(it + 1),
        min_separation=min_d,
        converged=bool(converged),
        wall_time=float(time.perf_counter() - t0),
        extras={"groups": int(G), "target": float(target)},
    )


@dataclass
class HierarchicalPlacement:
    """A full placement step: parent assignment + 2-D placement around parents."""

    child_positions: np.ndarray   # (n, 2) placed (n, 2)
    parent_ids: np.ndarray        # (n,)
    parent_xy: np.ndarray         # (P, 2)
    spacing: float


def hierarchical_place(
    points: np.ndarray,
    parent_features: np.ndarray,
    parent_xy: np.ndarray,
    *,
    spread: float = 0.3,
    seed: int = 0,
    device: str | None = None,
    local_spacing: bool = True,
) -> HierarchicalPlacement:
    """Assign ``points`` to nearest parents and place them around parent 2-D.

    Each child is placed at its parent's 2-D position plus a pseudo-random
    offset scaled to ``spread`` x the parent's local 2-D neighbour spacing
    (per-parent nearest-neighbour distance).  ``local_spacing=False`` falls
    back to the global median spacing for the whole cloud.
    """
    parents = nearest_parent(points, parent_features, device=device)
    sp_scale = _local_spacing(np.asarray(parent_xy, dtype=np.float64)) if local_spacing else None
    pos = place_around_parents(
        parents, parent_xy, spread=spread, seed=seed,
        local_spacing=local_spacing, parent_spacing=sp_scale,
    )
    if local_spacing:
        spacing = float(np.median(sp_scale))
    else:
        spacing = _neighbour_spacing(np.asarray(parent_xy, dtype=np.float64))
    return HierarchicalPlacement(
        child_positions=pos,
        parent_ids=parents,
        parent_xy=np.asarray(parent_xy, dtype=np.float64),
        spacing=spacing,
    )


@dataclass
class BasemapResult:
    """Output of :func:`run_basemap`.

    ``top_xy``/``mid_xy``/``fine_xy``/``tail_xy`` are the 2-D placements at
    each level; ``top_features``/``mid_features``/``fine_features`` are the
    centroid arrays that were embedded/placed.  ``voronoi``/``push_apart``
    are only populated when ``do_voronoi=True``.
    """

    top_features: np.ndarray       # (top_k, d)
    mid_features: np.ndarray | None  # (mid_k, d) or None when mid_k == 0
    fine_features: np.ndarray       # (fine_k, d)
    top_xy: np.ndarray              # (top_k, 2)
    mid_xy: np.ndarray | None       # (mid_k, 2)
    fine_xy: np.ndarray             # (fine_k, 2)
    tail_xy: np.ndarray             # (n, 2)
    tail_parents: np.ndarray        # (n,) fine-level parent index per row
    mid_parents: np.ndarray | None  # (mid_k,) top-level parent index per mid
    fine_parents: np.ndarray        # (fine_k,) mid-level parent index per fine
    fine_parents_to_top: np.ndarray | None  # (fine_k,) top-level parent via mid
    spacing: float
    wall_time: float
    voronoi: VoronoiResult | None = None
    push_apart: PushApartResult | None = None
    extras: dict = field(default_factory=dict)


def run_basemap(
    features: np.ndarray,
    *,
    top_k: int = 32_000,
    mid_k: int = 260_000,
    fine_k: int = 1_000_000,
    nepochs: int = 5,
    seed: int = 0,
    device: str | None = None,
    do_tail: bool = True,
    groups: int = 2_000,
    min_separation: float = 0.05,
    do_voronoi: bool = True,
    umap_config=None,
    status=None,
):
    """Run the MONET-style hierarchical 2-D placement pipeline.

    One feature matrix `features` (n, d) through the whole scheme:

    1. GPU e-means -> ``top_k`` top-level centroids and ``fine_k`` fine-level
       centroids (``mid_k`` optional mid-level; set ``mid_k=0`` to skip).
    2. UMAP the top centroids -> 2-D parent positions.
    3. place the mid centroids around the top level (nearest-parent, local spacing)
    4. place the fine centroids around the mid level (or directly around the
       top level when ``mid_k=0``)
    5. place every row around the fine centroids (optional, ``do_tail=True``)
    6. cut the fine centroids into ``groups`` Voronoi cells + push apart
    """
    import emaopt

    t0 = time.perf_counter()
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be (n, d)")
    n, d = features.shape
    top_k = int(min(top_k, n))
    mid_k = int(min(mid_k, n)) if mid_k else 0
    fine_k = int(min(fine_k, n))

    def _status(msg):
        if status is not None:
            status(msg)

    _status("fitting e-means (top)")
    top_fit = emaopt.emeans_gpu_fit(features, top_k, nepochs=nepochs, seed=seed, device=device)
    top_features = top_fit.centroids.astype(np.float32)

    mid_features = None
    mid_parents = None
    if mid_k:
        _status("fitting e-means (mid)")
        mid_fit = emaopt.emeans_gpu_fit(features, mid_k, nepochs=nepochs, seed=seed, device=device)
        mid_features = mid_fit.centroids.astype(np.float32)

    _status("fitting e-means (fine)")
    fine_fit = emaopt.emeans_gpu_fit(features, fine_k, nepochs=nepochs, seed=seed, device=device)
    fine_features = fine_fit.centroids.astype(np.float32)

    _status("UMAP top level")
    if umap_config is None:
        umap_config = emaopt.UMAPConfig(n_neighbors=15, n_epochs=200, burn_in_epoch=50, seed=seed)
    top_xy = emaopt.umap_embed(top_features, umap_config, device=device).embedding

    # mid placement: mid centroids around top centroids
    mid_xy = None
    if mid_k:
        mid_parents = nearest_parent(mid_features, top_features, device=device)
        mid_xy = place_around_parents(mid_parents, top_xy, spread=0.3, seed=seed,
                                      local_spacing=True)
    # fine placement: fine centroids around mid (or top when mid_k=0)
    if mid_k:
        fine_parents = nearest_parent(fine_features, mid_features, device=device)
        fine_parents_to_top = np.take(mid_parents, fine_parents)
        fine_xy = place_around_parents(fine_parents, mid_xy, spread=0.3, seed=seed,
                                       local_spacing=True)
    else:
        fine_parents = nearest_parent(fine_features, top_features, device=device)
        fine_xy = place_around_parents(fine_parents, top_xy, spread=0.3, seed=seed,
                                       local_spacing=True)
        fine_parents_to_top = fine_parents.copy()

    # tail: every row around the fine centroids
    if do_tail:
        _status("placing rows around fine centroids")
        tail_parents = nearest_parent(features, fine_features, device=device)
        tail_xy = place_around_parents(tail_parents, fine_xy, spread=0.3, seed=seed,
                                       local_spacing=True)
    else:
        tail_parents = np.full(n, -1, dtype=np.int64)
        tail_xy = np.zeros((n, 2), dtype=np.float64)

    # voronoi + push apart on the fine centroids
    voronoi = None
    push_apart_res = None
    if do_voronoi:
        _status("voronoi cut + push apart")
        voronoi = voronoi_cut(fine_xy, groups, iterations=50, seed=seed, device=device,
                             init="kmeans++")
        push_apart_res = push_apart(voronoi.centers, min_separation=min_separation,
                                    iterations=100, seed=seed, device=device)

    return BasemapResult(
        top_features=top_features,
        mid_features=mid_features,
        fine_features=fine_features,
        top_xy=top_xy,
        mid_xy=mid_xy,
        fine_xy=fine_xy,
        tail_xy=tail_xy,
        tail_parents=tail_parents,
        mid_parents=mid_parents,
        fine_parents=fine_parents,
        fine_parents_to_top=fine_parents_to_top,
        spacing=_local_spacing(fine_xy).mean() if fine_xy.shape[0] > 1 else 1.0,
        wall_time=float(time.perf_counter() - t0),
        voronoi=voronoi,
        push_apart=push_apart_res,
        extras={
            "n": int(n), "d": int(d),
            "top_k": int(top_k), "mid_k": int(mid_k), "fine_k": int(fine_k),
            "groups": int(groups),
        },
    )
