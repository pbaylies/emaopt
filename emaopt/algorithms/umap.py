"""Fast UMAP-style 2D embedding (not a bit-exact port of the reference).

The basemap pipeline (UMAP on a sample, e-means to centroids, UMAP the top-level
centroids, place the long tail back next to their parents) needs a fast,
memory-bounded embedding of a *sample* of rows.  This module is that:

1. :func:`find_neighbors` -- exact chunked k-NN (torch GEMM, topk per chunk).
2. :func:`fuzzy_membership` -- fuzzy-set membership from kNN distances
   (exponential kernel, per-row scale from the k-th neighbour).
3. :func:`umap_embed` -- SGD layout on an attractive term (fuzzy weights) plus
   a negative-sampled repulsive term.  NOT bit-identical to UMAP: the repulsive
   term is a Monte-Carlo estimate, so this is fast rather than a reference-grade
   embedding.

Init is a 2-D PCA basis (cheap SVD of the Gram matrix, no full-n SVD).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

__all__ = ["UMAPConfig", "UMAPResult", "find_neighbors", "fuzzy_membership", "umap_embed"]


@dataclass(frozen=True)
class UMAPConfig:
    n_neighbors: int = 15
    min_dist: float = 0.1
    burn_in_epoch: int = 50
    n_epochs: int = 200
    negative_sample_rate: int = 5
    learning_rate: float = 1.0
    init_method: str = "pca"          # "pca" | "random"
    chunk_size: int = 1024
    seed: int = 0
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if int(self.n_neighbors) < 1:
            raise ValueError("n_neighbors must be >= 1")
        if int(self.n_epochs) < 1:
            raise ValueError("n_epochs must be >= 1")
        if not 0.0 < float(self.min_dist) < 1.0:
            raise ValueError("min_dist must be in (0, 1)")
        if self.init_method not in ("pca", "random"):
            raise ValueError("init_method must be 'pca' or 'random'")
        if self.dtype not in ("float32", "float64", "bfloat16"):
            raise ValueError("dtype must be float32, float64 or bfloat16")
        if int(self.burn_in_epoch) < 0:
            raise ValueError("burn_in_epoch must be >= 0")


@dataclass
class UMAPResult:
    embedding: np.ndarray           # (n, 2) in [0, 1]
    n_neighbors: int
    n_epochs: int
    burn_in_epoch: int
    wall_time: float
    seed: int
    init_method: str
    method: str
    extras: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "algorithm": "umap_fast",
            "n": int(self.embedding.shape[0]),
            "d": int(self.embedding.shape[1]),
            "n_neighbors": int(self.n_neighbors),
            "n_epochs": int(self.n_epochs),
            "burn_in_epoch": int(self.burn_in_epoch),
            "wall_time": float(self.wall_time),
            "seed": int(self.seed),
            "init_method": self.init_method,
            "method": self.method,
        }


def _torch():
    try:
        import torch
    except ImportError as e:  # pragma: no cover - import guard
        raise ImportError(
            "umap_embed needs torch (pip install torch)"
        ) from e
    return torch


def _device_of(x: np.ndarray, device: str | None) -> str:
    if device is not None:
        return device
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _as_tensor(x: np.ndarray, device: str, dtype: str):
    torch = _torch()
    t = torch.as_tensor(np.asarray(x)).to(torch.float32)
    if device != "cpu":
        t = t.to(device)
    if dtype == "float64":
        t = t.to(torch.float64)
    elif dtype == "bfloat16":
        t = t.to(torch.bfloat16)
    return t


def find_neighbors(
    x: np.ndarray,
    k: int,
    *,
    device: str | None = None,
    chunk_size: int = 1024,
    method: str = "chunked",
    dtype: str = "float32",
) -> tuple[np.ndarray, np.ndarray]:
    """Exact chunked kNN over ``x`` (n, d).

    Returns ``(labels, dists)``: ``labels`` (n, k) int64 and ``dists`` (n, k)
    float32, nearest first, self excluded. ``method="faiss"`` uses faiss-cpu
    (when installed), otherwise falls back to the torch chunked path.
    """
    x = np.asarray(x)
    n, d = x.shape
    if n < 2:
        raise ValueError("need at least 2 rows")
    k = int(min(k, n - 1))

    if method == "faiss":
        try:
            import faiss  # type: ignore
        except ImportError:
            method = "chunked"
        else:
            index = faiss.IndexFlatL2(d)
            index.add(np.ascontiguousarray(x, dtype=np.float32))
            dists, labels = index.search(np.ascontiguousarray(x, dtype=np.float32), k + 1)
            labels = labels[:, 1:]
            dists = dists[:, 1:]
            order = np.argsort(dists, axis=1)
            labels = np.take_along_axis(labels, order, 1)
            dists = np.sqrt(np.clip(np.take_along_axis(dists, order, 1), 0.0, None))
            return labels.astype(np.int64), dists.astype(np.float32)

    torch = _torch()
    dev = _device_of(x, device)
    xt = _as_tensor(x, dev, dtype)
    labels_out = np.empty((n, k), dtype=np.int64)
    dists_out = np.empty((n, k), dtype=np.float32)
    xb = torch.sum(xt * xt, dim=1)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        chunk = xt[start:end]
        d2 = (
            torch.sum(chunk * chunk, dim=1, keepdim=True)
            + xb[None, :]
            - 2.0 * (chunk @ xt.T)
        )
        d2 = torch.clamp(d2, min=0.0)
        for j, gi in enumerate(range(start, end)):
            d2[j, gi] = float("inf")
        top = torch.topk(d2, k, dim=1, largest=False)
        labels_out[start:end] = top.indices.to("cpu", dtype=torch.int64).numpy()
        dists_out[start:end] = torch.sqrt(top.values.clamp(min=0.0)).to("cpu", dtype=torch.float32).numpy()
    return labels_out, dists_out


def fuzzy_membership(
    dists: np.ndarray,
    *,
    n_neighbors: int | None = None,
) -> np.ndarray:
    """(n, k) fuzzy-set membership from kNN distances.

    Exponential kernel: per-row scale ``sigma_i`` from the k-th neighbour
    distance, ``r_i`` its value, and ``w_ij = exp(-max(0, d_ij - r_i) / sigma_i)``.
    The k-th weight self-normalises to ~1; neighbours closer than ``r_i`` are
    clipped at 1.
    """
    dists = np.asarray(dists, dtype=np.float32)
    if dists.ndim != 2:
        raise ValueError("dists must be (n, k)")
    k = dists.shape[1] if n_neighbors is None else int(n_neighbors)
    if k < 1 or k > dists.shape[1]:
        raise ValueError("n_neighbors must be in [1, k]")
    dists = dists[:, :k]
    sigma = np.maximum(dists[:, -1], 1e-12)
    r = dists[:, -1:]
    weights = np.exp(-np.maximum(0.0, dists - r) / sigma[:, None])
    return weights.astype(np.float32)


def _pca_init(xt, torch):
    """2-D PCA basis from the Gram eigendecomposition (never an n x n SVD)."""
    n, d = xt.shape
    xc = xt - xt.mean(dim=0, keepdim=True)
    gram = xc.T @ xc
    evals, evecs = torch.linalg.eigh(gram)
    evecs = evecs[:, -2:]
    # ascending eigen values: flip so column 0 is the top component
    evecs = evecs.flip(1)
    return (xc @ evecs).to(torch.float32)


def umap_embed(
    x: np.ndarray,
    config: UMAPConfig | None = None,
    *,
    n_neighbors: int | None = None,
    device: str | None = None,
    method: str = "chunked",
    status: Callable[[int, int, float], None] | None = None,
) -> UMAPResult:
    """Embed ``x`` (n, d) to (n, 2) UMAP-style.

    ``device`` defaults to cuda when available.  ``method`` is forwarded to
    :func:`find_neighbors`.  ``status`` optional callable ``(epoch, total, loss)``.
    """
    t0 = time.perf_counter()
    torch = _torch()
    cfg = config or UMAPConfig()
    if n_neighbors is not None:
        cfg = UMAPConfig(
            n_neighbors=int(n_neighbors),
            min_dist=cfg.min_dist,
            burn_in_epoch=cfg.burn_in_epoch,
            n_epochs=cfg.n_epochs,
            negative_sample_rate=cfg.negative_sample_rate,
            learning_rate=cfg.learning_rate,
            init_method=cfg.init_method,
            chunk_size=cfg.chunk_size,
            seed=cfg.seed,
            dtype=cfg.dtype,
        )
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be (n, d)")
    n, d = x.shape
    if n < 2:
        raise ValueError("x must have at least 2 rows")
    k = min(cfg.n_neighbors, n - 1)

    dev = _device_of(x, device)
    xt = _as_tensor(x, dev, cfg.dtype)

    labels, dists = find_neighbors(
        x, k, device=dev, chunk_size=cfg.chunk_size, method=method, dtype=cfg.dtype
    )
    labels_t = torch.as_tensor(labels, device=dev)
    dists_t = torch.as_tensor(dists, device=dev).to(torch.float32)
    weights_t = torch.as_tensor(
        fuzzy_membership(dists_t.cpu().numpy()), device=dev
    ).to(torch.float32)

    if cfg.init_method == "pca":
        pos = _pca_init(xt, torch).to(torch.float32)
    else:
        pos = torch.randn(n, 2, device=dev, dtype=torch.float32) * 0.01

    burn_in = min(cfg.burn_in_epoch, cfg.n_epochs)
    neg_rate = int(cfg.negative_sample_rate)
    lr = float(cfg.learning_rate)
    min_dist = float(cfg.min_dist)

    rng_gen = torch.Generator(device=dev)
    rng_gen.manual_seed(int(cfg.seed))

    best_loss = None
    chunk = 2048 if n > 200_000 else min(n, 2048)

    pos = pos.detach().clone().requires_grad_(True)

    for epoch in range(cfg.n_epochs):
        loss = torch.zeros((), device=dev, dtype=torch.float32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pos_chunk = pos[start:end]
            # attractive: fuzzy weights x squared 2-D distance to neighbours
            nb_pos = pos[labels_t[start:end]]                      # (chunk, k, 2)
            d_att2 = torch.sum((pos_chunk[:, None, :] - nb_pos) ** 2, dim=2)
            loss = loss + torch.sum(weights_t[start:end] * d_att2)
            # repulsive: sampled negatives, min_dist^2/(min_dist^2 + d^2)
            neg_idx = torch.randint(0, n, (end - start, neg_rate), device=dev, generator=rng_gen)
            nb_neg = pos[neg_idx]                                  # (chunk, neg_rate, 2)
            d_neg2 = torch.sum((pos_chunk[:, None, :] - nb_neg) ** 2, dim=2)
            loss = loss + torch.sum(min_dist * min_dist / (min_dist * min_dist + d_neg2 + 1e-12))

        loss.backward()
        grad = pos.grad
        gn = grad.norm(dim=1, keepdim=True)
        lr_e = lr * (2.0 if epoch < burn_in else 1.0)
        with torch.no_grad():
            pos -= lr_e * grad / (gn + 1e-8)
            pos.grad = None

        loss_val = float(loss.detach())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
        if status is not None and (epoch % max(1, cfg.n_epochs // 10) == 0 or epoch == cfg.n_epochs - 1):
            status(epoch + 1, cfg.n_epochs, loss_val)

    emb = pos.detach().cpu().numpy().astype(np.float64)
    emb_min = emb.min(axis=0)
    emb_max = emb.max(axis=0)
    emb = (emb - emb_min) / np.maximum(emb_max - emb_min, 1e-12)
    total = time.perf_counter() - t0
    return UMAPResult(
        embedding=emb,
        n_neighbors=int(k),
        n_epochs=int(cfg.n_epochs),
        burn_in_epoch=int(burn_in),
        wall_time=float(total),
        seed=int(cfg.seed),
        init_method=cfg.init_method,
        method=method,
        extras={"min_dist": min_dist, "best_loss": best_loss, "device": str(dev),
                "n": int(n), "d": int(d)},
    )
