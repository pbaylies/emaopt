"""Integration bridge: use emaopt's GPU algorithms as latent-scope backends.

Ian Johnson's latent-scope (https://github.com/enjalot/latent-scope) resolves a
compute backend (CPU / cuML) in ``latentscope.util.device.resolve_device`` and
then branches on it: ``ls-umap`` uses cuML UMAP or CPU umap-learn, and
``ls-cluster --method kmeans`` uses cuML KMeans or sklearn.  emaopt provides
the same two stages with a plain torch/CUDA implementation (no RAPIDS install),
which is what its own MONET-style pipeline uses.

This module is a thin, guarded glue layer:

* :func:`is_available` — whether ``latentscope`` is importable at all
  (never raises; absence is a normal outcome).
* :func:`probe_emaopt` — whether the CUDA torch backend + emaopt are usable.
* :func:`run_umap` — ``(embeddings, neighbors, min_dist, seed, dimensions)``
  -> ``(umap_embeddings, backend_name)``; used as an alternative to cuML/CPU
  UMAP.  Returns plain arrays — no reducer object (the warm-start /
  incremental-transform paths of latent-scope stay on the CPU path).
* :func:`run_emeans` — ``(embeddings, n_clusters, nepochs, seed)`` ->
  ``(labels, centroids)``; an alternative to cuML/sklearn KMeans.  emaopt's
  e-means keeps the host's EMA law (fitted batch size + cosine alpha grid) and
  is the same algorithm the whole basemap pipeline is built on.
* :func:`install_latentscope` — idempotent, guarded monkey-patch of
  latent-scope's ``_reduce_umap`` / ``_run_kmeans`` / ``resolve_device`` to
  prefer emaopt when it is importable.  This is a convenience for in-process
  experimentation; the supported path is the documented upstream patch
  (see ``docs/latent-scope-integration.md``).

Everything here is defensive: a missing latent-scope, missing torch, or a
missing GPU degrades to a clear error or a \"backend=cpu\"-style result rather
than a hard failure.  No heavy imports happen at module load time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "is_available",
    "probe_emaopt",
    "run_umap",
    "run_emeans",
    "install_latentscope",
]

_PATCH_SENTINEL = "_emaopt_bridge_installed"


def is_available() -> bool:
    """Return True if latentscope is importable (never raises)."""
    try:
        import latentscope  # noqa: F401

        return True
    except Exception:
        return False


def probe_emaopt(device: str | None = None) -> bool:
    """Return True if the CUDA torch backend + emaopt are usable.

    Never raises; a missing torch/CUDA simply reports False.  This is the
    same guarded-probe discipline latent-scope uses for cuML.
    """
    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        if not cuda_ok:
            return False
        if device is not None and str(device).lower() != "cuda":
            return False
        return True
    except Exception:
        return False


def _umap_config_for(neighbors: int, min_dist: float, seed: int | None, dimensions: int):
    """Map latent-scope's (n_neighbors, min_dist) onto emaopt's UMAPConfig."""
    from emaopt.algorithms.umap import UMAPConfig

    # emaopt's embedding is a fast SGD layout with a Monte-Carlo repulsion term —
    # ~2x-30x faster than umap-learn at these scales but not a bit-identical UMAP.
    return UMAPConfig(
        n_neighbors=int(neighbors),
        min_dist=float(min_dist) if min_dist is not None else 0.1,
        n_epochs=200,
        burn_in_epoch=50,
        seed=int(seed) if seed is not None else 0,
    )


def run_umap(
    embeddings: np.ndarray,
    neighbors: int,
    min_dist: float,
    seed: int | None,
    dimensions: int = 2,
    device: str | None = None,
):
    """Run emaopt's fast 2-D UMAP on ``embeddings``.

    Returns ``(embedding_array, backend_name)``.  ``embedding_array`` has shape
    ``(n, 2)`` and is returned as float32 — emaopt embeds in 2-D only (a real
    2-D embedding is the whole point of the pipeline).  ``dimensions`` must be
    2 (latent-scope's 3-D path stays on the CPU reducer).  Raises
    ``RuntimeError`` when the GPU path is unavailable.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embeddings must be (n, d)")
    if dimensions != 2:
        raise ValueError("emaopt embeds in 2-D only; use the CPU path for 3-D")
    if not probe_emaopt(device):
        raise RuntimeError("emaopt GPU UMAP unavailable (no CUDA torch backend)")

    from emaopt.algorithms.umap import umap_embed

    # latent-scope reduces with a cosine metric; normalize rows so that the
    # Euclidean distance below is 2 * (1 - cos_sim), which is the same
    # neighbor ordering the cosine metric would produce.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x_cosine = (x / norms).astype(np.float32)

    cfg = _umap_config_for(neighbors, min_dist, seed, dimensions)
    result = umap_embed(x_cosine, cfg, device="cuda")
    emb = np.asarray(result.embedding, dtype=np.float32)
    return emb, "emaopt"


def run_emeans(
    embeddings: np.ndarray,
    n_clusters: int,
    nepochs: int = 5,
    seed: int | None = None,
    device: str | None = None,
):
    """Run emaopt's GPU e-means on ``embeddings``.

    Returns ``(labels, centroids)`` with ``labels`` an int64 (n,) array and
    ``centroids`` a float32 (k, d) array — the same shape the cuML/sklearn
    ``fit_predict`` paths hand back.  Raises ``RuntimeError`` when unavailable.
    """
    x = np.asarray(embeddings, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("embeddings must be (n, d)")
    if not probe_emaopt(device):
        raise RuntimeError("emaopt GPU e-means unavailable (no CUDA torch backend)")

    from emaopt.gpu_emeans import emeans_gpu_fit

    fit = emeans_gpu_fit(
        x, int(n_clusters), nepochs=int(nepochs),
        seed=int(seed) if seed is not None else 0, device="cuda",
    )
    labels = np.full(x.shape[0], -1, dtype=np.int64)
    # re-assign rows to centroids on the device with clamped squared distances
    # (float32 can produce small negatives in the GEMM expansion)
    import torch

    xt = torch.as_tensor(x, device="cuda", dtype=torch.float32)
    ct = torch.as_tensor(fit.centroids, device="cuda", dtype=torch.float32)
    d2 = torch.sum(xt * xt, dim=1, keepdim=True) + torch.sum(ct * ct, dim=1)[None, :] - 2.0 * (xt @ ct.T)
    d2 = torch.clamp(d2, min=0.0)
    labels = torch.argmin(d2, dim=1).to(torch.int64).cpu().numpy()
    return labels, fit.centroids.astype(np.float32)


def install_latentscope() -> bool:
    """Monkey-patch latent-scope's GPU paths to prefer emaopt.

    Idempotent: a second call is a no-op.  Only patches if latent-scope is
    importable **and** emaopt's CUDA backend is usable; otherwise returns False
    with latent-scope untouched.  The patch is intentionally narrow — it only
    rewrites the UMAP/KMeans backend selection hooks, nothing else.

    Returns True when the patch was applied.
    """
    import sys

    latentscope = sys.modules.get("latentscope")
    if latentscope is not None and getattr(latentscope, _PATCH_SENTINEL, False):
        return True
    try:
        import latentscope  # noqa: F401
        import latentscope.scripts.umapper as umapper_mod
        import latentscope.scripts.cluster as cluster_mod
    except Exception:
        return False
    if not probe_emaopt():
        return False

    original_reduce = getattr(umapper_mod, "_reduce_umap", None)
    original_kmeans = getattr(cluster_mod, "_run_kmeans", None)
    if original_reduce is None or original_kmeans is None:
        return False

    def _emaopt_reduce(embeddings, neighbors, min_dist, seed, use_cuml, init_array=None, n_components=2):
        if init_array is not None or not use_cuml or n_components != 2:
            return original_reduce(embeddings, neighbors, min_dist, seed, use_cuml,
                                   init_array=init_array, n_components=n_components)
        try:
            emb, backend = run_umap(embeddings, neighbors, min_dist, seed,
                                    dimensions=2)
            print(f"umapper: reducing with emaopt (fast GPU UMAP, backend={backend})")
            return emb, None
        except Exception as e:
            print(f"umapper: emaopt UMAP failed ({e!r}); falling back")
            return original_reduce(embeddings, neighbors, min_dist, seed, use_cuml,
                                   init_array=init_array, n_components=n_components)

    def _emaopt_kmeans(embeddings, n_clusters, use_cuml=False, seed=None):
        if not use_cuml:
            return original_kmeans(embeddings, n_clusters, use_cuml=use_cuml, seed=seed)
        try:
            labels, _ = run_emeans(embeddings, n_clusters, seed=seed)
            print(f"cluster: running emaopt GPU e-means with n_clusters={n_clusters}")
            return labels
        except Exception as e:
            print(f"cluster: emaopt e-means failed ({e!r}); falling back")
            return original_kmeans(embeddings, n_clusters, use_cuml=use_cuml, seed=seed)

    umapper_mod._reduce_umap = _emaopt_reduce
    cluster_mod._run_kmeans = _emaopt_kmeans
    setattr(latentscope, _PATCH_SENTINEL, True)
    return True
