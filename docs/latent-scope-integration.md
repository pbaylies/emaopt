# Integrating emaopt with latent-scope

emaopt is a standalone GPU/streaming library. latent-scope (enjalot's tooling,
`latentscope` on PyPI) is a different project — it has its own UMAP/clustering
pipeline with CPU fallbacks (umap-learn / sklearn) and an optional cuML GPU
path. The integration makes emaopt's GPU algorithms available as latent-scope's
backend, so a latent-scope install that also has `emaopt` gets the same fast
GPU embedding and e-means it uses for its own basemap scheme.

## How it works

The bridge lives in `emaopt.integrations` (guarded, off by default):

* `is_available()` — is latent-scope importable at all?
* `probe_emaopt()` — is emaopt usable (CUDA torch + emaopt)?
* `run_umap(embeddings, neighbors, min_dist, seed)` — emaopt's fast 2-D UMAP
* `run_emeans(embeddings, n_clusters, nepochs, seed)` — emaopt's GPU e-means
* `install_latentscope()` — idempotent monkey-patch of latent-scope's
  `_reduce_umap` / `_run_kmeans` to prefer emaopt when present (used by the
  in-process smoke test; see below)

Everything is guarded: the bridge never raises on import, and every call
falls back to latent-scope's own CPU/cuML path with a logged reason when
emaopt is missing or the GPU is unavailable.

## Requirements

```bash
# In latent-scope's venv (or any environment with torch + CUDA):
pip install emaopt   # our library (this repo)
# latent-scope's own GPU extra is still optional and left alone.
```

## Usage

### In-process (experiments, testing)

```python
import emaopt.integrations as integ
integ.install_latentscope()   # idempotent; patches latent-scope's _reduce_umap / _run_kmeans
import latentscope.scripts.umapper  # emaopt path is now used inside _reduce_umap
```

This is what the integration is smoke-tested with (see `tests/test_latentscope_bridge.py`).

### Upstream patch (recommended for production use)

The in-process monkey-patch is convenient but not idiomatic for a deployed
latent-scope. For a permanent integration, apply the patch to latent-scope
itself. The patch is kept in this directory as the source of truth; it touches
three files:

1. `latentscope/util/device.py` — add `_emaopt_importable()` (guarded probe,
   mirrors `_cuml_importable`), a `use_emaopt` field on `DeviceResolution`,
   and make the `cuda`/`auto` branches set `use_emaopt=True` when present
   (clearest precedence: emaopt > cuML > CPU). `use_cuml` stays as-is.
2. `latentscope/scripts/umapper.py` — in `_reduce_umap`, branch on
   `use_emaopt` before cuML: call `emaopt.integrations.run_umap` and return
   `(embedding, None)`. Warm-start (`init_array`) and 3-D paths stay CPU.
3. `latentscope/scripts/cluster.py` — in `_run_kmeans`, branch on
   `use_emaopt` before cuML: call `emaopt.integrations.run_emeans` and return
   `(labels, centroids)` — the same shape `_run_kmeans` returns today.

Keep the existing `use_cuml` semantics untouched; emaopt is an *additional*
backend, not a replacement.

### What emaopt gives you in latent-scope

| stage | latent-scope default | emaopt backend |
| --- | --- | --- |
| UMAP (2-D) | umap-learn / cuML UMAP | fast SGD-layout embedding (AR I-perfect on synthetic blobs) |
| KMeans / e-means | sklearn KMeans / cuML KMeans | GPU e-means (host's EMA law) |

Honest limits: emaopt's UMAP is a fast style-equivalent, not a bit-identical
UMAP ( Monte-Carlo negation); it is 2-D only; it does not implement
latent-scope's `--save` / `--transform-from` reducer-transform path (stay on
the CPU reducer there). e-means has no cluster revival on the device arm
(host RNG walk; latent-scope's default kmeans path is unaffected).

## What we verified (2026-11-23, RTX 4090)

* `install_latentscope()` -> True against a fresh clone of
  `enjalot/latent-scope` (main branch): `_reduce_umap(use_cuml=True)` returns a
  finite (900, 2) embedding in 0.7 s (emaopt) and `_run_kmeans(use_cuml=True)`
  recovers 3 synthetic blobs with ARI 1.0 in 0.1 s, via the bridge.
* `emaopt.integrations.run_umap` / `run_emeans` standalone: shapes, dtypes,
  finiteness, ARI 1.0 on blobs (see `tests/test_latentscope_bridge.py`).
* `install_latentscope()` is idempotent and never raises; without
  latent-scope installed it returns False.
