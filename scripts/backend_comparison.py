#!/usr/bin/env python
"""Fair 3-backend comparison on real data: cuML vs emaopt vs CPU.

Real data: jasperai/monet v1.2.0 cc12m (1536-dim DINOv2 embeddings).
Backends (select with --backend):
  cuml   - latent-scope's own GPU path (cuml UMAP / cuml KMeans), no bridge
  emaopt - emaopt's fast GPU backend via install_latentscope()
  cpu    - latent-scope's own CPU path (umap-learn / sklearn)

Usage:
    LATENTSCOPE_PATH=/home/pb/.hermes/cache/scratch/latent-scope \
        python scripts/backend_comparison.py --backend cuml [--sample 20000] [--k 2000] [--out evidence/...json]
"""
import os
import sys
import time
import json
import argparse

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

LS_PATH = os.environ.get("LATENTSCOPE_PATH", "")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if LS_PATH:
    sys.path.insert(0, LS_PATH)
sys.path.insert(0, REPO)

import latentscope.scripts.umapper as umapper_mod
import latentscope.scripts.cluster as cluster_mod
import emaopt.integrations as integ


def load_data(files):
    paths = [hf_hub_download("jasperai/monet", "v1.2.0/cc12m/0-0/%06d.parquet" % i,
                             repo_type="dataset") for i in range(files)]
    parts = []
    for p in paths:
        df = pd.read_parquet(p, columns=["embedding_dinov2-vitg14"])
        parts.append(np.stack([np.asarray(v, dtype=np.float32)
                               for v in df["embedding_dinov2-vitg14"]]))
    return np.concatenate(parts, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", choices=["cuml", "emaopt", "cpu"], required=True)
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--k", type=int, default=2_000)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    t_start = time.perf_counter()
    emb = load_data(args.files)
    n, d = emb.shape
    print(f"backend={args.backend} data: n={n} d={d}", flush=True)

    patched = False
    if args.backend == "emaopt":
        patched = integ.install_latentscope()
        print("install_latentscope ->", patched, flush=True)

    use_cuml = args.backend in ("cuml", "emaopt")

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(args.sample, n), replace=False)
    sample = emb[sample_idx]

    # ---- UMAP on the sample ----
    t0 = time.perf_counter()
    emb_out, _ = umapper_mod._reduce_umap(sample, 15, 0.075, 0, use_cuml, None, n_components=2)
    umap_s = time.perf_counter() - t0
    print("UMAP: shape=%s finite=%s  %.1fs" % (
        emb_out.shape, bool(np.isfinite(emb_out).all()), umap_s), flush=True)

    # ---- KMeans/e-means on the full dataset ----
    t0 = time.perf_counter()
    labels_out = cluster_mod._run_kmeans(emb, args.k, use_cuml=use_cuml, seed=0)
    kmeans_s = time.perf_counter() - t0
    print("KMeans: unique=%d time=%.1fs" % (len(np.unique(labels_out)), kmeans_s), flush=True)

    # ---- CPU sanity on the same sample (only when not already CPU) ----
    cpu_umap_s = None
    cpu_kmeans_s = None
    if args.backend != "cpu":
        t0 = time.perf_counter()
        import umap
        cpu_embed = umap.UMAP(n_neighbors=15, min_dist=0.075, random_state=0,
                              compatibility_layout=True).fit_transform(sample)
        cpu_umap_s = time.perf_counter() - t0
        print("umap-learn CPU (reference): shape=%s %.1fs" % (cpu_embed.shape, cpu_umap_s), flush=True)
        import sklearn.cluster
        t0 = time.perf_counter()
        sklearn_labels = sklearn.cluster.KMeans(n_clusters=args.k, n_init=1, random_state=0).fit_predict(sample)
        cpu_kmeans_s = time.perf_counter() - t0
        print("sklearn KMeans (reference, subsample): unique=%d %.1fs" % (
            len(np.unique(sklearn_labels)), cpu_kmeans_s), flush=True)

    record = {
        "backend": args.backend,
        "n_rows": int(n), "d": int(d),
        "sample_rows": int(len(sample)),
        "umap_s": round(umap_s, 2),
        "umap_shape": list(emb_out.shape),
        "umap_finite": bool(np.isfinite(emb_out).all()),
        "kmeans_k": int(args.k),
        "kmeans_s": round(kmeans_s, 2),
        "kmeans_unique": int(len(np.unique(labels_out))),
        "total_wall_s": round(time.perf_counter() - t_start, 2),
        "cpu_umap_s_reference": round(cpu_umap_s, 2) if cpu_umap_s is not None else None,
        "cpu_kmeans_s_reference": round(cpu_kmeans_s, 2) if cpu_kmeans_s is not None else None,
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)


if __name__ == "__main__":
    main()
