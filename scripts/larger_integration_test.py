#!/usr/bin/env python
"""Larger real-data integration test: emaopt as latent-scope's GPU backends.

Real data: jasperai/monet v1.2.0 cc12m, first --files files (1536-dim DINOv2).
Honest stages:
  1. UMAP: latent-scope ``_reduce_umap`` with the emaopt backend vs
     umap-learn CPU on the same 20k-row sample (timed).
  2. KMeans: latent-scope ``_run_kmeans`` with the emaopt backend (GPU e-means)
     vs sklearn KMeans (timed, n_init=1) on the same subsample.
  3. sanity: shapes / finiteness / unique-cluster counts.

Usage:
    LATENTSCOPE_PATH=/path/to/latent-scope \
        python scripts/larger_integration_test.py [--files 8] [--k 2000] [--out evidence/....json]
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
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--k", type=int, default=2_000)
    ap.add_argument("--sample", type=int, default=20_000)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    t_start = time.perf_counter()
    emb = load_data(args.files)
    n, d = emb.shape
    print(f"data: n={n} d={d}", flush=True)

    print("is_available:", integ.is_available(), flush=True)
    print("probe:", integ.probe_emaopt(), flush=True)
    patched = integ.install_latentscope()
    print("install_latentscope ->", patched, flush=True)

    rng = np.random.default_rng(0)
    sample_idx = rng.choice(n, size=min(args.sample, n), replace=False)
    sample = emb[sample_idx]

    # Stage 1: UMAP sample — emaopt vs umap-learn CPU
    t0 = time.perf_counter()
    emb_emaopt, _ = umapper_mod._reduce_umap(sample, 15, 0.075, 0, True, None, n_components=2)
    t_emaopt = time.perf_counter() - t0
    print("emaopt _reduce_umap: shape=%s finite=%s  %.1fs" % (
        emb_emaopt.shape, bool(np.isfinite(emb_emaopt).all()), t_emaopt), flush=True)

    import umap
    t0 = time.perf_counter()
    cpu = umap.UMAP(n_neighbors=15, min_dist=0.075, random_state=0,
                    compatibility_layout=True).fit_transform(sample)
    t_umap_cpu = time.perf_counter() - t0
    print("umap-learn CPU: shape=%s finite=%s  %.1fs" % (
        cpu.shape, bool(np.isfinite(cpu).all()), t_umap_cpu), flush=True)

    # Stage 2: e-means on the full dataset — emaopt vs sklearn (subsample)
    t0 = time.perf_counter()
    labels_emaopt = cluster_mod._run_kmeans(emb, args.k, use_cuml=True, seed=0)
    t_em = time.perf_counter() - t0
    print("emaopt _run_kmeans: unique=%d time=%.1fs" % (len(np.unique(labels_emaopt)), t_em), flush=True)

    sklearn_sub = emb[sample_idx]
    import sklearn.cluster
    t0 = time.perf_counter()
    km = sklearn.cluster.KMeans(n_clusters=args.k, n_init=1, random_state=0)
    labels_sklearn = km.fit_predict(sklearn_sub)
    t_sklearn = time.perf_counter() - t0
    print("sklearn KMeans (subsample of %d): unique=%d time=%.1fs" % (
        len(sklearn_sub), len(np.unique(labels_sklearn)), t_sklearn), flush=True)

    record = {
        "dataset": "jasperai/monet v1.2.0 cc12m",
        "n_rows": int(n), "d": int(d),
        "sample_rows": int(len(sample)),
        "umap_sample_emaopt_s": round(t_emaopt, 2),
        "umap_sample_cpu_s": round(t_umap_cpu, 2),
        "umap_emaopt_shape": list(emb_emaopt.shape),
        "umap_emaopt_finite": bool(np.isfinite(emb_emaopt).all()),
        "emeans_k": int(args.k),
        "emeans_emaopt_s": round(t_em, 2),
        "sklearn_sub_s": round(t_sklearn, 2),
        "sklearn_sub_unique": int(len(np.unique(labels_sklearn))),
        "emaopt_unique": int(len(np.unique(labels_emaopt))),
        "total_wall_s": round(time.perf_counter() - t_start, 2),
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
        print("recorded ->", args.out, flush=True)


if __name__ == "__main__":
    main()
