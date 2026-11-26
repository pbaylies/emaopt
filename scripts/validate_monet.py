#!/usr/bin/env python
"""Validate the basemap pipeline against the real MONET dataset.

    python scripts/validate_monet.py                 # 8 CC12M files (~71k rows, 1536-dim)
    python scripts/validate_monet.py --sample 20000 --k 5000 --voronoi-groups 1000
    python scripts/validate_monet.py --self-test     # offline synthetic mode (CI-safe)

Downloads the first ``--files`` v1.2.0 parquet files from jasperai/monet
(only the embedding column is loaded) and runs the four pipeline stages:

    1. UMAP the sample            2. GPU e-means (k=5000 default)
    3. UMAP the top-level centroids   4. place every row next to its parent
    5. cut the top level into Voronoi cells + push_apart

This is a downscale of the full scheme (Ian's post: 6M-row UMAP, 32k/260k/1m
centroids, 2k voronoi cells over 104M rows).  The point is to calibrate the
pipeline at real scale — 71.5k rows x 1536 dims — not to reproduce it exactly.
Results print as JSON to stdout and may be saved with --out.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=int, default=8, help="number of first parquet files to load")
    ap.add_argument("--sample", type=int, default=20_000, help="UMAP sample size")
    ap.add_argument("--k", type=int, default=5_000, help="e-means centroids")
    ap.add_argument("--voronoi-groups", type=int, default=1_000)
    ap.add_argument("--nepochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-test", action="store_true", help="offline synthetic mode")
    ap.add_argument("--out", type=str, default="", help="write JSON record here")
    args = ap.parse_args()

    import pandas as pd
    import emaopt

    if args.self_test:
        # offline synthetic stand-in: same shapes, no download
        rng = np.random.default_rng(args.seed)
        emb = rng.normal(size=(20_000, 1536)).astype(np.float32)
        dataset = "synthetic"
    else:
        from huggingface_hub import hf_hub_download
        files = [
            hf_hub_download("jasperai/monet", "v1.2.0/cc12m/0-0/%06d.parquet" % i,
                            repo_type="dataset")
            for i in range(args.files)
        ]
        print(f"loading {len(files)} MONET files...", flush=True)
        parts = []
        for p in files:
            df = pd.read_parquet(p, columns=["embedding_dinov2-vitg14"])
            parts.append(np.stack([np.asarray(v, dtype=np.float32)
                                   for v in df["embedding_dinov2-vitg14"]]))
        emb = np.concatenate(parts, axis=0)
        dataset = "jasperai/monet v1.2.0 cc12m"

    n, d = emb.shape
    print(f"dataset={dataset}  n={n}  d={d}", flush=True)
    record = {"dataset": dataset, "n": int(n), "d": int(d), "seed": args.seed}

    # stage 1: UMAP the sample
    idx = np.random.default_rng(args.seed).choice(n, size=min(args.sample, n), replace=False)
    t0 = time.perf_counter()
    res = emaopt.umap_embed(emb[idx], emaopt.UMAPConfig(
        n_neighbors=15, n_epochs=200, burn_in_epoch=50, seed=args.seed), device="cuda")
    record["stage1_umap_sample_s"] = round(time.perf_counter() - t0, 2)

    # stage 2: GPU e-means
    t0 = time.perf_counter()
    fit = emaopt.emeans_gpu_fit(emb, k=args.k, nepochs=args.nepochs, seed=args.seed, device="cuda")
    record["stage2_emeans_s"] = round(time.perf_counter() - t0, 2)
    record["e_means_objective"] = round(float(fit.objective), 4)
    record["dead_clusters"] = int(fit.extras["dead_clusters"])

    # stage 3: UMAP the top level
    t0 = time.perf_counter()
    top = emaopt.umap_embed(fit.centroids, emaopt.UMAPConfig(
        n_neighbors=15, n_epochs=200, burn_in_epoch=50, seed=args.seed), device="cuda")
    parent_xy = top.embedding
    record["stage3_umap_top_s"] = round(time.perf_counter() - t0, 2)

    # stage 4: place every row next to its parent
    t0 = time.perf_counter()
    pids = emaopt.nearest_parent(emb, fit.centroids, device="cuda")
    placed = emaopt.place_around_parents(pids, parent_xy, spread=0.3, seed=args.seed)
    disp = np.linalg.norm(placed - parent_xy[pids], axis=1)
    record["stage4_place_s"] = round(time.perf_counter() - t0, 2)
    record["mean_displacement"] = float(disp.mean())

    # stage 5: voronoi + push apart
    t0 = time.perf_counter()
    vr = emaopt.voronoi_cut(parent_xy, args.voronoi_groups, iterations=60,
                           seed=args.seed, device="cuda")
    pa = emaopt.push_apart(vr.centers, min_separation=0.05, iterations=100,
                           seed=args.seed, device="cuda")
    record["stage5_voronoi_s"] = round(time.perf_counter() - t0, 2)
    record["voronoi_groups"] = int(vr.extras["groups"])
    record["voronoi_labels_used"] = int(len(np.unique(vr.labels)))
    record["voronoi_converged"] = bool(vr.converged)
    record["push_apart_converged"] = bool(pa.converged)
    record["push_apart_min_separation"] = float(pa.min_separation)

    print(json.dumps(record, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"recorded -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
