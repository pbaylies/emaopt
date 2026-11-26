#!/usr/bin/env python
"""Timing + quality for the basemap stages on one GPU.

    python scripts/bench_basemap.py
    python scripts/bench_basemap.py --rows 30000 --k 1000 --rounds 3
    python scripts/bench_basemap.py --out evidence/basemap_bench.json

Stages, all device-first on the GPU when available:

* kNN: find_neighbors (exact chunked, top-15)
* embedding: umap_embed on a sample (fuzzy membership + SGD layout)
* e-means: emeans_gpu_fit vs the host emeans (both, for parity)
* placement: nearest_parent + place_around_parents (2-D around parents)
* voronoi: voronoi_cut (Lloyd) + push_apart (bounded repulsion)

Each candidate runs once per round (interleaved), and the reported figure is
the median.  Quality numbers are reported alongside timing: objective gap,
centroid displacement, label match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import emaopt  # noqa: E402


def _median(rs):
    rs = [r for r in rs if r is not None]
    if not rs:
        return None
    return float(np.median(rs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=30000, help="rows per stage")
    ap.add_argument("--k", type=int, default=1000, help="e-means centroids")
    ap.add_argument("--d", type=int, default=128, help="feature dim")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--nepochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="write JSON record here")
    args = ap.parse_args()

    torch = None
    try:
        import torch as _t
        torch = _t
    except ImportError:
        print("torch not importable; GPU arms will be skipped")
    if torch is not None and not torch.cuda.is_available():
        print("CUDA not available; GPU arms will run on CPU")

    dev = "cuda" if (torch is not None and torch.cuda.is_available()) else "cpu"
    rng = np.random.default_rng(args.seed)

    # synthetic data: 8 well-separated blobs (10% of rows as a UMAP sample)
    n, d = args.rows, args.d
    blob_locs = rng.normal(size=(8, d)) * 4.0
    labels = rng.integers(0, 8, size=n)
    x = blob_locs[labels] + rng.normal(size=(n, d)) * 0.5

    out = {"device": dev, "rows": n, "d": d, "k": args.k, "rounds": args.rounds, "seed": args.seed}

    # --- kNN -------------------------------------------------------------- #
    timings = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        lab, dsts = emaopt.find_neighbors(x, 15, device=dev)
        timings.append(time.perf_counter() - t0)
    out["knn"] = {"median_s": _median(timings), "k": 15}
    out["knn"]["self_excluded"] = bool(not (lab == np.arange(n)[:, None]).any())
    out["knn"]["nearest_hist_min"] = float(dsts[:, 0].min())

    # --- UMAP -------------------------------------------------------------- #
    sample = x[rng.choice(n, size=min(n, 8000), replace=False)]
    timings = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        res = emaopt.umap_embed(sample, emaopt.UMAPConfig(n_neighbors=15, n_epochs=60, burn_in_epoch=15, seed=args.seed), device=dev)
        timings.append(time.perf_counter() - t0)
    out["umap"] = {"median_s": _median(timings), "n": int(sample.shape[0]), "seed_quality": float(
        np.isfinite(res.embedding).all() and res.embedding.std(axis=0).mean())}

    # --- e-means ------------------------------------------------------------ #
    timings = dict(device=[], host=[])
    objectives = dict(device=[], host=[])
    for r in range(args.rounds):
        t0 = time.perf_counter()
        host = emaopt.emeans(x, args.k, nepochs=args.nepochs, seed=args.seed, unused_threshold=None)
        timings["host"].append(time.perf_counter() - t0)
        objectives["host"].append(float(host.objective))
        t0 = time.perf_counter()
        dev_res = emaopt.emeans_gpu_fit(x, args.k, nepochs=args.nepochs, seed=args.seed, device=dev)
        timings["device"].append(time.perf_counter() - t0)
        objectives["device"].append(float(dev_res.objective))
    out["emeans"] = {
        "device_median_s": _median(timings["device"]),
        "host_median_s": _median(timings["host"]),
        "device_objective": _median(objectives["device"]),
        "host_objective": _median(objectives["host"]),
        "objective_gap_pct": 100.0 * (_median(objectives["device"]) - _median(objectives["host"])) / max(_median(objectives["host"]), 1e-12),
    }

    # --- placement ---------------------------------------------------------- #
    parents = dev_res.centroids[: min(args.k, 1000)]
    timings = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        pids = emaopt.nearest_parent(x, parents, device=dev)
        pos = emaopt.place_around_parents(pids, np.random.default_rng(args.seed).normal(size=(parents.shape[0], 2)), spread=0.3, seed=args.seed)
        timings.append(time.perf_counter() - t0)
    out["placement"] = {"median_s": _median(timings), "n": int(n), "parents": int(parents.shape[0])}

    # --- voronoi + push_apart ------------------------------------------------ #
    m = min(n, 10000)
    vpos = np.random.default_rng(args.seed).normal(size=(m, 2))
    timings = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        vr = emaopt.voronoi_cut(vpos, 100, iterations=60, seed=args.seed, device=dev)
        pa = emaopt.push_apart(vr.centers, min_separation=0.1, iterations=50, seed=args.seed, device=dev)
        timings.append(time.perf_counter() - t0)
    out["voronoi"] = {"median_s": _median(timings), "groups": 100, "points": m,
                      "labels_used": int(len(np.unique(vr.labels))),
                      "converged": bool(vr.converged), "push_apart_min_sep": float(pa.min_separation)}

    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"recorded -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
