#!/usr/bin/env python
"""End-to-end run of ``emaopt.run_basemap`` on the real MONET dataset.

    python scripts/run_basemap_monet.py [--files 4] [--out evidence/...]

Loads the first ``--files`` CC12M parquet files (only the DINOv2 embedding
column), runs the full hierarchical placement (GPU e-means -> UMAP top ->
mid/fine placement -> tail placement -> Voronoi + push apart) and records
timings + metrics as JSON to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import emaopt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2_000)
    ap.add_argument("--mid-k", type=int, default=4_000)
    ap.add_argument("--fine-k", type=int, default=5_000)
    ap.add_argument("--nepochs", type=int, default=3)
    ap.add_argument("--groups", type=int, default=1_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--do-tail", action="store_true", default=True)
    ap.add_argument("--no-tail", dest="do_tail", action="store_false")
    ap.add_argument("--out", type=str, default="", help="write JSON record here")
    args = ap.parse_args()

    files = [hf_hub_download("jasperai/monet", "v1.2.0/cc12m/0-0/%06d.parquet" % i,
                             repo_type="dataset") for i in range(args.files)]
    parts = []
    for p in files:
        df = pd.read_parquet(p, columns=["embedding_dinov2-vitg14"])
        parts.append(np.stack([np.asarray(v, dtype=np.float32)
                               for v in df["embedding_dinov2-vitg14"]]))
    emb = np.concatenate(parts, axis=0)
    n, d = emb.shape
    print(f"data: n={n} d={d}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    res = emaopt.run_basemap(
        emb,
        top_k=args.top_k, mid_k=args.mid_k, fine_k=args.fine_k,
        nepochs=args.nepochs, seed=args.seed, groups=args.groups,
        do_tail=args.do_tail, device="cuda",
        status=lambda m: print(f"  ... {m}", flush=True),
    )
    peak = torch.cuda.max_memory_allocated() / 1e9
    wall = time.perf_counter() - t0
    record = {
        "n": int(n), "d": int(d),
        "top_k": int(args.top_k), "mid_k": int(args.mid_k), "fine_k": int(args.fine_k),
        "groups": int(args.groups),
        "wall_time_s": round(wall, 2),
        "peak_gpu_gb": round(float(peak), 2),
        "top_xy_shape": list(res.top_xy.shape),
        "mid_xy_shape": None if res.mid_xy is None else list(res.mid_xy.shape),
        "fine_xy_shape": list(res.fine_xy.shape),
        "tail_xy_shape": list(res.tail_xy.shape),
        "voronoi_groups": int(res.voronoi.extras["groups"]) if res.voronoi else None,
        "voronoi_labels_used": int(len(np.unique(res.voronoi.labels))) if res.voronoi else None,
        "voronoi_converged": bool(res.voronoi.converged) if res.voronoi else None,
        "push_apart_converged": bool(res.push_apart.converged) if res.push_apart else None,
        "push_apart_min_separation": float(res.push_apart.min_separation) if res.push_apart else None,
        "spacing_median": float(res.spacing),
        "extras": res.extras,
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(record, f, indent=2)
        print(f"recorded -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
