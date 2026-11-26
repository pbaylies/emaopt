#!/usr/bin/env python
"""What one covariance preparation costs per family, and how to trust it.

    python scripts/bench_covariance_prep.py
    python scripts/bench_covariance_prep.py --dim 512 --comps 64 --rows 8192
    python scripts/bench_covariance_prep.py --families tied,diag --repeats 5

Methodology, because this host is shared and its BLAS is unstable under load:

* **Interleaved rounds.**  Every candidate runs once per round, round-robin, and
  the reported figure is the median over rounds.  Slow drift -- another user's
  job starting, a clock change -- lands on all candidates alike instead of on
  whichever one happened to run in the bad window.
* **CPU time next to wall time.**  ``time.process_time`` counts the CPU this
  process actually burned, so it is immune to being descheduled.  When a family's
  wall time is inflated but its CPU time is not, the number to trust is the CPU
  one; when CPU time is inflated too, the work really did change.
* **A fixed BLAS thread count.**  ``OPENBLAS_NUM_THREADS`` is pinned (default 8,
  ``--threads`` to change) *before* numpy is imported, so the pool does not
  oversubscribe the machine differently on every run.

The point is comparison, not absolutes: an absolute millisecond on a box with a
load average in the teens is not a reproducible number, and the earlier version of
this script silently reported exactly those.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Pinned before numpy loads: OpenBLAS reads this once, at import.
_DEFAULT_THREADS = "8"
if not os.environ.get("OPENBLAS_NUM_THREADS"):
    os.environ["OPENBLAS_NUM_THREADS"] = _DEFAULT_THREADS
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"]

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emaopt import GaussianMixture  # noqa: E402
from emaopt.core import FitConfig  # noqa: E402


def interleaved(candidates: list[tuple[str, object]], rounds: int):
    """Time every candidate once per round; return median wall and CPU ms."""

    wall: dict[str, list[float]] = {name: [] for name, _ in candidates}
    cpu: dict[str, list[float]] = {name: [] for name, _ in candidates}
    for name, fn in candidates:            # warm every path once
        fn()
    for _ in range(rounds):
        for name, fn in candidates:
            started_wall, started_cpu = time.perf_counter(), time.process_time()
            fn()
            wall[name].append((time.perf_counter() - started_wall) * 1e3)
            cpu[name].append((time.process_time() - started_cpu) * 1e3)
    return ({k: float(np.median(v)) for k, v in wall.items()},
            {k: float(np.median(v)) for k, v in cpu.items()})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--comps", type=int, default=64)
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--probe", type=int, default=1024)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--families", default="tied,diag,full,lowrank",
                    help="comma-separated; the full family is slow on a host "
                         "whose LAPACK getrf is degraded")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    d = args.dim
    mix = rng.standard_normal((d, d)) / np.sqrt(d)
    x = rng.standard_normal((args.rows, d)) @ (mix + np.eye(d))
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan")
    print(f"rows={args.rows} dim={d} k={args.comps} probe={args.probe} "
          f"opens={os.environ['OPENBLAS_NUM_THREADS']} "
          f"cpus={os.cpu_count()} load1={load:.1f}")
    print(f"median of {args.repeats} interleaved rounds, ms")

    for family in [f.strip() for f in args.families.split(",") if f.strip()]:
        model = GaussianMixture(args.comps, covariance=family, seed=0)
        cfg = FitConfig(nepochs=1, batch_size=args.rows, seed=0,
                        shuffle=False, drop_last=False, tol=None)
        try:
            params = model.decode(model.initialize(x, cfg))
            probe = x[: args.probe]
        except Exception as exc:            # pragma: no cover - family quirks
            print(f"{family:10s} skipped: {type(exc).__name__}: {exc}")
            continue
        wall, cpu = interleaved([
            ("log_prob", lambda: model.component_log_prob(probe, params)),
            ("stats", lambda: model.statistics_from(probe, params)),
            ("decode", lambda: model.decode(model.initialize(x[:256], cfg))
             if False else model.decode(
                 model.statistics_from(probe, params))),
        ], args.repeats)
        print(f"{family:10s} log_prob  wall {wall['log_prob']:9.1f}  "
              f"cpu {cpu['log_prob']:9.1f}")
        print(f"{'':10s} estimate  wall {wall['stats']:9.1f}  "
              f"cpu {cpu['stats']:9.1f}")
        print(f"{'':10s} decode    wall {wall['decode']:9.1f}  "
              f"cpu {cpu['decode']:9.1f}")

    a = rng.standard_normal((d, d))
    a = a @ a.T + d * np.eye(d)
    chol = np.linalg.cholesky(a)
    eye = np.eye(d)
    candidates = [
        ("inv(dense)", lambda: np.linalg.inv(a)),
        ("inv(chol)", lambda: np.linalg.inv(chol)),
        ("solve(dense,eye)", lambda: np.linalg.solve(a, eye)),
        ("solve(chol,eye)", lambda: np.linalg.solve(chol, eye)),
        ("cholesky", lambda: np.linalg.cholesky(a)),
        ("slogdet", lambda: np.linalg.slogdet(a)),
        ("eigh", lambda: np.linalg.eigh(a)),
        ("matmul", lambda: a @ a),
    ]
    wall, cpu = interleaved(candidates, args.repeats)
    print("\nprimitives on this host:")
    for name, _ in candidates:
        print(f"  {name:16s} wall {wall[name]:9.1f}  cpu {cpu[name]:9.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
