"""Shared test helpers: make the package importable and provide small oracles."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def greedy_match_error(true_points: np.ndarray, estimated_points: np.ndarray) -> float:
    """Mean distance after greedily matching each true point to a distinct estimate."""
    a = np.asarray(true_points, dtype=float)
    b = np.asarray(estimated_points, dtype=float)
    if a.shape[-1] != b.shape[-1]:
        raise ValueError("points must share a dimension")
    dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    used: set[int] = set()
    errors = []
    for i in range(dist.shape[0]):
        for j in np.argsort(dist[i]):
            if int(j) not in used:
                used.add(int(j))
                errors.append(float(dist[i, j]))
                break
    return float(np.mean(errors))


def brute_force_sq_dist(x: np.ndarray, c: np.ndarray) -> np.ndarray:
    return ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2)


@pytest.fixture(scope="session")
def small_blobs():
    from emaopt.benchmarks import datasets

    return datasets.blobs(n=400, k=3, d=2, sep=7.0, seed=3)
