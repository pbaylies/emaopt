"""Synthetic datasets with known answers.

Every generator returns a :class:`Dataset` whose ``meta`` carries the ground
truth (centers, dictionary, topics, mixing matrix, ...) so that a test can check
recovery rather than only that an objective went down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Dataset:
    x: np.ndarray
    name: str
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        if self.x.ndim != 2:
            raise ValueError("dataset x must be 2-D")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def d(self) -> int:
        return int(self.x.shape[1])

    def split(self, frac: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Random train/test split of the *rows*."""
        rng = np.random.default_rng(seed)
        n = self.n
        order = rng.permutation(n)
        n_test = max(1, int(round(frac * n)))
        test = np.sort(order[:n_test])
        train = np.sort(order[n_test:])
        return self.x[train], self.x[test]

    def subsample(self, m: int, seed: int = 0) -> np.ndarray:
        if m >= self.n:
            return self.x
        rng = np.random.default_rng(seed)
        return self.x[rng.choice(self.n, size=int(m), replace=False)]


def blobs(
    n: int = 2000,
    k: int = 4,
    d: int = 2,
    sep: float = 6.0,
    seed: int = 0,
    weights: Sequence[float] | None = None,
) -> Dataset:
    """Isotropic Gaussian blobs around well-separated centers.

    ``weights`` makes the cluster masses unequal -- the regime where Lloyd's
    always-decreasing objective is most likely to get stuck (a big cluster left
    short of centroids while a small one takes two).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=sep, size=(k, d))
    if weights is None:
        labels = rng.integers(0, k, size=n)
    else:
        p = np.asarray(weights, dtype=np.float64)
        if p.shape != (k,) or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
            raise ValueError("weights must be k non-negative values summing to 1")
        labels = rng.choice(k, size=n, p=p)
    x = centers[labels] + rng.normal(size=(n, d))
    return Dataset(x, "blobs", {"labels": labels, "centers": centers, "k": k})


def gaussian_mixture(
    n: int = 3000,
    k: int = 3,
    d: int = 4,
    *,
    covariance: str = "diag",
    separation: float = 5.0,
    seed: int = 0,
) -> Dataset:
    """Mixture whose true means/weights are known; covariance shapes vary."""
    rng = np.random.default_rng(seed)
    means = rng.normal(scale=separation, size=(k, d))
    weights = np.full(k, 1.0 / k)
    labels = rng.choice(k, size=n, p=weights)
    x = np.empty((n, d))
    covs = []
    for j in range(k):
        if covariance == "spherical":
            cov = np.eye(d) * (0.5 + 0.1 * j)
        elif covariance == "diag":
            cov = np.diag(0.3 + rng.random(d) * 1.5)
        elif covariance == "tied":
            base = np.diag(0.5 + rng.random(d))
            cov = base
        elif covariance == "full":
            a = rng.normal(size=(d, d)) * 0.4
            cov = a @ a.T + 0.2 * np.eye(d)
        else:
            raise ValueError(f"unknown covariance {covariance!r}")
        covs.append(cov)
        idx = labels == j
        x[idx] = rng.multivariate_normal(means[j], cov, size=int(idx.sum()))
    return Dataset(x, "gaussian_mixture", {
        "labels": labels,
        "means": means,
        "weights": weights,
        "covariances": np.array(covs),
        "covariance": covariance,
    })


def concentric_rings(n: int = 1200, noise: float = 0.12, seed: int = 0) -> Dataset:
    """Two concentric circles: invisible to k-means, visible to spectral methods."""
    rng = np.random.default_rng(seed)
    half = n // 2
    r1 = 1.0 + rng.normal(scale=noise, size=half)
    r2 = 3.0 + rng.normal(scale=noise, size=n - half)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    radius = np.concatenate([r1, r2])
    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    labels = np.concatenate([np.zeros(half, dtype=int), np.ones(n - half, dtype=int)])
    return Dataset(xy, "concentric_rings", {"labels": labels, "k": 2})


def low_rank(n: int = 1000, d: int = 12, rank: int = 3, noise: float = 0.01,
             seed: int = 0) -> Dataset:
    """Exactly rank-``r`` matrix plus small noise."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(n, rank))
    v = rng.normal(size=(d, rank))
    x = u @ v.T
    x += noise * rng.normal(size=x.shape)
    return Dataset(x, "low_rank", {"u": u, "v": v, "rank": rank})


def nonneg_low_rank(n: int = 800, d: int = 10, rank: int = 3, noise: float = 0.0,
                    seed: int = 0) -> Dataset:
    """Non-negative rank-``r`` matrix (NMF ground truth)."""
    rng = np.random.default_rng(seed)
    u = rng.random((n, rank))
    v = rng.random((d, rank))
    x = u @ v.T
    if noise:
        x = np.maximum(x + noise * rng.random(x.shape), 0.0)
    return Dataset(x, "nonneg_low_rank", {"u": u, "v": v, "rank": rank})


def sparse_codes(
    n: int = 1500,
    d: int = 24,
    m: int = 12,
    sparsity: int = 2,
    noise: float = 0.01,
    seed: int = 0,
) -> Dataset:
    """``x = D z + noise`` with at most ``sparsity`` non-zero codes per row."""
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(d, m))
    atoms /= np.maximum(np.linalg.norm(atoms, axis=0, keepdims=True), 1e-12)
    z = np.zeros((n, m))
    for i in range(n):
        support = rng.choice(m, size=min(sparsity, m), replace=False)
        z[i, support] = rng.normal(scale=1.0, size=support.size)
    x = z @ atoms.T + noise * rng.normal(size=(n, d))
    return Dataset(x, "sparse_codes", {"dictionary": atoms, "codes": z, "atoms": m})


def sparse_decomposition(
    n: int = 1200,
    d: int = 16,
    k: int = 8,
    sparsity: int = 2,
    noise: float = 0.15,
    seed: int = 0,
) -> Dataset:
    """Sparse diag+tied ground truth: sparse non-negative codes times atoms."""
    rng = np.random.default_rng(seed)
    atoms = rng.normal(scale=1.5, size=(k, d))
    codes = np.zeros((n, k))
    for i in range(n):
        support = rng.choice(k, size=min(sparsity, k), replace=False)
        codes[i, support] = rng.random(support.size) + 0.5
    x = codes @ atoms + noise * rng.normal(size=(n, d))
    return Dataset(x, "sparse_decomposition", {"atoms": atoms, "codes": codes, "k": k})


def topics(
    n_docs: int = 400,
    vocab: int = 60,
    n_topics: int = 3,
    words_per_topic: int = 20,
    doc_len: int = 40,
    seed: int = 0,
) -> Dataset:
    """Documents drawn from topics with disjoint word blocks (easy recovery check)."""
    rng = np.random.default_rng(seed)
    beta = np.full((n_topics, vocab), 1e-4)
    for t in range(n_topics):
        block = slice(t * words_per_topic, (t + 1) * words_per_topic)
        beta[t, block] = rng.random(words_per_topic) + 0.5
    beta /= beta.sum(axis=1, keepdims=True)
    doc_topic = rng.dirichlet(np.full(n_topics, 0.3), size=n_docs)
    counts = np.zeros((n_docs, vocab))
    for i in range(n_docs):
        words = rng.choice(vocab, size=doc_len, p=doc_topic[i] @ beta)
        counts[i] = np.bincount(words, minlength=vocab)
    return Dataset(counts, "topics", {"beta": beta, "doc_topic": doc_topic,
                                      "n_topics": n_topics, "words_per_topic": words_per_topic})


def laplace_sources(n: int = 4000, d: int = 3, seed: int = 0) -> Dataset:
    """Super-Gaussian sources (ICA ground truth)."""
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(n, d))
    return Dataset(sources, "laplace_sources", {"sources": sources})


def linear_mix(dataset: Dataset, seed: int = 0, scale: float = 1.0) -> Dataset:
    """Apply a random invertible linear mixing to a dataset's rows."""
    rng = np.random.default_rng(seed)
    d = dataset.d
    mixing = rng.normal(size=(d, d))
    while abs(np.linalg.det(mixing)) < 1e-3:
        mixing = rng.normal(size=(d, d))
    x = dataset.x @ mixing.T * scale
    meta = dict(dataset.meta)
    meta["mixing"] = mixing
    return Dataset(x, f"{dataset.name}_mixed", meta)


DATASETS: dict[str, Callable[..., Dataset]] = {
    "blobs": blobs,
    "gaussian_mixture": gaussian_mixture,
    "concentric_rings": concentric_rings,
    "low_rank": low_rank,
    "nonneg_low_rank": nonneg_low_rank,
    "sparse_codes": sparse_codes,
    "sparse_decomposition": sparse_decomposition,
    "topics": topics,
    "laplace_sources": laplace_sources,
}


def make_dataset(name: str, **kwargs: Any) -> Dataset:
    """Build a dataset by name."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)
