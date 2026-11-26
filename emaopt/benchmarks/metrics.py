"""Metrics: the numbers that become evidence.

The primary research metric is quality per unit compute, which is why every
curve-producing runner records ``examples_processed`` alongside the objective.
The helpers here define *quality* unambiguously for each model family.
"""

from __future__ import annotations

import numpy as np

from ..core import squared_distances

_EPS = 1e-12


def mse_to_centroids(x: np.ndarray, centroids: np.ndarray) -> float:
    """k-means MSE: mean squared distance to the nearest centroid."""
    x = np.asarray(x, dtype=np.float64)
    dist = squared_distances(x, np.asarray(centroids, dtype=np.float64))
    return float(dist.min(axis=1).mean())


def reconstruction_error(x: np.ndarray, recon: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    return float(np.mean((x - recon) ** 2))


def relative_error(x: np.ndarray, recon: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return reconstruction_error(x, recon) / max(float(np.mean(x**2)), _EPS)


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """Hubert-Arabie adjusted Rand index (1 = identical partition, 0 = chance)."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError("label vectors must have the same length")
    n = a.size
    if n < 2:
        return 0.0
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    table = np.zeros((ai.max() + 1, bi.max() + 1), dtype=np.int64)
    np.add.at(table, (ai, bi), 1)
    comb = lambda x: x * (x - 1) / 2.0  # noqa: E731
    sum_ij = comb(table.astype(float)).sum()
    sum_a = comb(table.sum(axis=1).astype(float)).sum()
    sum_b = comb(table.sum(axis=0).astype(float)).sum()
    total = comb(float(n))
    expected = sum_a * sum_b / max(total, _EPS)
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if abs(denom) <= _EPS:
        return 1.0 if abs(sum_ij - expected) <= _EPS else 0.0
    return float((sum_ij - expected) / denom)


def cluster_purity(true_labels: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of points in the majority true class of their predicted cluster."""
    true_labels = np.asarray(true_labels)
    pred = np.asarray(pred)
    total = 0
    for cluster in np.unique(pred):
        mask = pred == cluster
        _, counts = np.unique(true_labels[mask], return_counts=True)
        total += int(counts.max())
    return float(total / max(true_labels.size, 1))


def subspace_alignment(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Mean squared cosine of principal angles between two column spaces (1 = equal)."""
    a = np.asarray(basis_a, dtype=np.float64)
    b = np.asarray(basis_b, dtype=np.float64)
    if a.shape[1] == 0 or b.shape[1] == 0:
        return 0.0
    qa, _ = np.linalg.qr(a)
    qb, _ = np.linalg.qr(b)
    qa = qa[:, : a.shape[1]]
    qb = qb[:, : b.shape[1]]
    s = np.linalg.svd(qa.T @ qb, compute_uv=False)
    k = min(a.shape[1], b.shape[1])
    return float(np.mean(s[:k] ** 2))


def max_abs_correlation(sources: np.ndarray, recovered: np.ndarray) -> np.ndarray:
    """(d_recovered, d_true) absolute Pearson correlation matrix."""
    s = np.asarray(sources, dtype=np.float64)
    r = np.asarray(recovered, dtype=np.float64)
    s = s - s.mean(axis=0)
    r = r - r.mean(axis=0)
    denom = np.outer(np.linalg.norm(r, axis=0), np.linalg.norm(s, axis=0))
    return np.abs((r.T @ s) / np.maximum(denom, _EPS))


def best_source_correlation(sources: np.ndarray, recovered: np.ndarray) -> float:
    """Greedy best match: mean over recovered components of their best |corr|."""
    corr = max_abs_correlation(sources, recovered)
    return float(np.mean(corr.max(axis=1)))


def explanation_ratio(x: np.ndarray, components: np.ndarray) -> float:
    """Fraction of squared energy captured by the given orthonormal components."""
    x = np.asarray(x, dtype=np.float64)
    comps = np.asarray(components, dtype=np.float64)
    total = float(np.mean(np.sum(x**2, axis=1)))
    if total <= _EPS:
        return 1.0
    captured = float(np.mean(np.sum((x @ comps) ** 2, axis=1)))
    return captured / total


def perplexity(per_word_nll: float) -> float:
    return float(np.exp(per_word_nll))


def quantization_bits(n_codewords: int) -> float:
    return float(np.ceil(np.log2(max(n_codewords, 1))))


def dead_fraction(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    return float(np.mean(counts <= 1e-12))


def imbalance(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    mean = counts.mean()
    return float(counts.max() / max(mean, _EPS))


def summary_table(rows: list[dict]) -> str:
    """Fixed-width table for terminal evidence."""
    if not rows:
        return ""
    keys = list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(f"{r.get(k, ''):.6g}" if isinstance(
        r.get(k), float) else str(r.get(k, ""))) for r in rows)) for k in keys}
    header = "  ".join(str(k).ljust(widths[k]) for k in keys)
    lines = [header, "  ".join("-" * widths[k] for k in keys)]
    for row in rows:
        cells = []
        for k in keys:
            v = row.get(k, "")
            cells.append((f"{v:.6g}" if isinstance(v, float) else str(v)).ljust(widths[k]))
        lines.append("  ".join(cells))
    return "\n".join(lines)


def projection_pair_scores(x_fit, x_eval, *, width: int = 64, chunk: int = 32_768):
    """Untrained pairwise-proximity detector: cosine of the RAW (variance-
    ordered, un-whitened) top-``width`` PCA projections of the data.

    T13's attribution study (REPORT §12.2) showed this beats every trained
    mixture score on the CLIP-pair AUROC task (0.90-0.92 vs <=0.76) at every
    width tested: the pairwise signal is feature geometry, and whitening
    discards exactly the part (the variance ordering) that carries it.  The
    plan (mean + directions) is fitted on ``x_fit`` in streaming chunks via
    GramianStat and only ``x_eval`` is scored; rows of ``x_eval`` are assumed
    already L2-normalised, matching the shipped embeddings.

    Returns the (m, m) cosine-score matrix (diagonal 1 by construction).
    """
    from ..state.sufficient import GramianStat

    x_fit = np.asarray(x_fit, dtype=np.float64)
    stat = GramianStat(n_components=None, center=True)
    for start in range(0, x_fit.shape[0], chunk):
        stat.update(x_fit[start : start + chunk])
    plan = stat.decode()
    comp = np.asarray(plan["components"], dtype=np.float64)[:, :width]
    mean = np.asarray(plan["mean"], dtype=np.float64)
    z = (np.asarray(x_eval, dtype=np.float64) - mean) @ comp
    norms = np.maximum(np.linalg.norm(z, axis=1, keepdims=True), _EPS)
    zn = z / norms
    return zn @ zn.T
