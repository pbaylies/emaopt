"""Acceptance tests for the Tier-3 pieces: subspace tracking and robust PCA.

Every threshold below is the measured margin of a *comparison*, not a tuned
number: the tracker must beat the frozen first-block basis on a rotating
subspace, the robust tracker must beat plain covariance PCA at the same
contamination level, and the PCP solution must beat its own undebiased version.
"""

from __future__ import annotations

import numpy as np
import pytest

from emaopt.algorithms.pca import PCACovariance
from emaopt.algorithms.robust_pca import (
    StreamingRobustPCA,
    debias_low_rank,
    robust_pca,
    robust_scale,
)
from emaopt.algorithms.subspace import SubspaceTracker, top_basis
from emaopt.benchmarks import metrics
from emaopt.state import EMA


def _rotating_stream(rate: float = 0.10, blocks: int = 12, per: int = 150,
                     d: int = 10, k: int = 2, seed: int = 200):
    """Rank-k signal whose subspace rotates by ``rate`` radians per block.

    The signal lives in columns 0 and 2 of a rotation that acts on the (0, 1)
    and (2, 3) planes, so the *span* genuinely moves: rotating inside an
    occupied plane (e.g. columns 0 and 1) leaves the subspace unchanged and
    would make a static basis look perfect.
    """
    rotations = []
    for b in range(blocks):
        theta = rate * b
        r = np.eye(d)
        for i, j in ((0, 1), (2, 3)):
            r[i, i] = np.cos(theta)
            r[i, j] = -np.sin(theta)
            r[j, i] = np.sin(theta)
            r[j, j] = np.cos(theta)
        rotations.append(r)
    block_data = [
        np.random.default_rng(seed + b).normal(size=(per, k)) @ rotations[b][:, [0, 2]].T
        for b in range(blocks)
    ]
    x = np.vstack(block_data)
    truth = rotations[-1][:, [0, 2]]
    first = np.vstack(block_data[:2])
    static = top_basis(first.T @ first / first.shape[0], k, iters=40)
    return x, truth, static


def _spiked(frac: float, mag: float, n: int = 400, d: int = 10, rank: int = 2,
            seed: int = 7):
    """Rank-``rank`` signal plus ``frac`` element-wise spikes of magnitude ``mag``."""
    latent = np.random.default_rng(seed).normal(size=(n, rank)) @ np.random.default_rng(
        seed + 1).normal(size=(rank, d))
    truth = np.linalg.svd(latent, full_matrices=False)[2][:rank].T
    spikes = np.zeros_like(latent)
    mask = np.random.default_rng(seed + 2).random(spikes.shape) < frac
    spikes[mask] = np.random.default_rng(seed + 3).choice(
        [-1.0, 1.0], size=int(mask.sum())) * mag
    return latent + spikes, truth


def _pcp_problem(rank: int, mag: float, frac: float, n: int = 300, d: int = 20,
                 seed: int = 3):
    rng = np.random.default_rng(seed)
    low = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d))
    spikes = np.zeros((n, d))
    mask = rng.random(spikes.shape) < frac
    spikes[mask] = rng.choice([-1.0, 1.0], size=int(mask.sum())) * mag
    return low + spikes, low, mask


# -- subspace tracking ------------------------------------------------------ #
def test_tracker_follows_a_rotating_subspace():
    x, truth, static = _rotating_stream()
    model = SubspaceTracker(2, updater=EMA(alpha=0.2), power_iters=4, seed=0)
    result = model.fit(x, batch_size=150, nepochs=1, seed=0)
    tracked = metrics.subspace_alignment(result.params["components"], truth)
    frozen = metrics.subspace_alignment(static, truth)
    assert frozen < 0.4, "the static basis has to be genuinely wrong for this test to mean anything"
    assert tracked > 0.65
    assert tracked > frozen + 0.3


def test_tracker_captures_the_signal_energy():
    x, truth, _ = _rotating_stream()
    model = SubspaceTracker(2, updater=EMA(alpha=0.2), power_iters=4, seed=0)
    result = model.fit(x, batch_size=150, nepochs=1, seed=0)
    # crude stationary baseline: energy captured by the frozen basis on the last block
    last = x[-150:]
    frozen_capture = metrics.explanation_ratio(last, _rotating_stream()[2])
    tracked_capture = metrics.explanation_ratio(last, result.params["components"])
    assert tracked_capture > frozen_capture
    assert tracked_capture > 0.6


# -- streaming robust PCA --------------------------------------------------- #
@pytest.mark.parametrize("frac,mag,plain_ceiling", [(0.15, 10.0, 0.90), (0.25, 20.0, 0.65)])
def test_streaming_robust_beats_plain_pca_on_gross_spikes(frac, mag, plain_ceiling):
    x, truth = _spiked(frac, mag)
    plain = PCACovariance(2, seed=0).fit(x, batch_size=None, nepochs=1, seed=0)
    robust = StreamingRobustPCA(2, updater=EMA(alpha=0.3), seed=0).fit(
        x, batch_size=128, nepochs=10, seed=0)
    plain_align = metrics.subspace_alignment(plain.params["components"], truth)
    robust_align = metrics.subspace_alignment(robust.params["components"], truth)
    assert plain_align < plain_ceiling
    assert robust_align > 0.95
    assert robust_align > plain_align + 0.10


def test_streaming_threshold_is_fixed_by_the_data_scale():
    """Regression: deriving the threshold from residuals makes it decay to zero.

    Residual-based thresholds collapse as the basis converges (residuals vanish),
    which trimmed everything and drove alignment from 0.998 to 0.647 over ten
    epochs.  The threshold must stay pinned to the data scale.
    """
    x, _ = _spiked(0.15, 10.0)
    model = StreamingRobustPCA(2, updater=EMA(alpha=0.3), seed=0)
    result = model.fit(x, batch_size=128, nepochs=10, seed=0)
    assert result.params["threshold"] == pytest.approx(
        3.0 * robust_scale(x.ravel()), rel=1e-9)
    assert result.params["trimmed_fraction"] == pytest.approx(0.15, abs=0.08)


def test_robust_scale_survives_a_quarter_of_catastrophic_values():
    """MAD degrades slowly where std collapses: 1.6x versus 22x at 25% spikes."""
    rng = np.random.default_rng(0)
    clean = rng.normal(size=20000)
    dirty = clean.copy()
    dirty[:5000] = 50.0
    assert dirty.std() > 20 * clean.std()
    assert robust_scale(dirty) < 2.0 * robust_scale(clean)


# -- batch robust PCA ------------------------------------------------------- #
def test_pcp_recovers_rank_one_from_ten_percent_gross_spikes():
    x, low, mask = _pcp_problem(1, 20.0, 0.10)
    out = robust_pca(x)
    relative = np.linalg.norm(out["low_rank"] - low) / np.linalg.norm(low)
    assert out["rank"] == 1
    assert out["converged"]
    assert relative < 0.08
    support = np.abs(out["sparse"]) > 0.5
    f1 = 2 * int((mask & support).sum()) / int(mask.sum() + support.sum())
    assert f1 > 0.9


def test_pcp_recovers_rank_two_after_penalty_balancing():
    """Regression: geometric penalty growth pinned mu at its cap.

    The old schedule stopped after 33 iterations at rank 6 with relative error
    0.1417 and convex objective 921.04; a residual-balanced ADMM reaches 919.74,
    which is now what this solver returns (rank 2, 0.0697).
    """
    x, low, _ = _pcp_problem(2, 20.0, 0.10)
    out = robust_pca(x)
    relative = np.linalg.norm(out["low_rank"] - low) / np.linalg.norm(low)
    assert out["rank"] == 2
    assert relative < 0.09
    assert out["converged"]


def test_debiasing_removes_nuclear_norm_shrinkage():
    x, low, _ = _pcp_problem(1, 20.0, 0.10)
    plain = robust_pca(x)
    refined = robust_pca(x, refine_rank=1)

    def relative(matrix: np.ndarray) -> float:
        return float(np.linalg.norm(matrix - low) / np.linalg.norm(low))

    assert relative(refined["refined_low_rank"]) < 0.01
    assert relative(refined["refined_low_rank"]) < relative(plain["low_rank"])


def test_debiasing_is_exact_given_the_true_support():
    x, low, _ = _pcp_problem(2, 25.0, 0.10)
    recovered = debias_low_rank(x, x - low, 2)
    assert np.linalg.norm(recovered - low) / np.linalg.norm(low) < 1e-6


def test_sparse_component_flags_only_the_corrupted_entries():
    x, low, mask = _pcp_problem(2, 25.0, 0.10)
    model = StreamingRobustPCA(2, updater=EMA(alpha=0.3), seed=0)
    result = model.fit(x, batch_size=64, nepochs=10, seed=0)
    flagged = model.sparse_component(x, result.params) != 0.0
    # the recovered subspace explains the signal, so what it cannot explain is
    # dominated by the spikes
    hit_rate = float(flagged[mask].mean())
    false_rate = float(flagged[~mask].mean())
    assert hit_rate > 0.9
    assert false_rate < 0.05
