"""s-means: the scale- and distortion-aware few-epoch recipe, as options.

`s-means <https://github.com/suayptalha/s-means>`_ (Kocabay, Yuksel & Yalcin,
2026, MIT) is a few-epoch k-means that changes four cheap things about the
e-means recipe it is compared against:

1. **Subsampled greedy D^1.5 seeding** instead of greedy k-means++ on the
   full data: draw a 16-rows-per-centre coreset (1,024-row floor) and keep the
   candidate whose insertion most lowers the D^1.5 potential.  The old batched
   four-round form is retained as ``greedy=False``; it collapsed at high k
   because several centres were drawn from one weight vector.
2. **Distortion-weighted anisotropic dead-cluster revival**: every split is
   aimed along a scale-invariant feature-RMS direction with a random sign
   pattern, and donors are chosen by distortion and usage rather than by mass
   alone.  Above k=64 SMeans uses the isotropic split: the anisotropic form
   measured 48.0 MSE against 9.6 for isotropic at k=128 on the same seed.
3. **Empirical-Bayes shrinkage** of the final centroid toward the smoothed EMA
   centroid, so a low-count cluster is not handed wholesale to the noisy
   final-pass mean.
4. **Power-cosine step decay** (``0.06 + 0.94 * cos^0.92``) instead of the
   plain cosine.

This module is a NumPy re-derivation of those published mechanisms for
emaopt's measurement -> update -> decode loop.  No upstream code is copied: the
reference implementation is PyTorch/Triton, and its GPU kernels and CUDA-graph
capture have no analogue here.  Four deliberate deviations, each measured
where it matters:

* s-means tracks a per-cluster ``sse_ema``; this port uses the batch's own
  per-cluster SSE in the donor weight (``sse * usage^0.35``), because an EMA of
  it would widen the state beyond the sufficient statistics the reference
  e-means ledger pins.  Under the compiled epoch runner the split happens in
  the Python fallback with no batch SSE, so donor weights fall back to usage.
* ``sigma_hat_feat`` (the feature-wise RMS distortion) is estimated once at
  initialization from a <=4096-row probe instead of online.  That keeps it
  available on the compiled path and costs one pass.
* The batch/step laws stay the e-means ones: they are the control this arm is
  measured against.
* Anisotropic revival is replaced by the isotropic split above k=64; the
  published direction is a small-cluster mechanism, not a wide-codebook one.

Everything is opt-in through :class:`SMeans`; the reference ``EMeans`` defaults
and the parity ledger are unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import UpdateContext, squared_distances
from ..state import EMA

#: Lloyd/coreset-init floor: ``m = max(4k, 512)`` rows, capped at the dataset.
#: The D^1.5 seed has its own floor (16 rows/centre, 1,024 rows) below.
CORESET_MIN = 512

#: D^1.5 seeding needs enough rows per centre to see the geometry.  The old
#: ``max(4k, 512)`` rule had only 4 rows/centre at k=128 and measured 48.9 MSE
#: against 10.0 for full k-means++; 16 rows/centre with a 1,024-row floor and 12
#: greedy candidates measured 9.6 MSE at k=128 and 32.0 at k=32.
D15_CORESET_PER_CLUSTER = 16
D15_CORESET_FLOOR = 1024
D15_TRIALS = 12

#: Anisotropic revival is a small-cluster recipe.  Above this k its measured
#: split direction is worse than the isotropic split (k=128: 48.0 against 9.6
#: MSE on the same seed), while k <= 64 is neutral-to-better.  SMeans therefore
#: asks for anisotropic and falls back to isotropic above the boundary.
ANISOTROPIC_REVIVAL_MAX_K = 64

#: Rounds the published seeding builds its centroids in (batched, not greedy).
SEED_ROUNDS = 4

#: Weight exponent on usage for donor selection (s-means: ``usage^0.35``).
USAGE_EXPONENT = 0.35


def _d15_weights(squared: np.ndarray) -> np.ndarray:
    """Sampling weights proportional to the *distance* to the 1.5th power.

    ``d_min^1.5`` on distances is ``(d_min^2)^0.75`` on the squared distances
    the library already computes, so the exponent is folded rather than taking a
    square root of every entry first.
    """
    sq = np.maximum(np.asarray(squared, dtype=np.float64), 0.0)
    return np.power(sq, 0.75)


def d15_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    coreset: int | None = None,
    rounds: int = SEED_ROUNDS,
    greedy: bool = True,
    trials: int | None = None,
) -> np.ndarray:
    """Initial centroids from subsampled D^1.5 sampling.

    The first centre is a coreset row.  The default greedy form draws
    ``trials`` candidates per centre from ``d_min^1.5`` and keeps the one whose
    insertion lowers the coreset potential most.  That is the D^1.5 weighting
    with the same candidate test k-means++ uses; the previous batched form drew
    several centres from one weight vector, which made their draws correlated
    and collapsed at high ``k`` (20k x 32 k=32: 84.1 MSE against 32.0 for full
    k-means++, while greedy D^1.5 reached 32.0 in 1.7 ms).

    ``greedy=False`` keeps the old multi-round batched path for comparison.  Its
    ``rounds`` distance refreshes are the reason it is cheap, but the draws are
    not independent; it is retained, not recommended.
    """
    data = np.asarray(data, dtype=np.float64)
    k = int(k)
    m = int(coreset or 0)
    m = min(data.shape[0], max(m, D15_CORESET_PER_CLUSTER * k, D15_CORESET_FLOOR))
    sample = data
    if m < data.shape[0]:
        sample = data[rng.choice(data.shape[0], size=m, replace=False)]
    rows = sample.shape[0]
    centres = np.empty((k, sample.shape[1]), dtype=np.float64)
    centres[0] = sample[rng.integers(rows)]
    closest = squared_distances(sample, centres[:1]).ravel()

    if not greedy:
        order = np.arange(1, k)
        rounds = max(1, int(rounds))
        per_round = int(math.ceil(order.size / rounds)) if order.size else 0
        cursor = 0
        while cursor < order.size:
            take = order[cursor : cursor + per_round]
            weights = _d15_weights(closest)
            total = float(weights.sum())
            live = int(np.count_nonzero(weights))
            if not np.isfinite(total) or total <= 0.0:
                pick = rng.integers(rows, size=take.size)
            elif live >= take.size:
                pick = rng.choice(rows, size=take.size, replace=False, p=weights / total)
            else:
                # Degenerate (mostly duplicate) data: fill the rest at random, the
                # same thing greedy k-means++ does when its potential collapses.
                pick = np.concatenate(
                    [
                        rng.choice(rows, size=live, replace=False, p=weights / total)
                        if live
                        else np.empty(0, dtype=np.int64),
                        rng.integers(rows, size=take.size - live),
                    ]
                )
            centres[take] = sample[np.asarray(pick, dtype=np.int64)]
            refreshed = squared_distances(sample, centres[take]).min(axis=1)
            closest = np.minimum(closest, refreshed)
            cursor += take.size
        return centres

    trials = (min(rows, max(D15_TRIALS, 2 + int(np.log(max(k, 2)))))
              if trials is None else max(1, min(int(trials), rows)))
    for j in range(1, k):
        weights = _d15_weights(closest)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            centres[j] = sample[rng.integers(rows)]
            continue
        candidates = np.unique(
            rng.choice(rows, size=min(trials, rows), p=weights / total)
        )
        d2 = squared_distances(sample, sample[candidates])
        potentials = np.minimum(closest[:, None], d2).sum(axis=0)
        best = int(np.argmin(potentials))
        centres[j] = sample[candidates[best]]
        closest = np.minimum(closest, d2[:, best])
    return centres


def feature_scale_spread(
    data: np.ndarray, *, sample: int = 4096, rng: np.random.Generator | None = None
) -> float:
    """Largest per-feature standard deviation over the median: a scale-free
    anisotropy score for the coreset seed.

    A coreset is a uniform draw of ``m`` rows, so it represents a feature in
    proportion to how much of the data's variance that feature carries.  When
    every feature has the same scale that is harmless (the sweep's isotropic
    cells measure 1.10-1.30); when the scales span ``e^2`` -- the aniso cell,
    which multiplies each feature by ``exp(N(0, 1))`` -- the draw resolves the
    dominant directions and the coreset Lloyd seed lands 3.4% behind a
    full-data k-means++ seed at 20000 x 32 k=64.  The median (not the min) in
    the denominator keeps a couple of near-constant features from dominating:
    pixels and one-hot columns are common and are not anisotropy.

    One sampled standard deviation, capped at ``sample`` rows, and the estimate
    tracks the full-data value (4.10 against 4.09 on the aniso cell, 1.30
    against 1.30 on the wide isotropic one).  A deterministic local RNG is used
    by default so measuring the guard cannot shift the seed draw that follows.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("feature_scale_spread needs a non-empty 2-D array")
    n = data.shape[0]
    if n > int(sample):
        picker = np.random.default_rng(0) if rng is None else rng
        data = data[picker.choice(n, size=int(sample), replace=False)]
    scale = data.std(axis=0)
    median = float(np.median(scale))
    largest = float(scale.max())
    if median <= 0.0:
        # every feature constant: no scale to be anisotropic about, unless one
        # is not, in which case the ratio is unbounded by construction
        return 1.0 if largest <= 0.0 else float("inf")
    return largest / median


def coreset_size(
    n: int,
    k: int,
    *,
    per_cluster: int = 32,
    floor: int = CORESET_MIN,
) -> int:
    """Coreset rows for ``init="coreset"``: ``clamp(32k, 512, n)``.

    The measured curve is non-monotone (the draw matters as much as the size),
    so this is a plateau rule, not an optimum: 32 rows per cluster is where
    every cell measured is within a few percent of
    its best observed value, while the very small ``max(4k, 512)`` seed is the
    worst point in most cells.  Going past ``32k`` costs init time without a
    systematic gain -- at 200k x 8 and under Zipf imbalance ``32k`` is the best
    point measured, and at 20k x 32 the best draw is ``128k`` but ``4m`` is not
    reliably better than ``2m``.
    """
    return int(min(int(n), max(int(per_cluster) * int(k), int(floor))))


def coreset_lloyd_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    coreset: int | None = None,
    niter: int = 10,
    sampling: str = "random",
) -> np.ndarray:
    """Seed by running Lloyd to completion on a coreset.

    The coreset is the ``init="coreset"`` rule (``max(4k, 512)``), not the
    D^1.5 seed's 16-rows/centre/1,024-row floor.  The clustering on it is
    Lloyd's 10 full passes rather than 3 EMA epochs, which
    is the difference that matters: e-means on the coreset never converges and
    measured 1.31x median MSE as a seeding device, while this measures 1.05x.
    The result is then grown into the full set by the caller's own fit -- that
    is the hierarchical step, and it costs nothing extra because the fit was
    going to run anyway.

    Measured (7 cells, 2 seeds, MSE / total wall against full-data k-means++):
    1.05x median MSE for ~2x less wall, and the best arm outright on the k=128
    shape (9.08 against 9.83 / 9.68, ARI 0.909, 66 against 162 ms).  Available
    as ``init="coreset"``; not the default, because it is 5% behind k-means++
    on the easy shapes and defaults do not move on a trade without a decision.
    """
    data = np.asarray(data, dtype=np.float64)
    m = int(coreset) if coreset else coreset_size(data.shape[0], k)
    sample = data
    if m < data.shape[0]:
        if str(sampling) == "stratified":
            sample = stratified_sample(data, m, rng)
        elif str(sampling) == "random":
            sample = data[rng.choice(data.shape[0], size=m, replace=False)]
        else:
            raise ValueError(
                f"unknown sampling {sampling!r}; available: random, stratified"
            )
    model = LloydKMeans(
        int(k), seed=int(rng.integers(2**31)), niter=int(niter), n_init=1
    )
    return model.fit(sample, niter=int(niter), n_init=1).params["centroids"]


def multi_coreset_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    coreset: int | None = None,
    niter: int = 10,
    sampling: str = "random",
    draws: int = 2,
) -> np.ndarray:
    """Draw several coreset seeds and keep the best one on the **full** objective.

    The size sweep found the draw matters as much as the size -- one 1024-row
    coreset scored 175.5 where another of the same size scored 127.7 -- so the
    honest remedy is not a bigger sample but more than one, judged on the data
    the fit will actually run on.  Each candidate is scored by the full-data
    inertia ``sum_i min_j ||x_i - c_j||^2``, which is the k-means objective the
    seed exists to start, and is one ``O(n k d)`` pass per draw: the same work
    as a single fit epoch, which is what makes this measurable rather than free.

    ``draws=1`` is the plain single-coreset seed; the default of 2 is chosen by
    measured, not assumed.
    """
    data = np.asarray(data, dtype=np.float64)
    draws = max(1, int(draws))
    best = None
    best_score = math.inf
    for _ in range(draws):
        candidate = coreset_lloyd_seed(
            data, k, rng, coreset=coreset, niter=niter, sampling=sampling
        )
        score = float(squared_distances(data, candidate).min(axis=1).sum())
        if score < best_score:
            best, best_score = candidate, score
    return best


def guarded_coreset_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    coreset: int | None = None,
    niter: int = 10,
    draws: int = 2,
    check_rows: int = 4096,
    samplings: tuple[str, ...] = ("random", "stratified"),
    refine_rows: int | None = 4096,
    refine_niter: int = 3,
    verify: bool = False,
) -> np.ndarray | tuple[np.ndarray, bool]:
    """Draw coresets and keep the one that fits a **held-out slice** best.

    The five-draw study left one problem with the
    coreset seed: the median is fine, but a single observation was 52% worse than
    k-means++ because the random draw occasionally lands on a coreset that
    merges clusters.  The remedy is not a bigger sample (``32k`` rows is already
    a plateau) and not more draws judged on the *full* data (that cost 2.6-5x the
    init and measured no gain) -- it is judging a couple of cheap draws on a
    slice the seed did not see, which is what detects a merged cluster.

    ``draws`` candidates are taken from ``samplings`` in rotation (by default one
    uniform and one stratified), and each is scored by the inertia of
    ``check_rows`` rows sampled once: one ``O(check_rows * k * d)`` pass per
    draw, a fraction of a fit epoch.  The winner is then refined with a few
    Lloyd passes on another ``refine_rows`` sample: selection on a slice fixes
    which draw wins, refinement fixes the 512-row local optimum the winner was
    stuck in.  Both stages are optional and both are far below a full-data
    k-means++ init.

    Measured; this is the rule
    ``init="auto"`` resolves to when the coreset branch is chosen.
    """
    data = np.asarray(data, dtype=np.float64)
    draws = max(1, int(draws))
    check = data
    if check_rows and data.shape[0] > int(check_rows):
        rows = rng.choice(data.shape[0], size=int(check_rows), replace=False)
        check = data[rows]
    best: np.ndarray | None = None
    best_score = math.inf
    for i in range(draws):
        sampling = samplings[i % len(samplings)] if samplings else "random"
        candidate = coreset_lloyd_seed(
            data, k, rng, coreset=coreset, niter=niter, sampling=sampling
        )
        score = float(squared_distances(check, candidate).min(axis=1).sum())
        if best is None or score < best_score:
            best, best_score = candidate, score
    assert best is not None
    if refine_rows and refine_niter and data.shape[0] > int(refine_rows):
        pick = rng.choice(data.shape[0], size=int(refine_rows), replace=False)
        best = _lloyd_refine(data[pick], best, niter=int(refine_niter))
    if not verify:
        return best
    # Structural check, not inertia.  Both full-data and slice inertia were
    # measured as selection criteria and neither predicted held-out MSE after the
    # fit re-optimises (full-data multi-draw: no gain; slice guard: worst case
    # still +88%).  What a bad draw actually does is merge clusters, which shows
    # up as components with almost no members on the slice the seed did not see.
    # Reject when any component takes less than a quarter of its equal share.
    dist = squared_distances(check, best)
    counts = np.bincount(dist.argmin(axis=1), minlength=int(k))
    total = max(float(counts.sum()), 1.0)
    share = counts / total
    return best, float(share.min()) >= 0.25 / float(k)


def stratified_sample(
    data: np.ndarray, m: int, rng: np.random.Generator
) -> np.ndarray:
    """Evenly spaced rows along a random projection (variance-reduced coreset).

    A random coreset's seeding error has two parts: bias from ``m`` being small,
    and the draw itself.  The size sweep shows the draw dominates above ~32 rows
    per cluster -- one 1024-row coreset measured 175.5 where another of the same
    size measured 127.7.  Stratifying on one cheap projection (a single matvec,
    then a strided pick) spreads the sample across the data's main axis instead
    of leaving a gap, which is the standard variance reduction and costs one
    pass over the rows.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    m = int(m)
    if m >= n or m < 2:
        return data
    key = data @ rng.normal(size=data.shape[1])
    order = np.argsort(key, kind="stable")
    idx = np.clip((np.arange(m) * (n / m)).astype(np.int64), 0, n - 1)
    return data[order[idx]]


def _weighted_lloyd(
    points: np.ndarray,
    weights: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    niter: int = 10,
) -> np.ndarray:
    """Lloyd over weighted points (the combination step for shard seeds)."""
    points = np.asarray(points, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n, d = points.shape
    k = int(k)
    centres = np.empty((k, d), dtype=np.float64)
    first = int(rng.choice(n, p=w / w.sum())) if w.sum() > 0 else 0
    centres[0] = points[first]
    closest = np.sum((points - centres[0]) ** 2, axis=1)
    for j in range(1, k):
        prob = w * closest
        total = float(prob.sum())
        if total > 0.0:
            centres[j] = points[int(rng.choice(n, p=prob / total))]
        else:
            centres[j] = points[int(rng.integers(n))]
        closest = np.minimum(
            closest, np.sum((points - centres[j]) ** 2, axis=1)
        )
    rows = np.arange(n)
    wp = w[:, None] * points
    for _ in range(int(niter)):
        dist = squared_distances(points, centres)
        lab = dist.argmin(axis=1)
        wsum = np.bincount(lab, weights=w, minlength=k)
        sums = np.empty((k, points.shape[1]), dtype=np.float64)
        for dim in range(points.shape[1]):
            sums[:, dim] = np.bincount(lab, weights=wp[:, dim], minlength=k)
        empty = wsum <= 0.0
        if np.any(empty):
            worst = int(np.argmax(w * dist[rows, lab]))
            centres[empty] = points[worst]
        live = ~empty
        centres[live] = sums[live] / wsum[live, None]
    return centres


def _lloyd_refine(points: np.ndarray, centres: np.ndarray, *, niter: int = 10) -> np.ndarray:
    """Plain Lloyd on ``points`` starting from ``centres`` (the recovery step)."""
    points = np.asarray(points, dtype=np.float64)
    centres = np.array(centres, dtype=np.float64, copy=True)
    k = centres.shape[0]
    n = points.shape[0]
    rows = np.arange(n)
    for _ in range(int(niter)):
        dist = squared_distances(points, centres)
        lab = dist.argmin(axis=1)
        counts = np.bincount(lab, minlength=k).astype(np.float64)
        sums = np.empty((k, points.shape[1]), dtype=np.float64)
        for dim in range(points.shape[1]):
            sums[:, dim] = np.bincount(lab, weights=points[:, dim], minlength=k)
        empty = counts <= 0.0
        if np.any(empty):
            worst = int(np.argmax(dist[rows, lab]))
            centres[empty] = points[worst]
        live = ~empty
        centres[live] = sums[live] / counts[live, None]
    return centres


def _weighted_fps(points: np.ndarray, weights: np.ndarray, k: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Coverage-style combination: weighted farthest-point over the candidates."""
    w = np.asarray(weights, dtype=np.float64)
    first = int(rng.choice(points.shape[0], p=w / w.sum()))
    chosen = [first]
    closest = ((points - points[first]) ** 2).sum(axis=1)
    for _ in range(1, int(k)):
        j = int(np.argmax(closest * w))
        chosen.append(j)
        closest = np.minimum(closest, ((points - points[j]) ** 2).sum(axis=1))
    return np.asarray(points)[chosen]


def shard_union_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    shards: int = 4,
    coreset: int | None = None,
    niter: int = 10,
) -> np.ndarray:
    """Seed from several *disjoint* coresets, then combine their centroids.

    Each shard is clustered on its own with Lloyd, and the ``shards * k``
    resulting centroids are then re-clustered down to ``k`` with weights equal
    to the shard cluster sizes.  The point of alternating disjoint coresets is
    coverage: one coreset sees ``m`` rows, four shards see ``4m`` while keeping
    every individual clustering cheap.  The control it has to beat is a single
    coreset of the same total size (``coreset=4m``), which ``--`` measured in
    a separate study -- is the only fairness question here.
    """
    data, points, weights, pooled = _shard_parts(
        data, k, rng, shards=shards, coreset=coreset, niter=niter
    )
    if not points:
        return coreset_lloyd_seed(
            data, k, rng,
            coreset=max(int(coreset or 0), 4 * int(k), CORESET_MIN), niter=niter,
        )
    return _weighted_lloyd(
        np.vstack(points), np.concatenate(weights), k, rng, niter=niter
    )


def _shard_parts(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    shards: int = 4,
    coreset: int | None = None,
    niter: int = 10,
) -> tuple[np.ndarray, list, list, np.ndarray]:
    """Disjoint coresets, each clustered with Lloyd; returns candidates + rows."""
    data = np.asarray(data, dtype=np.float64)
    n = data.shape[0]
    m = int(coreset) if coreset else coreset_size(n, k)
    shards = max(1, int(shards))
    picked = rng.permutation(n)[: min(n, m * shards)]
    points, weights, used = [], [], []
    for s in range(shards):
        rows = picked[s * m : (s + 1) * m]
        if rows.size < int(k):
            continue
        model = LloydKMeans(
            int(k), seed=int(rng.integers(2**31)), niter=int(niter), n_init=1
        )
        res = model.fit(data[rows], niter=int(niter), n_init=1)
        points.append(np.asarray(res.params["centroids"], dtype=np.float64))
        weights.append(np.asarray(res.params["counts"], dtype=np.float64))
        used.append(rows)
    pooled = np.concatenate(used) if used else picked
    return data, points, weights, pooled


def shard_refine_seed(
    data: np.ndarray,
    k: int,
    rng: np.random.Generator,
    *,
    shards: int = 4,
    coreset: int | None = None,
    niter: int = 10,
    combine: str = "weighted",
) -> np.ndarray:
    """Alternate coresets, combine the candidates, then refine *on the rows*.

    This is the repaired version of :func:`shard_union_seed`.  That one stopped
    at re-clustering the ``shards * k`` candidate centroids, which measured
    1.055x median MSE against a single equal-size coreset (1.003x) because
    averaging local optima is not refinement.  Here the combined ``k`` centroids
    are used as the starting point of Lloyd on the pooled coreset *rows*, so a
    full-resolution stage exists before the final fit.

    ``combine="weighted"`` re-clusters the candidates with their cluster sizes;
    ``combine="fps"`` picks ``k`` of them by weighted farthest-point coverage.
    """
    data, points, weights, pooled = _shard_parts(
        data, k, rng, shards=shards, coreset=coreset, niter=niter
    )
    if not points:
        return coreset_lloyd_seed(
            data, k, rng,
            coreset=max(int(coreset or 0), 4 * int(k), CORESET_MIN), niter=niter,
        )
    cand = np.vstack(points)
    w = np.concatenate(weights)
    if str(combine) == "fps":
        centres = _weighted_fps(cand, w, int(k), rng)
    elif str(combine) == "weighted":
        centres = _weighted_lloyd(cand, w, int(k), rng, niter=niter)
    else:
        raise ValueError(f"unknown combine {combine!r}; available: weighted, fps")
    return _lloyd_refine(data[pooled], centres, niter=niter)


def shrink_final(
    mean: np.ndarray,
    smoothed: np.ndarray,
    counts: np.ndarray,
    tau: float = 0.20,
) -> np.ndarray:
    """Empirical-Bayes blend ``n/(n+tau) * mean + tau/(n+tau) * smoothed``.

    The final-pass mean is noisy for a cluster with few members; the smoothed
    EMA centroid is not.  ``tau = 0.20`` is the published count scale, and a
    zero-count cluster keeps the smoothed centroid instead of a fallback.
    """
    mean = np.asarray(mean, dtype=np.float64)
    smoothed = np.asarray(smoothed, dtype=np.float64)
    n = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    tau = max(float(tau), 0.0)
    if tau == 0.0:
        return mean.copy()
    weight = np.where(n > 0.0, n / (n + tau), 0.0)
    return weight[:, None] * mean + (1.0 - weight)[:, None] * smoothed


def anisotropic_split(
    mean: np.ndarray,
    count: float,
    distortion: float,
    feature_scale: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Two mean-preserving centroids along a scale-invariant distortion axis.

    The published step is ``m_split * sqrt(sse / (n * d)) * r * sigma_hat_feat``
    with ``r`` a random sign per feature: the magnitude follows the cluster's
    own RMS distortion and the direction follows the data's anisotropy rather
    than an isotropic Gaussian draw (which collapses near the origin and
    explodes under a global offset -- the failure the upstream recipe exists to
    fix).  ``m_split`` is not published as a number, and taking the per-feature
    form literally gives a step of norm ``sqrt(d)`` times the RMS radius, which
    measured 4.5-7.4x worse than the isotropic split this engine ships.  So the
    axis is kept and the *norm* is set to this engine's validated convention
    (``sqrt(inertia / (batch * d))``), i.e. the direction is anisotropic and the
    distance is the one already tuned here.
    """
    mean = np.asarray(mean, dtype=np.float64)
    d = mean.size
    scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
    if scale.size != d:
        scale = np.ones(d, dtype=np.float64)
    norm = float(np.sqrt(np.mean(scale**2))) if d else 0.0
    if np.isfinite(norm) and norm > 0.0:
        scale = scale / norm
    else:
        scale = np.ones(d, dtype=np.float64)
    signs = rng.choice(np.array([-1.0, 1.0]), size=d)
    axis = signs * scale
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        axis, norm = np.ones(d, dtype=np.float64), math.sqrt(d)
    step = math.sqrt(max(float(distortion), 0.0) / max(float(count) * d, 1e-12))
    delta = (step / norm) * axis
    return mean + delta, mean - delta


@dataclass
class PowerCosineSchedule:
    """``alpha0 * (floor + (1 - floor) * cos^power)``, decreasing.

    s-means uses ``floor = 0.06`` and ``power = 0.92``: a slower start and a
    steeper tail than the plain cosine e-means anneals on.
    """

    alpha0: float = 0.1
    floor: float = 0.06
    power: float = 0.92

    def _progress(self, ctx: UpdateContext) -> float:
        return min(1.0, max(0.0, (ctx.step + 1) / max(1, ctx.total_steps)))

    def alpha(self, ctx: UpdateContext) -> float:
        frac = 0.5 * (1.0 + math.cos(math.pi * self._progress(ctx)))
        return float(self.alpha0) * (self.floor + (1.0 - self.floor) * frac**self.power)

    def alpha_grid(self, total_steps: int, batch_size: int = 0,
                   dataset_size: int = 0) -> np.ndarray:
        """Vector form, equal to asking :meth:`alpha` for every step."""
        total = max(int(total_steps), 1)
        steps = np.arange(total, dtype=np.float64)
        p = np.clip((steps + 1.0) / total, 0.0, 1.0)
        frac = 0.5 * (1.0 + np.cos(np.pi * p))
        return float(self.alpha0) * (
            self.floor + (1.0 - self.floor) * frac**self.power
        )


from .kmeans import EMeans, LloydKMeans  # kmeans imports this lazily, inside functions


class SMeans(EMeans):
    """e-means with the s-means recipe knobs set, nothing else changed.

    Every mechanism lives in the functions above and is opt-in: seeding
    (``init="d1.5"``), the anisotropic split (``revival_geometry``), the
    shrinkage (``tau``), and the power-cosine step.  The compiled epoch runner
    is reused unchanged -- the kernel takes the same alpha grid and stops on a
    dead cluster so the Python fallback performs the split.
    """

    name = "smeans"

    def __init__(
        self,
        k: int,
        *,
        tau: float = 0.20,
        init: str = "d1.5",
        revival_geometry: str = "anisotropic",
        alpha_floor: float = 0.06,
        alpha_power: float = 0.92,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("finalize", True)
        # The published anisotropic split is a small-cluster mechanism; at
        # k=128 it measured 48.0 MSE against 9.6 for the isotropic split on the
        # same seed, so SMeans asks for the recipe but uses the measured-safe
        # geometry above the boundary.  k <= 64 is neutral-to-better.
        effective_geometry = revival_geometry
        if (str(revival_geometry) == "anisotropic"
                and int(k) > ANISOTROPIC_REVIVAL_MAX_K):
            effective_geometry = "isotropic"
        super().__init__(
            k,
            init=init,
            finalize_shrink=float(tau),
            revival_geometry=effective_geometry,
            **kwargs,
        )
        self.alpha_floor = float(alpha_floor)
        self.alpha_power = float(alpha_power)

    def begin_fit(self, state, config, n, batch_size, total_steps):
        super().begin_fit(state, config, n, batch_size, total_steps)
        self.updater = EMA(
            schedule=PowerCosineSchedule(
                float(self._resolved_alpha0),
                floor=self.alpha_floor,
                power=self.alpha_power,
            )
        )
        self._alpha_cache = None

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "tau": self.finalize_shrink,
                "alpha_floor": self.alpha_floor,
                "alpha_power": self.alpha_power,
            }
        )
        return out


def smeans(
    data: np.ndarray,
    k: int,
    *,
    nepochs: int = 5,
    seed: int = 0,
    eval_data: np.ndarray | None = None,
    tau: float = 0.20,
    **kwargs: Any,
) -> "FitResult":  # noqa: F821  (FitResult imported for typing only)
    """Fit the s-means arm on ``data``, mirroring the ``emeans`` signature."""
    model = SMeans(k, tau=tau, nepochs=nepochs, seed=seed, **kwargs)
    return model.fit(data, nepochs=nepochs, seed=seed, eval_data=eval_data)


__all__ = [
    "CORESET_MIN",
    "SEED_ROUNDS",
    "coreset_size",
    "USAGE_EXPONENT",
    "PowerCosineSchedule",
    "SMeans",
    "anisotropic_split",
    "coreset_lloyd_seed",
    "multi_coreset_seed",
    "shard_union_seed",
    "shard_refine_seed",
    "d15_seed",
    "shrink_final",
    "smeans",
]
