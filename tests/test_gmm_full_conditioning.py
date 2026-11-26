"""Full covariance must refuse the correlations it cannot identify.

A covariance estimated from ``n_j`` observations has at most ``n_j - 1``
identifiable directions in ``d`` dimensions.  Below ``n_j = d + 1`` the small
eigenvalues are noise, and flooring them at ``reg_covar`` makes a spike whose log
density at its own mean is ``O(-d log(reg) / 2)``; EM then parks one point on each
spike.  At d=512, k=8, batch 4096 that produced 667637 nats/point and seven of
eight components sitting exactly on the floor.

The guard keeps the diagonal for those components.  Two properties:

1. under-identified (streaming, small batch) fits stay finite, keep their
   components alive, and say so in ``regularized_components``;
2. identified fits are untouched -- the off-diagonal structure a full covariance
   exists for is still there, and nothing is reported as regularised.
"""

from __future__ import annotations

import numpy as np

from emaopt.algorithms import GaussianMixture
from emaopt.state import EMA


def _blobs(k: int, d: int, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d))
    labels = rng.integers(0, k, size=n)
    return centres[labels] + rng.normal(size=(n, d))


def test_underidentified_full_components_fall_back_to_diagonal():
    k, d = 4, 64
    x = _blobs(k, d, 3000)
    model = GaussianMixture(k, covariance="full", updater=EMA(alpha=0.1), seed=0)
    result = model.fit(
        x, batch_size=32, nepochs=2, seed=0, enforce_identifiability=False
    )

    # ~8 observations per component per batch against d=64: nothing off-diagonal
    # is identifiable, and before the guard this objective was in the 1e5 range.
    assert np.isfinite(result.objective)
    assert result.objective < 1e4, result.objective
    assert result.params["regularized_components"] == k

    weights = np.asarray(result.params["weights"])
    assert (weights > 1e-3).sum() >= k - 1, weights

    # the decoded covariance is the measured diagonal, not a reg_covar spike
    ev = np.linalg.eigvalsh(np.asarray(result.params["covariances"]))
    assert ev[:, 0].min() > 1e-3, ev[:, 0]


def test_initial_full_state_is_the_moment_decode_subtracts_from():
    """The initialisation fallback must store an *uncentred* second moment.

    At d=512 with the default 2048-point initialisation subsample there are ~512
    points per component, fewer than ``d + 1``, so every component takes the
    fallback branch.  It used to write ``diag(global_var)`` there, and decode
    subtracts ``mean mean^T`` from that slot -- negative variance in every
    dimension whose mean is non-zero.  Replace overwrote the state on the next
    update (the bug was invisible); EMA kept 90% of it per step and the objective
    ran to 6.4e6 nats/point.
    """
    k, d, n = 4, 512, 4000
    rng = np.random.default_rng(4)
    centres = rng.normal(size=(k, d))
    scales = np.exp(rng.uniform(np.log(0.5), np.log(2.0), size=(k, d)))
    labels = rng.integers(0, k, size=n)
    x = centres[labels] + rng.normal(size=(n, d)) * scales[labels]

    model = GaussianMixture(k, covariance="full", updater=EMA(alpha=0.1), seed=0)
    state = model.initialize(x)
    params = model.decode(state)
    diag = np.diag(np.asarray(params["covariances"])[0]) if False else np.einsum(
        "kii->ki", np.asarray(params["covariances"])
    )
    # every variance is a real data variance, not reg_covar and not negative
    assert diag.min() > 1e-2, diag.min()

    # and the streaming fit that used to diverge stays finite
    result = model.fit(
        x, batch_size=512, nepochs=2, seed=0, enforce_identifiability=False
    )
    assert np.isfinite(result.objective)
    assert result.objective < 1e4, result.objective


def test_full_covariance_raises_a_small_batch_to_its_identification_floor():
    """A batch too small to identify the off-diagonals is raised, and says so."""
    k, d, n = 4, 64, 12_000
    x = _blobs(k, d, n)

    model = GaussianMixture(k, covariance="full", seed=0)
    floor = model.identification_min_batch(d)
    assert floor == 5 * d * k

    result = model.fit(x, batch_size=64, nepochs=1, seed=0)
    assert result.extras["batch_size_requested"] == 64
    assert result.extras["batch_size"] == floor
    assert result.extras["identification_min_batch"] == floor
    # the raised batch identifies the family: the diagonal guard is not needed
    assert result.params["regularized_components"] < k

    # the same request, honoured, is the under-identified path the guard exists for
    under = model.fit(
        x, batch_size=64, nepochs=1, seed=0, enforce_identifiability=False
    )
    assert under.extras["batch_size"] == 64
    assert under.params["regularized_components"] == k

    # a diagonal component has no such floor: it stays estimable at batch 64
    diag = GaussianMixture(k, covariance="diag", seed=0).fit(
        x, batch_size=64, nepochs=1, seed=0
    )
    assert diag.extras["batch_size"] == 64
    assert diag.extras["identification_min_batch"] == 0


def test_identified_full_components_keep_their_correlations():
    k, d, n = 3, 8, 6000
    rng = np.random.default_rng(1)
    labels = rng.integers(0, k, size=n)
    factors = rng.normal(size=(k, d, d)) * 0.4
    x = np.empty((n, d), dtype=np.float64)
    for j in range(k):
        index = labels == j
        x[index] = (
            rng.normal(size=(int(index.sum()), d)) @ factors[j].T
            + rng.normal(size=d)
        )

    model = GaussianMixture(k, covariance="full", seed=0)
    result = model.fit(x, batch_size=None, nepochs=15, seed=0)

    assert result.params["regularized_components"] == 0
    cov = np.asarray(result.params["covariances"])
    off_diagonal = cov * (1.0 - np.eye(d))
    assert np.abs(off_diagonal).max() > 0.05, np.abs(off_diagonal).max()
