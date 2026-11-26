"""ICA on the sufficient-stat protocol: batch pin, TD quality, merge gate."""

from __future__ import annotations

import numpy as np
import pytest

from emaopt import accel
from emaopt.algorithms import ica as ica_module
from emaopt.algorithms.ica import FastICAStat, fastica, fastica_td
from emaopt.benchmarks import datasets, metrics
from emaopt.stream import partitioned_stat, stat_fit


def _legacy_fastica(data, n_components=None, *, niter=500, tol=1e-8, alpha=1.0,
                    seed=0, record_history=True):
    """The pre-stat loop, frozen so the refactor can be pinned.

    The negentropy line calls ``accel.logcosh``, the same substrate the shipped
    path uses: the accelerator's values are NumPy's to 8.9e-16 but not
    bit-identical, so pinning this loop against ``np.cosh`` would pin the
    transcendental rather than the refactor.  The substrate itself is bounded
    against NumPy by ``test_logcosh_substrate_agrees_with_numpy`` below.
    """
    data = np.asarray(data, dtype=np.float64)
    mean = data.mean(axis=0)
    _, wz = ica_module._whiten(data - mean, n_components)
    # the shipped path projects first and subtracts the projected mean, which
    # avoids an (n, d) temporary; mirror that form so this pin stays bit-exact
    xw = data @ wz.T - mean @ wz.T
    n, d = xw.shape
    rng = np.random.default_rng(seed)
    w = ica_module.symmetric_decorrelation(rng.normal(size=(d, d)))
    history: list[float] = []
    converged = False
    it = 0
    for it in range(1, int(niter) + 1):
        gx, gprime = ica_module._logcosh_g(xw @ w.T, alpha)
        w_new = (gx.T @ xw) / n - gprime.mean(axis=0)[:, None] * w
        w_new = ica_module.symmetric_decorrelation(w_new)
        delta = float(np.max(np.abs(np.abs(np.einsum("ij,ij->i", w_new, w)) - 1.0)))
        w = w_new
        if record_history:
            history.append(float(np.mean(accel.logcosh(xw @ w.T))))
        if delta < tol:
            converged = True
            break
    return {
        "unmixing": w @ wz,
        "history": history,
        "converged": converged,
        "n_iter": it,
        "objective": -float(np.mean(accel.logcosh(xw @ w.T))),
    }


@pytest.mark.parametrize("n_components", [None, 3])
def test_fastica_batch_mode_is_bit_pinned_to_the_pre_stat_loop(n_components):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(700, 5))
    old = _legacy_fastica(x, n_components, niter=40, seed=3)
    new = fastica(x, n_components, niter=40, seed=3)
    assert np.array_equal(new.unmixing, old["unmixing"])
    assert new.history == old["history"]
    assert new.converged == old["converged"]
    assert new.n_iter == old["n_iter"]
    assert new.objective == old["objective"]


def test_logcosh_substrate_agrees_with_numpy():
    """The accelerated log-cosh is NumPy's to machine epsilon, not bit for bit.

    libmvec and NumPy are different implementations of ``cosh`` and ``log``;
    measured 8.9e-16 absolute at 1e6 elements.  What this bounds is the size of
    that difference, which is what makes swapping the substrate safe for a
    diagnostic that never feeds the fixed-point step.
    """
    rng = np.random.default_rng(0)
    x = np.ascontiguousarray(rng.standard_normal(20_000) * 1.7)
    got = accel.logcosh(x)
    want = np.log(np.cosh(x))
    assert np.max(np.abs(got - want)) <= 1e-14
    assert np.max(np.abs(got)) > 0.1          # not a degenerate all-zero result


def test_td_fastica_reaches_batch_negentropy_on_a_laplace_fixture():
    sources = datasets.laplace_sources(n=6000, d=3, seed=0)
    mixed = datasets.linear_mix(sources, seed=1)
    batch = fastica(mixed.x, 3, niter=200, seed=0)
    # this gate is about the running path's trajectory; |corr| is not comparable
    # across whitening bases (a different basis rotates the recovered sources),
    # so auto has its own objective gate below.
    td = fastica_td(mixed.x, 3, batch_size=128, nepochs=6, decay=0.05, seed=0,
                    whitening="running")

    assert abs(td.objective - batch.objective) < 0.05
    assert td.objective <= batch.objective + 0.02
    corr = metrics.best_source_correlation(sources.x, td.sources)
    assert corr > 0.9, corr
    assert td.n_iter > 100
    assert td.examples_processed == 6 * 6000


def test_td_running_form_centres_a_significant_offset():
    """The one-pass running arm must not whiten a large constant offset."""
    sources = datasets.laplace_sources(n=6000, d=3, seed=0)
    mixed = datasets.linear_mix(sources, seed=1)
    shifted = mixed.x + 100.0
    result = fastica_td(shifted, 3, batch_size=128, nepochs=10, decay=0.05,
                        seed=0, whitening="running", record_history=False)
    # the running form stores its first-block estimate, not a full-stream mean
    assert np.allclose(result.mean, shifted.mean(axis=0), atol=0.25)
    corr = metrics.best_source_correlation(sources.x, result.sources)
    assert corr > 0.9, corr


def test_td_state_does_not_grow_with_the_dataset():
    rng = np.random.default_rng(2)
    small = rng.normal(size=(2000, 8))
    large = rng.normal(size=(40_000, 8))
    a = fastica_td(small, 4, batch_size=256, nepochs=1, decay=0.1, seed=0)
    b = fastica_td(large, 4, batch_size=256, nepochs=1, decay=0.1, seed=0)
    assert a.nbytes == b.nbytes
    assert a.nbytes < 8 * 8 * 8 * 4  # C + M + dprime + W, independent of n


def test_td_fastica_honours_a_memory_cap_and_records_it():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30_000, 24))  # 5.76 MB as float64
    result = fastica_td(x, 6, batch_size=4096, nepochs=1, decay=0.1, seed=0,
                        max_bytes=200_000)
    plan = result.config["compute_plan"]
    assert plan["mode"] == "stream"
    assert result.config["batch_size"] <= plan["block_size"]
    assert plan["limit_bytes"] == 100_000  # 0.5 * 200_000
    assert np.isfinite(result.objective)


def test_ica_stat_merge_over_partitions_tracks_the_single_stream():
    """Fixed whitening puts every partition's moment in one coordinate system.

    With ``whitening="running"`` each partition whitens in its own space, so
    merging the moments is comparing coordinates -- the tensors match up to a
    linear map, not elementwise.  The pipeline uses a whitening fixed by stage 1
    for exactly this reason; the merge gate uses the same setting.
    """
    rng = np.random.default_rng(4)
    x = rng.normal(size=(4000, 6))
    mean = x.mean(axis=0)
    covariance = ((x - mean).T @ (x - mean)) / x.shape[0]

    def make(mode):
        stat = FastICAStat(4, mode=mode, alpha=0.2, whitening="fixed",
                           act_every=10 ** 9, seed=1)
        stat.prepare(d=6, covariance=covariance, mean=mean)
        return stat

    from emaopt.stream import iter_blocks, stat_from_blocks

    single = stat_from_blocks(lambda: make("sum"), iter_blocks(x, 200))
    single.act()
    merged = partitioned_stat(lambda: make("sum"), x, n_partitions=4, block_size=200)
    a = single.decode()["unmixing"]
    b = merged.decode()["unmixing"]
    assert np.allclose(a.shape, b.shape)
    assert float(np.abs(a - b).max()) < 1e-9  # additive sum: exact reduction

    # the EMA merge is a superposition of decayed histories, not an identity;
    # the measured error is the law's, and it stays small on a stationary stream
    ema_single = stat_from_blocks(lambda: make("ema"), iter_blocks(x, 200))
    ema_single.act()
    ema_merged = partitioned_stat(lambda: make("ema"), x, n_partitions=4, block_size=200)
    rel = float(np.abs(ema_single.values["M"] - ema_merged.values["M"]).max()
                / max(float(np.abs(ema_single.values["M"]).max()), 1e-12))
    assert rel < 0.15, rel


def test_ica_stat_reports_the_same_footprint_as_the_td_run():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(3000, 5))
    stat = FastICAStat(3, mode="ema", alpha=0.1, whitening="running")
    stat.update(x)
    assert stat.nbytes > 0
    assert stat.config()["whitening"] == "running"
    assert stat.config()["acts"] >= 1


def test_ica_stat_normalizes_a_column_mean_with_a_supplied_covariance():
    stat = FastICAStat(2, whitening="fixed")
    stat.prepare(d=3, covariance=np.eye(3),
                 mean=np.array([[1.0], [2.0], [3.0]]))
    assert stat.params["mean"].shape == (3,)
    assert stat.whiten(np.ones((4, 3))).shape == (4, 2)
    assert stat.decode()["mean"].shape == (3,)


def test_auto_whitening_solves_once_and_matches_an_explicit_fixed_solve():
    """The default path pays one covariance pass, not one solve per block."""
    sources = datasets.laplace_sources(n=4000, d=3, seed=7)
    mixed = datasets.linear_mix(sources, seed=8)
    common = dict(batch_size=256, nepochs=2, seed=0, shuffle=False,
                  record_history=False)
    auto = fastica_td(mixed.x, 3, **common)
    mean = mixed.x.mean(axis=0)
    cov = ((mixed.x - mean).T @ (mixed.x - mean)) / mixed.x.shape[0]
    fixed = fastica_td(mixed.x, 3, whitening="fixed", covariance=cov,
                       mean=mean, **common)

    assert auto.config["whitening"] == "auto"
    assert auto.config["whitening_effective"] == "fixed"
    assert auto.config["covariance_source"] == "full_pass"
    assert auto.config["whitening_solves"] == 1
    # the prepass is the same second moment the explicit arm is handed, so the
    # two agree to rounding rather than merely by objective
    assert np.allclose(auto.unmixing, fixed.unmixing, atol=1e-9)
    assert abs(auto.objective - fixed.objective) < 1e-6


def test_auto_whitening_is_not_worse_than_the_running_path_by_its_objective():
    """The cheap default must win on the quantity it optimises.

    Measured on this fixture: auto -0.34839 vs running -0.34486 nats.  Source
    correlation is deliberately *not* the gate -- a different whitening basis
    rotates the recovered sources, which is the T61 lesson.
    """
    sources = datasets.laplace_sources(n=6000, d=3, seed=0)
    mixed = datasets.linear_mix(sources, seed=1)
    common = dict(batch_size=128, nepochs=6, decay=0.05, seed=0)
    auto = fastica_td(mixed.x, 3, **common)
    running = fastica_td(mixed.x, 3, whitening="running", **common)
    assert auto.config["whitening_solves"] == 1
    assert auto.objective <= running.objective + 0.02, (
        auto.objective, running.objective)


def test_whitening_diagnostics_report_conditioning():
    """A rank-starved covariance must be visible, not silently jittered."""
    stat = FastICAStat(None, mode="ema", alpha=0.2, whitening="fixed")
    stat.prepare(d=3, covariance=np.diag([1.0, 1e-20, 1e-20]))
    stat.whitening_matrix()
    diag = stat.config()["whitening_diagnostics"]
    assert diag["components"] == 3 and diag["dim"] == 3
    assert diag["clipped"] == 2
    assert diag["rcond"] <= 1e-11
    assert diag["max_eig"] == 1.0
