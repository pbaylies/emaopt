"""T19: the device-resident eGMM loop (``emaopt.gpu.egmm_gpu_fit``).

The loop gives up bit-identity to the host arm by design, so the tests are
arranged around what it must still share with the library:

* **the batch order** -- if the loop walked the data differently it would be a
  different algorithm, and the pinned test compares its index stream against
  ``core._iter_batches`` row for row;
* **the fold algebra** -- ``S <- (1 - a) S + a s`` with ``a`` read from the
  model's own updater; the test uses each host updater as the oracle for a
  statistic of ones on a state of zeros, so every supported law is checked
  against its implementation rather than against a restatement of it;
* **the endpoint** -- decoded parameters and NLL against the host arm at f64,
  and again for ``spherical`` and ``finalize=True``.

Two tests need no device (batch order, fold algebra) plus one CPU-device
parity test, so the wiring is exercised on torch-only hosts too; the drift and
evaluation gates skip without CUDA.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from emaopt import GaussianMixture
from emaopt.core import FitConfig, State, UpdateContext, _iter_batches, fit
from emaopt.gpu import _batch_slices, _fold_alphas, egmm_gpu_fit
from emaopt.state import EMA, AdaptiveEMA, Replace, RunningMean

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _toy(seed: int = 0, n: int = 4096, d: int = 32, k: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + 3.0
    return x, k


def _rel_drift(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30))


def _param_drift(a, b):
    return max(_rel_drift(np.asarray(a[key]), np.asarray(b[key]))
               for key in ("means", "covariances", "weights"))


# --------------------------------------------------------------------------- #
# no device required: the two ways the loop could become a different algorithm
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shuffle,drop_last,batch,epochs,max_updates", [
    (True, True, 1024, 3, None),
    (False, True, 1024, 2, None),
    (True, False, 1000, 2, None),        # batch does not divide n
    (True, True, 1024, 5, 7),            # early stop mid-epoch
])
def test_batch_slices_match_library(shuffle, drop_last, batch, epochs,
                                    max_updates):
    """The device batch stream must be the library's batch stream, row for row."""
    n = 4096
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, 3))
    x[:, 0] = np.arange(n)               # unique rows so equality is a real test
    cfg = FitConfig(batch_size=batch, nepochs=epochs, seed=5, shuffle=shuffle,
                    drop_last=drop_last, max_updates=max_updates)
    # the library consumes the same generator the loop does
    from_library = list(_iter_batches(x, batch, cfg.nepochs,
                                      np.random.default_rng(cfg.seed),
                                      cfg.shuffle, cfg.drop_last,
                                      cfg.max_updates))
    from_loop = list(_batch_slices(n, batch, cfg.nepochs, cfg.seed,
                                   cfg.shuffle, cfg.drop_last,
                                   cfg.max_updates))
    assert len(from_library) == len(from_loop) > 0
    for (epoch, i, rows), (epoch2, i2, order, start, size) in zip(from_library,
                                                                  from_loop):
        assert (epoch, i) == (epoch2, i2)
        assert np.array_equal(rows, x[order[start : start + size]])


@pytest.mark.parametrize("factory", [
    lambda: Replace(),
    lambda: EMA(0.3),
    lambda: EMA(0.2, per_key={"counts": 0.5}),
    lambda: RunningMean(),
    lambda: AdaptiveEMA(amin=0.02, amax=0.5),
])
def test_fold_alpha_matches_host_updater(factory):
    """Each supported law's device blend coefficient equals the host updater's.

    A statistic of ones folded into a state of zeros leaves exactly ``a``, so
    the host updater itself is the oracle -- no restatement of the formula.
    """
    x, k = _toy(n=512, d=8)
    model = GaussianMixture(k, covariance="diag", updater=factory(), seed=0)
    seed_state = model.initialize(x)
    keys = model.state_keys
    zeros = State({key: np.zeros_like(np.asarray(seed_state[key], dtype=np.float64))
                   for key in keys}, 0, dict(seed_state.meta))
    ones = State({key: np.ones_like(np.asarray(seed_state[key], dtype=np.float64))
                  for key in keys}, 0, {})
    previous, statistic, aux = zeros, ones, {}
    for step in range(5):
        ctx = UpdateContext(step=step, batch_size=256, dataset_size=len(x),
                            total_steps=10, epoch=0, examples_processed=step * 256)
        host = model.updater.update(previous, statistic, ctx, keys=keys)
        alphas = _fold_alphas(model, ctx, keys, aux)
        for key in keys:
            assert alphas[key] == pytest.approx(float(np.ravel(host[key])[0]),
                                                abs=1e-12), key
        # hold the values at zero so the host output IS the coefficient, but
        # carry the updater's own recursion meta (RunningMean's ``_n``,
        # AdaptiveEMA's ``_ess``) exactly as the loop's ``aux`` does.
        previous = State({key: np.zeros_like(np.asarray(host[key])) for key in keys},
                         0, dict(host.meta))
        statistic = State({key: np.ones_like(np.asarray(host[key])) for key in keys},
                          0, {})


def test_unsupported_updater_is_refused_not_approximated():
    x, k = _toy(n=256, d=8)
    from emaopt.state import MomentumEMA

    model = GaussianMixture(k, covariance="diag", updater=MomentumEMA(), seed=0)
    with pytest.raises(NotImplementedError, match="affine blend"):
        _fold_alphas(model, UpdateContext(step=0, batch_size=64,
                                          dataset_size=len(x), total_steps=4),
                     model.state_keys, {})


def test_cpu_device_matches_host_fit():
    """torch-on-CPU is a real path: same endpoint as the host at f64."""
    x, k = _toy(n=1500, d=16)
    cfg = FitConfig(batch_size=500, nepochs=4, seed=0)
    host = fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg)
    loop = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                        device="cpu")
    assert _param_drift(loop.params, host.params) <= 1e-10
    assert abs(loop.objective - host.objective) <= 1e-10
    assert loop.updates == host.updates
    assert loop.extras["batch_size"] == host.extras["batch_size"]
    assert loop.extras["residency"]["bit_identity_to_cpu"] is False


@CUDA
def test_stat_dtype_f32_tracks_f64_within_the_reduction_error():
    """f32 statistics: a reduction-precision change, not an E-step change."""
    x, k = _toy(n=1024, d=16)
    cfg = FitConfig(batch_size=256, nepochs=2, seed=0)
    f64 = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                       device="cuda")
    f32 = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                       device="cuda", stat_dtype=torch.float32)
    assert f64.extras["residency"]["stat_dtype"] == "float64"
    assert f32.extras["residency"]["stat_dtype"] == "float32"
    assert abs(f32.objective - f64.objective) <= 1e-5
    for key, ref in f64.params.items():
        if not isinstance(ref, np.ndarray):      # e.g. params["covariance"]
            continue
        ref = np.asarray(ref, dtype=np.float64)
        got = np.asarray(f32.params[key], dtype=np.float64)
        assert np.max(np.abs(got - ref)) <= 1e-4 * max(
            float(np.max(np.abs(ref))), 1.0), key


def test_stat_dtype_refuses_anything_but_f64_f32():
    x, k = _toy(n=300, d=4)
    for bad in (torch.float16, torch.bfloat16, torch.int32):
        with pytest.raises(ValueError, match="stat_dtype"):
            egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x,
                         FitConfig(batch_size=150, nepochs=1, seed=0),
                         device="cpu", stat_dtype=bad)


def test_cpu_device_instrument_records_steps_without_device_events():
    """A CPU run still reports one record per step, with no segment timings."""
    x, k = _toy(n=900, d=8)
    cfg = FitConfig(batch_size=300, nepochs=2, seed=0)
    log: list[dict] = []
    result = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                          device="cpu", instrument=log)
    assert len(log) == result.updates == 6
    assert [rec["step"] for rec in log] == [1, 2, 3, 4, 5, 6]
    assert all(set(rec) == {"step"} for rec in log)
    assert result.extras["residency"]["nonfinite_logits"] == 0


def test_validation():
    x, k = _toy(n=128, d=8)
    with pytest.raises(TypeError, match="GaussianMixture"):
        egmm_gpu_fit(object(), x, device="cpu")
    with pytest.raises(ValueError, match="diagonal"):
        egmm_gpu_fit(GaussianMixture(k, covariance="full", seed=0), x,
                     device="cpu")
    with pytest.raises(NotImplementedError, match="revival"):
        egmm_gpu_fit(GaussianMixture(k, covariance="diag", revival=True,
                                     seed=0), x, device="cpu")
    with pytest.raises(ValueError, match="eval_mode"):
        egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x,
                     eval_mode="full", device="cpu")
    with pytest.raises(ValueError, match="stream"):
        egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x,
                     eval_data=x[:64], device="cpu")


# --------------------------------------------------------------------------- #
# CUDA gates
# --------------------------------------------------------------------------- #
@CUDA
def test_resident_loop_endpoint_parity_f64():
    """G1: same model as the host arm, not the same bits."""
    x, k = _toy(n=4096, d=32)
    cfg = FitConfig(batch_size=1024, nepochs=5, seed=0)
    host = fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg)
    loop = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                        device="cuda")
    assert loop.updates == host.updates == 20
    assert _param_drift(loop.params, host.params) <= 1e-9
    assert abs(loop.objective - host.objective) <= 1e-9
    assert set(loop.params) == set(host.params)


@CUDA
def test_resident_loop_exact_eval_reproduces_host_curve():
    """G2: scoring the full data on device matches the host evaluate()."""
    x, k = _toy(n=4096, d=32)
    cfg = FitConfig(batch_size=1024, nepochs=4, seed=0)
    host = fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg)
    loop = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                        device="cuda", eval_mode="exact")
    assert len(loop.history) == len(host.history) == 4
    gaps = [abs(a["objective"] - b["objective"])
            for a, b in zip(host.history, loop.history)]
    assert max(gaps) <= 1e-8, gaps
    assert [h["examples_processed"] for h in loop.history] == \
        [h["examples_processed"] for h in host.history]


@CUDA
def test_resident_loop_streamed_eval_and_instrument():
    """Stream mode records one objective per epoch and times each segment."""
    x, k = _toy(n=4096, d=32)
    cfg = FitConfig(batch_size=1024, nepochs=3, seed=0)
    log: list[dict] = []
    loop = egmm_gpu_fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg,
                        device="cuda", instrument=log)
    assert len(log) == 12                      # every step instrumented
    for rec in log:
        for key in ("gather_ms", "estep_ms", "fold_ms"):
            assert rec[key] >= 0.0
    assert loop.extras["residency"]["eval"] == "stream"
    assert loop.extras["residency"]["nonfinite_logits"] == 0
    host = fit(GaussianMixture(k, covariance="diag", seed=0), x, cfg)
    gaps = [b["objective"] - a["objective"]
            for a, b in zip(host.history, loop.history)]   # streamed - host
    assert len(gaps) == 3
    # A streamed objective averages each batch under the parameters in force
    # BEFORE that batch's update, while the host scores the whole set after
    # the epoch's last update -- so it is a lagged (higher) estimate: never
    # below the host's number by more than round-off, and bounded above.
    assert min(gaps) >= -1e-9, gaps
    assert max(gaps) <= 0.2, gaps
    # the curve still falls, and the loop's own final score is exact
    assert loop.history[0]["objective"] > loop.history[-1]["objective"]
    assert abs(loop.objective - host.objective) <= 1e-9


@CUDA
def test_resident_loop_spherical_and_finalize():
    """The two other supported configurations keep the same endpoint."""
    x, k = _toy(n=4096, d=24)
    cfg = FitConfig(batch_size=1024, nepochs=4, seed=0)
    for kwargs in ({"covariance": "spherical"}, {"covariance": "diag",
                                                 "finalize": True}):
        host = fit(GaussianMixture(k, seed=0, **kwargs), x, cfg)
        loop = egmm_gpu_fit(GaussianMixture(k, seed=0, **kwargs), x, cfg,
                            device="cuda")
        assert _param_drift(loop.params, host.params) <= 1e-9, kwargs
        assert abs(loop.objective - host.objective) <= 1e-9, kwargs


# --------------------------------------------------------------------------- #
# captured step (opt-in CUDA graph): same kernels, so the same bits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("updater", ["ema", "replace"])
@CUDA
def test_cuda_graph_step_is_bit_identical(updater):
    """Capture is a launch optimisation, not a numeric one: exact equality.

    The batch size does not divide n, so the loop captures two shapes (the full
    batch and the tail) and both must stay exact.  ``replace`` exercises the
    in-place ``copy_`` fold and ``ema`` the in-place blend.
    """
    x, k = _toy(n=1000, d=16)

    def build():
        return GaussianMixture(
            k, covariance="diag", seed=0,
            updater=Replace() if updater == "replace" else EMA(0.25),
        )

    cfg = FitConfig(batch_size=400, nepochs=2, seed=0, drop_last=False)
    eager = egmm_gpu_fit(build(), x, cfg, device="cuda")
    graph = egmm_gpu_fit(build(), x, cfg, device="cuda", cuda_graph=True)
    assert graph.updates == eager.updates == 6
    for key in ("means", "covariances", "weights"):
        assert np.array_equal(np.asarray(graph.params[key]),
                              np.asarray(eager.params[key])), key
    assert graph.objective == eager.objective
    residency = graph.extras["residency"]
    assert residency["cuda_graph"] is True
    assert residency["graph_replays"] == graph.updates
    assert residency["graphs"] == 2          # 400-row batches and the 200-row tail
    assert eager.extras["residency"]["cuda_graph"] is False


@CUDA
def test_cuda_graph_refuses_what_it_cannot_do():
    x, k = _toy(n=256, d=8)
    cfg = FitConfig(batch_size=128, nepochs=1, seed=0)
    model = lambda: GaussianMixture(k, covariance="diag", seed=0)  # noqa: E731
    with pytest.raises(ValueError, match="instrument"):
        egmm_gpu_fit(model(), x, cfg, device="cuda", cuda_graph=True,
                     instrument=[])
    with pytest.raises(ValueError, match="CUDA"):
        egmm_gpu_fit(model(), x, cfg, device="cpu", cuda_graph=True)
    with pytest.raises(ValueError, match="cuda_graph must be bool"):
        egmm_gpu_fit(model(), x, cfg, device="cpu", cuda_graph=1)


@CUDA
def test_resident_loop_tracks_host_trajectory_step_by_step():
    """The whole trajectory, not just the endpoint (callback trace)."""
    x, k = _toy(n=2048, d=16)
    cfg = FitConfig(batch_size=512, nepochs=3, seed=0)
    host_model = GaussianMixture(k, covariance="diag", seed=0)
    loop_model = GaussianMixture(k, covariance="diag", seed=0)
    a, b = [], []
    fit(host_model, x, cfg, callback=lambda st, c: a.append(host_model.decode(st)))
    egmm_gpu_fit(loop_model, x, cfg, device="cuda",
                 callback=lambda st, c: b.append(loop_model.decode(st)))
    assert len(a) == len(b) == 12
    worst = max(_param_drift(sa, sb) for sa, sb in zip(a, b))
    assert worst <= 1e-9, worst
