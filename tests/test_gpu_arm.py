"""GPU arm for the streaming diagonal eGMM (emaopt.gpu, Phase 7 promotion).

Everything CUDA-dependent skips cleanly on CPU-only boxes; the two tests that
run anywhere prove the wiring is *additive* -- validation errors without
device access, and a bit-identical default path when the arm is never used.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("no CUDA device", allow_module_level=True)

from emaopt import FitConfig, GaussianMixture
from emaopt.core import fit
from emaopt.gpu import (
    egmm_gpu_measure,
    egmm_gpu_measure_resident,
    transcribe_diag_logprob,
)


def _toy(seed: int = 0, n: int = 4096, d: int = 32, k: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d)) + 3.0
    return x, k


def _arm(wire: bool, x, k, *, batch=1024, epochs=5, seed=0, dtype=torch.float64):
    model = GaussianMixture(k, covariance="diag", seed=seed)
    snaps: list[dict[str, np.ndarray]] = []
    if wire:
        model.measure = egmm_gpu_measure(model, device="cuda", dtype=dtype)
    cb = lambda st, ctx: snaps.append(
        {n_: v.copy() for n_, v in model.decode(st).items()
         if isinstance(v, np.ndarray)}
    )
    res = fit(model, x, FitConfig(batch_size=batch, nepochs=epochs, seed=seed),
              callback=cb)
    return res, snaps


def _rel_drift(a, b):
    return float(np.abs(a - b).max() / max(float(np.abs(b).max()), 1e-30))


def test_semantic_gate_elementwise_f64():
    """The transcription must match component_log_prob within ~1e-9 nats."""
    x, k = _toy(seed=1, n=2048)
    model = GaussianMixture(k, covariance="diag", seed=1)
    params = model.fit(x, batch_size=512, nepochs=1).params
    logits = transcribe_diag_logprob(
        torch.as_tensor(x, dtype=torch.float64, device="cuda"),
        torch.as_tensor(params["means"], dtype=torch.float64, device="cuda"),
        torch.as_tensor(params["covariances"], dtype=torch.float64, device="cuda"),
    )
    ref = model.component_log_prob(x, params)
    gap = float(np.abs(logits.cpu().numpy() - ref).max())
    assert gap < 1e-9, gap


def test_measure_matches_cpu_statistics():
    """Swapped-in GPU measure() returns the same State shapes/values as CPU."""
    x, k = _toy(seed=2, n=1024)
    cpu = GaussianMixture(k, covariance="diag", seed=2)
    state = cpu.initialize(x)
    batch = x[:256]
    ref = cpu.statistics_from(batch, cpu.decode(state))
    gpu = GaussianMixture(k, covariance="diag", seed=2)
    got = egmm_gpu_measure(gpu, device="cuda")(batch, state)
    assert set(got.keys()) == set(ref.keys()) == {"counts", "sums", "sq"}
    for key in ref.keys():
        a, b = np.asarray(got[key]), np.asarray(ref[key])
        assert a.dtype == np.float64 and b.dtype == np.float64
        scale = max(float(np.abs(b).max()), 1e-30)
        assert float(np.abs(a - b).max()) <= 1e-9 * scale, key
    assert abs(float(got.meta["nll"]) - float(ref.meta["nll"])) < 1e-9


def test_20_step_drift_at_f64():
    """Real fit loop, measure swapped only: 20-step drift must stay <= 1e-10."""
    x, k = _toy()
    rc, sc = _arm(False, x, k)
    rg, sg = _arm(True, x, k)
    assert len(sg) == len(sc) == 20  # 4096/1024 x 5 epochs (drop_last default)
    worst = max(
        max(_rel_drift(sg[i][key], sc[i][key])
            for key in ("means", "covariances", "weights"))
        for i in range(len(sc))
    )
    assert worst <= 1e-10, worst
    assert abs(rg.objective - rc.objective) <= 1e-10


def test_validation_without_device():
    """Family/type gates fire before any CUDA work (and on CPU-only hosts)."""
    x, _ = _toy(n=64, d=8)
    with pytest.raises(ValueError, match="diagonal"):
        egmm_gpu_measure(GaussianMixture(2, covariance="full", seed=0))
    with pytest.raises(TypeError, match="GaussianMixture"):
        egmm_gpu_measure(object())
    with pytest.raises(RuntimeError, match="CUDA"):
        egmm_gpu_measure(GaussianMixture(2, covariance="diag", seed=0),
                         device="cuda:7")


def test_instrument_records_transfer_and_diagnostics():
    x, k = _toy(n=2048)
    model = GaussianMixture(k, covariance="diag", seed=3)
    log: list[dict] = []
    model.measure = egmm_gpu_measure(model, device="cuda", instrument=log)
    fit(model, x, FitConfig(batch_size=512, nepochs=2, seed=3))
    assert len(log) == 8
    for rec in log:
        assert rec["payload_mb"] > 0
        assert rec["transfer_ms"] >= 0 and rec["compute_ms"] >= 0
        assert rec["stats_nonfinite"] == 0
        assert rec["resp_nonfinite"] == 0
        assert np.isfinite(rec["cov_min"]) and rec["cov_min"] > 0


def test_default_path_bit_identical_after_import():
    """Importing/wiring the GPU arm must not perturb an untouched model."""
    x, k = _toy(seed=4)
    reference = GaussianMixture(k, covariance="diag", seed=4).fit(
        x, batch_size=1024, nepochs=5, seed=4
    )
    # wire a *different* instance first, then refit the untouched one
    other = GaussianMixture(k, covariance="diag", seed=4)
    other.measure = egmm_gpu_measure(other, device="cuda")
    fit(other, x, FitConfig(batch_size=1024, nepochs=5, seed=4))
    after = GaussianMixture(k, covariance="diag", seed=4).fit(
        x, batch_size=1024, nepochs=5, seed=4
    )
    for key in ("means", "covariances", "weights"):
        assert np.array_equal(reference.params[key], after.params[key]), key
    assert reference.objective == after.objective


def _arm_resident(x, k, *, batch=1024, epochs=5, seed=0):
    model = GaussianMixture(k, covariance="diag", seed=seed)
    snaps = []
    model.measure = egmm_gpu_measure_resident(model, x)
    cb = lambda st, ctx: snaps.append(
        {n_: v.copy() for n_, v in model.decode(st).items()
         if isinstance(v, np.ndarray)}
    )
    res = fit(model, x, FitConfig(batch_size=batch, nepochs=epochs, seed=seed),
              callback=cb)
    return res, snaps


def test_resident_arm_is_bit_identical_to_transfer_arm():
    """The Phase-10 promotion gate: same computation, delivery only changed."""
    x, k = _toy()
    _, a = _arm(True, x, k)
    _, b = _arm_resident(x, k)
    assert len(a) == len(b) and len(b) > 5
    for sa, sb in zip(a, b):
        for key in sb:
            assert np.array_equal(sa[key], sb[key]), key
