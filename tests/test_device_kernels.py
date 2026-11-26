"""The CUDA E-step is opt-in, torch-free, and must agree with the host path.

These tests do not require a GPU: the unavailable path is part of the contract,
and the dispatch wiring is exercised with a stand-in module.  The parity tests
skip when no artifact has been built for this machine.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from emaopt import devicekernels


def _reset(monkeypatch, path=None):
    monkeypatch.setattr(devicekernels, "_probed", False)
    monkeypatch.setattr(devicekernels, "_lib", None)
    monkeypatch.setattr(devicekernels, "_error", "")
    monkeypatch.setattr(devicekernels, "_info", {})
    if path is not None:
        monkeypatch.setenv("EMAOPT_DEVICE_KERNEL_PATH", str(path))


def _reference(x, means, covs, weights):
    inv = 1.0 / covs
    quad = (x * x) @ inv.T - 2.0 * (x @ (means * inv).T)
    quad += np.einsum("kd,kd->k", means, means * inv)[None, :]
    np.maximum(quad, 0.0, out=quad)
    off = -0.5 * (x.shape[1] * np.log(2.0 * np.pi)
                  + np.log(covs).sum(axis=1))
    if weights is not None:
        off = off + np.log(np.maximum(weights, 1e-300))
    logits = -0.5 * quad + off[None, :]
    peak = logits.max(axis=1, keepdims=True)
    lse = peak[:, 0] + np.log(np.exp(logits - peak).sum(axis=1))
    resp = np.exp(logits - lse[:, None])
    return ({"counts": resp.sum(axis=0), "sums": resp.T @ x,
             "sq": resp.T @ (x * x), "nll": float(-lse.mean())}, resp)


def test_missing_artifact_is_reported_not_raised(monkeypatch, tmp_path):
    _reset(monkeypatch, tmp_path / "absent.so")
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    assert devicekernels.available() is False
    assert "not built" in devicekernels.status()["error"]
    with pytest.raises(RuntimeError, match="unavailable"):
        devicekernels.estep(np.zeros((2, 2)), np.zeros((2, 2)),
                            np.ones((2, 2)))


def test_opt_out_environment_disables_the_probe(monkeypatch):
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "numpy")
    _reset(monkeypatch)
    assert devicekernels.available() is False
    assert "disabled" in devicekernels.status()["error"]


def _ready(monkeypatch):
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    monkeypatch.delenv("EMAOPT_DEVICE_KERNEL_PATH", raising=False)
    _reset(monkeypatch)
    if not devicekernels.available():
        pytest.skip(f"CUDA E-step artifact not built: "
                    f"{devicekernels.status()['error']}")


def test_float32_agrees_with_the_numpy_reference(monkeypatch):
    _ready(monkeypatch)
    rng = np.random.default_rng(3)
    n, k, d = 259, 5, 3
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = 0.5 + rng.random((k, d))
    weights = rng.random(k)
    weights /= weights.sum()
    ref, _ = _reference(x, means, covs, weights)
    got = devicekernels.estep(x, means, covs, weights, precision="float32")
    for key in ("counts", "sums", "sq"):
        assert np.allclose(got[key], ref[key], rtol=1e-4, atol=1e-4), key
    assert abs(got["nll"] - ref["nll"]) <= 1e-6 * max(1.0, abs(ref["nll"]))
    assert got["counts"].sum() == pytest.approx(float(n), rel=1e-5)


def test_float64_matches_the_numpy_reference(monkeypatch):
    _ready(monkeypatch)
    rng = np.random.default_rng(4)
    n, k, d = 257, 4, 6
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = 0.5 + rng.random((k, d))
    ref, _ = _reference(x, means, covs, None)
    got = devicekernels.estep(x, means, covs, None, precision="float64")
    for key in ("counts", "sums", "sq"):
        assert np.allclose(got[key], ref[key], rtol=1e-10, atol=1e-10), key
    assert abs(got["nll"] - ref["nll"]) <= 1e-12 * max(1.0, abs(ref["nll"]))


def _fake_device(calls):
    def estep(x, means, covs, weights=None, **kwargs):
        calls.append(np.asarray(x).shape[0])
        k, d = np.asarray(means).shape
        return {"counts": np.full(k, 5.0), "sums": np.zeros((k, d)),
                "sq": np.ones((k, d)), "nll": 1.25, "precision": "float32"}

    return types.SimpleNamespace(available=lambda: True, WORK_GATE=0,
                                 work_gate=lambda: 0, estep=estep,
                                 status=lambda: {})


def test_work_gate_selects_the_measured_device_class(monkeypatch):
    monkeypatch.delenv("EMAOPT_DEVICE_WORK_GATE", raising=False)
    monkeypatch.setattr(devicekernels, "_info", {"sms": 128})
    assert devicekernels.work_gate() == devicekernels.WORK_GATE
    monkeypatch.setattr(devicekernels, "_info", {"sms": 48})
    assert devicekernels.work_gate() == devicekernels._SHARED_WORK_GATE


def test_gmm_uses_the_device_only_when_opted_in(monkeypatch):
    import emaopt
    from emaopt.algorithms.gmm import GaussianMixture

    calls: list[int] = []
    monkeypatch.setattr(emaopt, "devicekernels", _fake_device(calls),
                        raising=False)
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    model = GaussianMixture(2, covariance="diag")
    params = {"means": np.zeros((2, 2)), "covariances": np.ones((2, 2)),
              "weights": np.array([0.5, 0.5])}
    state = model.statistics_from(np.ones((10, 2)), params)
    assert calls == [10]
    assert np.array_equal(state.values["counts"], np.full(2, 5.0))
    assert state.meta["nll"] == pytest.approx(1.25)
    assert state.meta["device"] == "float32"


def test_gmm_stays_on_the_host_without_the_opt_in(monkeypatch):
    import emaopt
    from emaopt.algorithms.gmm import GaussianMixture

    calls: list[int] = []
    monkeypatch.setattr(emaopt, "devicekernels", _fake_device(calls),
                        raising=False)
    monkeypatch.delenv("EMAOPT_DEVICE_ESTEP", raising=False)
    model = GaussianMixture(2, covariance="diag")
    params = {"means": np.zeros((2, 2)), "covariances": np.ones((2, 2)),
              "weights": np.array([0.5, 0.5])}
    state = model.statistics_from(np.ones((10, 2)), params)
    assert calls == []
    assert float(np.sum(state.values["counts"])) == pytest.approx(10.0, rel=1e-9)


def test_resident_session_matches_the_single_shot_wrapper(monkeypatch):
    _ready(monkeypatch)
    rng = np.random.default_rng(7)
    n, k, d = 513, 6, 3
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = 0.5 + rng.random((k, d))
    weights = rng.random(k)
    weights /= weights.sum()
    try:
        session = devicekernels.Session(x, k, precision="float64")
    except RuntimeError as exc:
        pytest.skip(f"resident session unavailable: {exc}")
    with session:
        got = session.estep(means, covs, weights)
        assert got["resident"] is True
        assert session.estep(means, covs, weights)["nll"] == pytest.approx(
            got["nll"], rel=1e-15)
        assert session.closed is False
    assert session.closed is True
    ref = devicekernels.estep(x, means, covs, weights, precision="float64")
    for key in ("counts", "sums", "sq"):
        np.testing.assert_allclose(got[key], ref[key], rtol=1e-12, atol=1e-12)
    assert got["nll"] == pytest.approx(ref["nll"], rel=1e-12)


def test_closed_session_refuses_work(monkeypatch):
    _ready(monkeypatch)
    x = np.zeros((4, 2))
    try:
        session = devicekernels.Session(x, 2)
    except RuntimeError as exc:
        pytest.skip(f"resident session unavailable: {exc}")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.estep(np.zeros((2, 2)), np.ones((2, 2)))


def test_resident_session_validates_parameter_shape(monkeypatch):
    _ready(monkeypatch)
    try:
        session = devicekernels.Session(np.zeros((4, 2)), 3)
    except RuntimeError as exc:
        pytest.skip(f"resident session unavailable: {exc}")
    with session:
        with pytest.raises(ValueError, match="parameters"):
            session.estep(np.zeros((2, 2)), np.ones((2, 2)))


class _FakeSession:
    """Counts construction/estep calls; same return shape as the real one."""

    instances: list["_FakeSession"] = []

    def __init__(self, data, k):
        self.shape = np.asarray(data).shape
        self.k = int(k)
        self.esteps = 0
        self.closed = False
        _FakeSession.instances.append(self)

    def estep(self, means, covs, weights=None):
        self.esteps += 1
        k, d = np.asarray(means).shape
        return {"counts": np.full(k, 5.0), "sums": np.zeros((k, d)),
                "sq": np.ones((k, d)), "nll": 1.25,
                "precision": "float32", "resident": True}

    def close(self):
        self.closed = True


def _fake_device_with_session(calls):
    fake = _fake_device(calls)
    fake.Session = _FakeSession
    return fake


def test_gmm_reuses_one_resident_session_across_steps(monkeypatch):
    import emaopt
    from emaopt.algorithms.gmm import GaussianMixture

    _FakeSession.instances = []
    calls: list[int] = []
    monkeypatch.setattr(emaopt, "devicekernels",
                        _fake_device_with_session(calls), raising=False)
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    monkeypatch.delenv("EMAOPT_DEVICE_RESIDENT", raising=False)
    model = GaussianMixture(2, covariance="diag")
    params = {"means": np.zeros((2, 2)), "covariances": np.ones((2, 2)),
              "weights": np.array([0.5, 0.5])}
    x = np.ones((10, 2))
    first = model.statistics_from(x, params)
    second = model.statistics_from(x, params)
    assert calls == []                       # single-shot wrapper bypassed
    assert len(_FakeSession.instances) == 1
    assert _FakeSession.instances[0].esteps == 2
    assert first.meta["device_resident"] is True
    assert second.meta["device_resident"] is True


def test_gmm_resident_keeps_the_host_batch_alive(monkeypatch):
    """The address-keyed cache must own the array whose address it keys on."""
    import gc
    import weakref

    import emaopt
    from emaopt.algorithms.gmm import GaussianMixture

    _FakeSession.instances = []
    monkeypatch.setattr(emaopt, "devicekernels",
                        _fake_device_with_session([]), raising=False)
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    model = GaussianMixture(2, covariance="diag")
    params = {"means": np.zeros((2, 2)), "covariances": np.ones((2, 2)),
              "weights": np.array([0.5, 0.5])}
    x = np.ones((10, 2))
    alive = weakref.ref(x)
    model.statistics_from(x, params)
    del x
    gc.collect()
    assert alive() is not None, (
        "the resident session dropped the host batch; a recycled address "
        "could then serve stale device statistics")


def test_gmm_resident_can_be_disabled(monkeypatch):
    import emaopt
    from emaopt.algorithms.gmm import GaussianMixture

    _FakeSession.instances = []
    calls: list[int] = []
    monkeypatch.setattr(emaopt, "devicekernels",
                        _fake_device_with_session(calls), raising=False)
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    monkeypatch.setenv("EMAOPT_DEVICE_RESIDENT", "0")
    model = GaussianMixture(2, covariance="diag")
    params = {"means": np.zeros((2, 2)), "covariances": np.ones((2, 2)),
              "weights": np.array([0.5, 0.5])}
    model.statistics_from(np.ones((10, 2)), params)
    assert calls == [10]
    assert _FakeSession.instances == []


def test_precision_is_validated(monkeypatch):
    monkeypatch.setenv("EMAOPT_DEVICE_ESTEP", "1")
    _reset(monkeypatch)
    with pytest.raises(ValueError, match="precision"):
        devicekernels.estep(np.zeros((2, 2)), np.zeros((2, 2)),
                            np.ones((2, 2)), precision="float16")
