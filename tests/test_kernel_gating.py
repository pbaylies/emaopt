"""The compiled-arm contract: one flag per question, and nothing implicit.

The two facts that used to be one were "the fused E-step can run" and "a
log-density can be produced"; a install served by an optional backend can answer
yes to the second and no to the first, and a caller that confuses them either
forces NumPy unnecessarily or raises inside a fit.
"""

from __future__ import annotations

import sys

import numpy as np

from emaopt import hostkernels, kernels


def _reset(monkeypatch) -> None:
    monkeypatch.setattr(kernels, "_INTERNAL", False)
    monkeypatch.setattr(hostkernels, "_numba_arm", False)


def test_available_matches_status(monkeypatch):
    monkeypatch.delenv("EMAOPT_HOST_KERNEL", raising=False)
    _reset(monkeypatch)
    assert hostkernels.available() == hostkernels.status()["available"]
    assert hostkernels.logits_available() == hostkernels.status()["logits_available"]


def test_disabling_turns_off_every_arm(monkeypatch):
    monkeypatch.setenv("EMAOPT_HOST_KERNEL", "numpy")
    monkeypatch.setenv("EMAOPT_NUMBA", "1")
    _reset(monkeypatch)
    assert hostkernels.available() is False
    assert hostkernels.logits_available() is False
    status = hostkernels.status()
    assert status["available"] is False and status["logits_available"] is False
    assert status["internal"] is False and status["numba"] is False
    assert hostkernels.logsumexp_rows(np.zeros((4, 2))) is None


def test_internal_provider_absent_by_default(monkeypatch):
    monkeypatch.delenv("EMAOPT_INTERNAL_KERNELS", raising=False)
    _reset(monkeypatch)
    assert kernels.internal() is None


def test_internal_provider_is_loaded_when_named(monkeypatch, tmp_path):
    (tmp_path / "internal_kernels.py").write_text(
        "def available():\n    return True\n\n"
        "def status():\n    return {'available': True}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EMAOPT_INTERNAL_KERNELS", str(tmp_path))
    _reset(monkeypatch)
    try:
        provider = kernels.internal()
        assert provider is not None and provider.available()
    finally:
        sys.modules.pop("internal_kernels", None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
        _reset(monkeypatch)


def test_logits_available_covers_the_fused_step(monkeypatch):
    monkeypatch.delenv("EMAOPT_HOST_KERNEL", raising=False)
    _reset(monkeypatch)
    if hostkernels.available():
        assert hostkernels.logits_available()
