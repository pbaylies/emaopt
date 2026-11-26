"""Opt-in CUDA E-step, with no torch and no warm-up dependency.

``emaopt/gpu.py`` is the torch arm.  It is the right tool when torch exists, but
the E-step is small enough to keep a torch-free twin: one source that needs only
the CUDA toolkit, gates line by line against the C host kernel, and runs in a
Python environment we do not control (the GB10's venv has NumPy and torch; its
system Python has neither).  This module loads ``kernels/libegmm_diag_cuda.so``
-- built from ``kernels/egmm_diag_device.cu`` by ``python -m
emaopt.kernels.build --backend cuda`` -- with ctypes, so the runtime requirement
is the NVIDIA driver.

Device execution is a request, never a default.  A fit that silently changed
precision would make every recorded number ambiguous, so:

* ``EMAOPT_DEVICE_ESTEP=1`` (or ``on``/``cuda``/``true``) opts in;
* ``EMAOPT_DEVICE_ESTEP=numpy`` (or ``0``/``off``/``false``) opts out even when
  a caller passed the flag previously;
* absence of the artifact is not an error -- :func:`available` is ``False`` and
  the caller keeps the host path.

Precision: :func:`estep` defaults to ``float32`` logits/exp/responsibilities
with the statistics and the NLL still accumulated in double.  Measured against
the float64 host kernel at the score-bound shapes its counts/sums/sq agree to
1.2e-5 and its nll to 3.9e-10 (a device gate study); ``float64`` is
available for parity work and agrees to ~3e-14.

Two APIs: :func:`estep` is single-shot (it uploads data, parameters and
downloads the statistics on every call), while :class:`Session` uploads the
data once and keeps the parameter and statistics buffers on the device, so a
repeated E-step pays only a small parameter upload and the statistics
download.  Both are honest costs of a ctypes API, and they are why
:data:`WORK_GATE` exists: below the measured crossover the launch and the
parameter round trip dominate and the host kernel is faster.
"""

from __future__ import annotations

import ctypes
import os
import weakref
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["available", "status", "estep", "Session", "enabled",
           "work_gate", "WORK_GATE"]

_LIB_NAME = "libegmm_diag_cuda.so"
_PACKAGE_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _PACKAGE_DIR / "kernels"

_lib: Any = None
_error: str = ""
_probed = False
_info: dict[str, Any] = {}

#: ``rows * components * features`` below which a device call is not worth it.
#: The library dispatches through a resident :class:`Session`, so a step pays
#: only the parameter upload and the statistics download.  Measured with
#: a device gate study at nine shapes (best of 3-5; the GB10's GPU
#: is shared with VLLM): the RTX 4090 wins every cell down to 0.52M
#: (1.54-7.02x), so its class floor is 1M; the GB10 loses 0.52M (0.14-0.48x),
#: swings at 1.05M (0.35-1.40x), 2.1M (0.65-1.91x) and 4.2M (1.06-1.40x), and
#: only 41M and above are reliably 1.5-2.9x.  The floor is selected from the
#: probed device: >=100 SMs uses the 1M large-discrete floor, anything smaller
#: or shared keeps the 41M union.  The override ``EMAOPT_DEVICE_WORK_GATE``
#: remains for a machine that wants a different class floor.
WORK_GATE = 1_000_000
_SHARED_WORK_GATE = 41_000_000
_LARGE_DEVICE_SMS = 100

_DISABLED = {"numpy", "0", "off", "false", "no"}


def enabled() -> bool:
    """Whether the caller asked for device execution (``EMAOPT_DEVICE_ESTEP``)."""
    return os.environ.get("EMAOPT_DEVICE_ESTEP", "").strip().lower() in {
        "1", "cuda", "on", "true", "yes"}


def _candidate_path() -> Path:
    override = os.environ.get("EMAOPT_DEVICE_KERNEL_PATH")
    return Path(override) if override else _KERNEL_DIR / _LIB_NAME


def _load() -> Any:
    """Load and probe the artifact once; ``None`` with ``status()['error']`` set."""
    global _lib, _error, _probed, _info
    if _probed:
        return _lib
    _probed = True
    if os.environ.get("EMAOPT_DEVICE_ESTEP", "").strip().lower() in _DISABLED:
        _error = "disabled by EMAOPT_DEVICE_ESTEP"
        return None
    path = _candidate_path()
    if not path.exists():
        _error = (f"{path.name} is not built; run "
                  "`python -m emaopt.kernels.build --backend cuda`")
        return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:              # pragma: no cover - depends on the host
        _error = f"could not load {path.name}: {exc}"
        return None
    lib.emaopt_device_last_error.restype = ctypes.c_char_p
    lib.emaopt_device_probe.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int,
    ]
    lib.emaopt_device_probe.restype = ctypes.c_int
    f32 = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    f64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    tail = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
            f64, f64, f64, f64]
    lib.emaopt_device_estep_f32.argtypes = [f32, f32, f32, f32, f32] + tail
    lib.emaopt_device_estep_f32.restype = ctypes.c_int
    lib.emaopt_device_estep_f64.argtypes = [f64, f64, f64, f64, f64] + tail
    lib.emaopt_device_estep_f64.restype = ctypes.c_int
    if hasattr(lib, "emaopt_device_session_create"):
        lib.emaopt_device_session_create.argtypes = [
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int]
        lib.emaopt_device_session_create.restype = ctypes.c_void_p
        lib.emaopt_device_session_set_data.argtypes = [ctypes.c_void_p,
                                                       ctypes.c_void_p]
        lib.emaopt_device_session_set_data.restype = ctypes.c_int
        lib.emaopt_device_session_estep.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, f64, f64, f64, f64]
        lib.emaopt_device_session_estep.restype = ctypes.c_int
        lib.emaopt_device_session_destroy.argtypes = [ctypes.c_void_p]
        lib.emaopt_device_session_destroy.restype = None
    name = ctypes.create_string_buffer(256)
    major = ctypes.c_int(0)
    minor = ctypes.c_int(0)
    sms = ctypes.c_int(0)
    err = ctypes.create_string_buffer(512)
    code = lib.emaopt_device_probe(name, 256, ctypes.byref(major),
                                   ctypes.byref(minor), ctypes.byref(sms),
                                   err, 512)
    if code != 0:
        detail = err.value.decode("utf-8", "replace") or (
            lib.emaopt_device_last_error() or b"").decode("utf-8", "replace")
        _error = (f"{path.name} failed its device self-check: {detail}; "
                  "rebuild with `python -m emaopt.kernels.build --backend cuda`")
        return None
    _info = {
        "name": name.value.decode("utf-8", "replace"),
        "major": int(major.value),
        "minor": int(minor.value),
        "sms": int(sms.value),
    }
    _lib = lib
    return lib


def status() -> dict[str, Any]:
    """``{'available', 'path', 'error', 'name', 'major', 'minor', 'sms'}``."""
    lib = _load()
    out = {"available": lib is not None, "path": str(_candidate_path()),
           "error": _error}
    out.update(_info)
    return out


def available() -> bool:
    """Whether a probed device kernel is ready (does not check the opt-in flag)."""
    return _load() is not None


def work_gate() -> int:
    """The work floor to use here, honouring ``EMAOPT_DEVICE_WORK_GATE``.

    A large discrete device (>=100 SMs; the 4090 measured here) uses the 1M
    floor.  A smaller or shared device keeps the 41M union, because the GB10's
    4-8M cells swing with VLLM contention.  The override exists because the true
    crossover belongs to the machine, not to the library.
    """
    raw = os.environ.get("EMAOPT_DEVICE_WORK_GATE", "").strip()
    if raw:
        try:
            return max(1, int(float(raw)))
        except ValueError:
            pass
    if _info.get("sms", 0) < _LARGE_DEVICE_SMS:
        return _SHARED_WORK_GATE
    return WORK_GATE


def estep(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
          weights: np.ndarray | None = None, *,
          precision: str = "float32") -> dict[str, Any]:
    """One fused E-step on the device: ``counts``, ``sums``, ``sq``, ``nll``.

    The return is the same mapping :func:`emaopt.hostkernels.diag_estep`
    produces, so a caller can swap arms without changing the decode.  ``nll`` is
    per-row ``-log p(x)`` (the host kernel's convention).

    ``precision`` selects the logits/exp/responsibility arithmetic: ``float32``
    (default, the device's mode) or ``float64`` (parity).  The statistics and
    the NLL are summed in double in both.
    """
    if precision not in ("float32", "float64"):
        raise ValueError("precision must be 'float32' or 'float64'")
    lib = _load()
    if lib is None:
        raise RuntimeError(f"CUDA E-step kernel unavailable: {_error}")
    from .hostkernels import _prepare

    xq, invT, miT, bias, off = _prepare(x, means, covs, weights)
    n, d = xq.shape
    k = int(np.asarray(means).shape[0])
    dtype = np.float32 if precision == "float32" else np.float64
    fn = (lib.emaopt_device_estep_f32 if precision == "float32"
          else lib.emaopt_device_estep_f64)
    # _prepare returns (d, k) transposes; the C layout is [c*k + j], i.e. C-order.
    a_x, a_inv, a_mi, a_bias, a_off = (
        np.ascontiguousarray(arr.reshape(-1), dtype=dtype)
        for arr in (xq, invT, miT, bias, off))
    counts = np.empty(k, dtype=np.float64)
    sums = np.empty((k, d), dtype=np.float64)
    sq = np.empty((k, d), dtype=np.float64)
    nll = np.zeros(1, dtype=np.float64)
    code = fn(a_x, a_inv, a_mi, a_bias, a_off, n, d, k,
              counts, sums.reshape(-1), sq.reshape(-1), nll)
    if code != 0:
        raw = lib.emaopt_device_last_error()
        detail = (raw or b"").decode("utf-8", "replace") if raw else ""
        raise RuntimeError(f"CUDA E-step failed (code {code}): {detail}")
    return {"counts": counts, "sums": sums, "sq": sq,
            "nll": float(nll[0]) / max(int(n), 1), "precision": precision,
            "resident": False}


def _packed_parameters(means: np.ndarray, covs: np.ndarray,
                       weights: np.ndarray | None, dtype: Any) -> np.ndarray:
    """The parameter block the session uploads: ``[invT][miT][bias][off]``.

    Only parameters are touched here -- the data stays device-resident -- so
    this is the k*d work a repeated step still pays on the host.
    """
    means = np.asarray(means, dtype=np.float64)
    covs = np.asarray(covs, dtype=np.float64)
    k, d = means.shape
    inv = 1.0 / covs
    mean_inv = means * inv
    bias = (means * mean_inv).sum(axis=1)
    logdet = np.log(covs).sum(axis=1)
    if weights is None:
        logw = np.zeros(k, dtype=np.float64)
    else:
        logw = np.log(np.maximum(np.asarray(weights, dtype=np.float64), 1e-300))
    off = -0.5 * (d * float(np.log(2.0 * np.pi)) + logdet) + logw
    packed = np.empty(2 * d * k + 2 * k, dtype=dtype)
    dk = d * k
    packed[:dk] = inv.T.reshape(-1)
    packed[dk:2 * dk] = mean_inv.T.reshape(-1)
    packed[2 * dk:2 * dk + k] = bias
    packed[2 * dk + k:] = off
    return packed


def _destroy_session(lib: Any, handle: Any) -> None:
    try:
        lib.emaopt_device_session_destroy(handle)
    except Exception:                   # pragma: no cover - interpreter teardown
        pass


class Session:
    """A device-resident data copy for repeated E-steps (one fit's worth).

    ``Session(x, k)`` casts and uploads ``x`` once.  Each :meth:`estep` then
    sends only the ``k*d`` parameters and reads back the statistics, so the
    per-call allocation, cast and data upload that :func:`estep` pays
    disappear.  The data is assumed not to change while the session lives;
    create a new session (or call :meth:`close` and reopen) when it does.

    The handle is freed by :meth:`close`, by the context manager, or by a
    finalizer at garbage collection.  ``precision`` is ``float32`` (default) or
    ``float64``; statistics and NLL are summed in double in both.
    """

    def __init__(self, x: np.ndarray, k: int, *,
                 precision: str = "float32") -> None:
        if precision not in ("float32", "float64"):
            raise ValueError("precision must be 'float32' or 'float64'")
        data = np.asarray(x)
        if data.ndim != 2:
            raise ValueError("x must be a 2-D array")
        n, d = (int(data.shape[0]), int(data.shape[1]))
        k = int(k)
        if k <= 0:
            raise ValueError("k must be positive")
        lib = _load()
        if lib is None:
            raise RuntimeError(f"CUDA E-step kernel unavailable: {_error}")
        if not hasattr(lib, "emaopt_device_session_create"):
            raise RuntimeError(
                "the loaded CUDA artifact has no resident session API; "
                "rebuild with `python -m emaopt.kernels.build --backend cuda`")
        self._lib = lib
        self.n, self.d, self.k = n, d, k
        self.precision = precision
        self._dtype = np.float32 if precision == "float32" else np.float64
        self._handle = None
        err = ctypes.create_string_buffer(512)
        handle = lib.emaopt_device_session_create(
            n, d, k, 0 if precision == "float32" else 1, err, 512)
        if not handle:
            detail = err.value.decode("utf-8", "replace") or (
                lib.emaopt_device_last_error() or b"").decode("utf-8", "replace")
            raise RuntimeError(f"CUDA E-step session failed: {detail}")
        self._handle = handle
        self._finalizer = weakref.finalize(self, _destroy_session, lib, handle)
        device_data = np.ascontiguousarray(data, dtype=self._dtype)
        code = lib.emaopt_device_session_set_data(
            handle, ctypes.c_void_p(int(device_data.ctypes.data)))
        if code != 0:
            self.close()
            raw = lib.emaopt_device_last_error()
            detail = (raw or b"").decode("utf-8", "replace") if raw else ""
            raise RuntimeError(f"CUDA E-step session upload failed: {detail}")

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        if self._handle is not None:
            self._finalizer()
            self._handle = None

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def estep(self, means: np.ndarray, covs: np.ndarray,
              weights: np.ndarray | None = None) -> dict[str, Any]:
        """One E-step from the resident copy; same mapping as :func:`estep`."""
        if self._handle is None:
            raise RuntimeError("session is closed")
        means = np.asarray(means)
        covs = np.asarray(covs)
        if means.shape != (self.k, self.d) or covs.shape != (self.k, self.d):
            raise ValueError(
                f"parameters must be ({self.k}, {self.d}); got "
                f"means {means.shape}, covs {covs.shape}")
        packed = _packed_parameters(means, covs, weights, self._dtype)
        counts = np.empty(self.k, dtype=np.float64)
        sums = np.empty((self.k, self.d), dtype=np.float64)
        sq = np.empty((self.k, self.d), dtype=np.float64)
        nll = np.zeros(1, dtype=np.float64)
        code = self._lib.emaopt_device_session_estep(
            self._handle, ctypes.c_void_p(int(packed.ctypes.data)),
            counts, sums.reshape(-1), sq.reshape(-1), nll)
        if code != 0:
            raw = self._lib.emaopt_device_last_error()
            detail = (raw or b"").decode("utf-8", "replace") if raw else ""
            raise RuntimeError(f"CUDA E-step session failed (code {code}): {detail}")
        return {"counts": counts, "sums": sums, "sq": sq,
                "nll": float(nll[0]) / max(self.n, 1),
                "precision": self.precision, "resident": True}
