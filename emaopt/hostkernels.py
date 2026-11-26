"""Optional compiled host kernels for the diagonal eGMM E-step (T21).

The default path of this library is NumPy and stays NumPy: nothing here is
imported unless a caller reaches for it, and every function falls back to the
NumPy implementation when the compiled artifact is absent.

Where the artifact pays
-----------------------
On the *fused* step -- logits, softmax, statistics and the NLL in one pass over
the batch -- because that is the entry point that stops re-reading the batch and
stops materialising ``(n, k)`` temporaries.  Whether that measures ahead of
NumPy at a given shape is a per-shape question, and the per-shape answer is in
``evidence/host_kernels.json``: the ratios there move between runs, so the
record from ``scripts/gen_hostkernels.py`` is the authority rather than a claim
in this docstring.

Where the code comes from
-------------------------
The C is a hand transcription of the same arithmetic written for an internal
code generator the maintainer runs outside this package (``tools/``), where the
two are checked against each other and both against NumPy.  Two optional arms
can serve the row-shaped kernels here: the numba backend (``kernels/numba.py``,
opt-in with ``EMAOPT_NUMBA=1``) and a maintainer-side provider reached through
``EMAOPT_INTERNAL_KERNELS``, which is absent from every release install and is
never named here.  Both are optional: the C artifact or NumPy always remains.

Installed from a wheel, build the artifact for this machine with
``python -m emaopt.kernels.build`` (default ``--backend c``; the C source is
packaged, the .so is not).
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["available", "logits_available", "tied_available", "status",
           "diag_logits", "diag_estep", "tied_logits", "tied_estep",
           "logsumexp_rows"]

_PACKAGE_DIR = Path(__file__).resolve().parent
_KERNEL_DIR = _PACKAGE_DIR / "kernels"
_LIB_NAME = "libegmm_diag.so"
_LOG_2PI = float(np.log(2.0 * np.pi))

_lib: Any = None
_error: str = ""
_probed = False
_stamp: dict[str, Any] = {}
#: Whether the loaded artifact exports ``egmm_logsumexp_rows``.  An artifact
#: built before that function existed still serves the E-step, so a missing
#: symbol must not fail the whole load -- it only disables this one entry point.
_has_logsumexp = False
#: Older artifacts have only the expanded log-density ABI.  The loader keeps
#: them usable instead of rejecting a build that predates the direct function.
_has_logits_direct = False

#: CPU features a ``-march=native`` build may have baked in.  The artifact is
#: compiled against whatever the *build* machine advertised, so any of these can
#: appear in the instruction stream; a host missing one would get SIGILL inside
#: a fit, not an exception a caller can catch.  The stamp lists them and the
#: loader refuses rather than crashing (T35).
_ISA_FLAGS = ("avx", "avx2", "fma", "avx512f", "avx512dq", "avx512vl",
              "avx512bw", "avx512cd", "avx512vnni")


def _candidate_path() -> Path:
    override = os.environ.get("EMAOPT_HOST_KERNEL_PATH")
    return Path(override) if override else _KERNEL_DIR / _LIB_NAME


def stamp_path(path: Path | None = None) -> Path:
    """Sidecar JSON written by the build script next to the shared object."""
    target = path or _candidate_path()
    return target.with_name(target.name + ".build.json")


def cpu_flags() -> set[str] | None:
    """This host's advertised ISA flags, or ``None`` where it cannot be read."""
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:                     # pragma: no cover - non-Linux
        return None
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return None                         # pragma: no cover - unusual kernel


def check_stamp(stamp: dict[str, Any], flags: set[str] | None) -> str | None:
    """Why this artifact cannot run here, or ``None`` when it can.

    An absent stamp, or a host whose flags cannot be read, is not a refusal:
    older artifacts predate the stamp and refusing them would turn a portability
    check into a build-system requirement.  The refusal is for the case that
    actually crashes -- a stamp that asks for a feature this CPU does not have.
    """
    if not stamp or flags is None:
        return None
    required = set(stamp.get("required_flags") or ())
    missing = sorted(required - flags)
    if not missing:
        return None
    return (f"artifact {stamp.get('artifact', 'libegmm_diag.so')} was built for "
            f"{stamp.get('machine', 'another CPU')} and needs "
            f"{', '.join(missing)}, which this CPU does not advertise; rebuild "
            "with `python -m emaopt.kernels.build`")


def _load() -> Any:
    """Load (once) the shared object; ``None`` with ``status()['error']`` set."""
    global _lib, _error, _probed
    if _probed:
        return _lib
    _probed = True
    setting = os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower()
    if setting in {"numpy", "0", "off", "false"}:
        _error = "disabled by EMAOPT_HOST_KERNEL"
        return None
    path = _candidate_path()
    if not path.exists():
        _error = (f"{path.name} is not built; run "
                  "`python -m emaopt.kernels.build` to build it")
        return None
    global _stamp
    sidecar = stamp_path(path)
    if sidecar.exists():
        try:
            _stamp = json.loads(sidecar.read_text())
        except (OSError, ValueError) as exc:   # pragma: no cover - corrupt file
            _error = f"could not read {sidecar.name}: {exc}"
            return None
        problem = check_stamp(_stamp, cpu_flags())
        if problem is not None:
            _error = problem
            return None
    try:
        lib = ctypes.CDLL(str(path))
    except OSError as exc:          # pragma: no cover - depends on the host
        _error = f"could not load {path.name}: {exc}"
        return None
    dbl_p = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.egmm_diag_logits.argtypes = [
        dbl_p, dbl_p, dbl_p, dbl_p, dbl_p, dbl_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
    ]
    lib.egmm_diag_logits.restype = ctypes.c_int
    global _has_logits_direct
    try:
        lib.egmm_diag_logits_direct.argtypes = [
            dbl_p, dbl_p, dbl_p, dbl_p, dbl_p,
            ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        lib.egmm_diag_logits_direct.restype = ctypes.c_int
        _has_logits_direct = True
    except AttributeError:      # pragma: no cover - artifact older than the fn
        _has_logits_direct = False
    lib.egmm_diag_estep.argtypes = [
        dbl_p, dbl_p, dbl_p, dbl_p, dbl_p,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        dbl_p, dbl_p, dbl_p,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.egmm_diag_estep.restype = ctypes.c_int
    global _has_logsumexp
    try:
        lib.egmm_logsumexp_rows.argtypes = [
            dbl_p, dbl_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int,
        ]
        lib.egmm_logsumexp_rows.restype = ctypes.c_int
        _has_logsumexp = True
    except AttributeError:      # pragma: no cover - artifact older than the fn
        _has_logsumexp = False
    # Exercise the object once.  A stale build with an older ABI loads happily
    # and then returns error codes or garbage, which is how a mismatched artifact
    # reaches a fit; here it is reported as unavailable instead of wrong.
    probe = np.ones((1, 1), dtype=np.float64)
    probe_zero = np.zeros(1, dtype=np.float64)
    probe_out = np.empty((1, 1), dtype=np.float64)
    direct_x = np.array([2.0], dtype=np.float64)
    direct_mean = np.array([0.5], dtype=np.float64)
    direct_inv = np.array([0.5], dtype=np.float64)
    direct_off = np.array([-0.3], dtype=np.float64)
    direct_out = np.empty((1, 1), dtype=np.float64)
    try:
        code = lib.egmm_diag_logits(probe, probe, probe, probe_zero,
                                    probe_zero, probe_out, 1, 1, 1, 1)
        if code != 0 or not np.isfinite(probe_out[0, 0]):
            raise RuntimeError(f"self-check status {code}")
        if _has_logits_direct:
            # Exercise the entry point ``_c_logits`` will actually use.  A
            # direct-only ABI or arithmetic break must not pass because the
            # legacy expanded function still works.
            code = lib.egmm_diag_logits_direct(
                direct_x, direct_mean, direct_inv, direct_off, direct_out,
                1, 1, 1, 1)
            if code != 0 or abs(float(direct_out[0, 0]) + 0.8625) > 1e-12:
                raise RuntimeError(f"direct self-check status {code}")
    except Exception as exc:            # pragma: no cover - stale artifact
        _error = (f"{path.name} failed its self-check ({exc}); rebuild with "
                  "`python -m emaopt.kernels.build`")
        return None
    _lib = lib
    return lib


def status() -> dict[str, Any]:
    """Which arm will run, and why (diagnostics; never raises).

    ``available`` is the fused E-step -- the C artifact and nothing else, the
    same answer :func:`available` gives.  ``logits_available`` covers every arm
    that can produce a log-density, so the two are separate facts rather than
    one flag that means different things to different callers.
    """
    setting = os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower()
    disabled = setting in {"numpy", "0", "off", "false"}
    lib = None if disabled else _load()
    provider = None if disabled else _internal()
    backend = None if disabled else _numba()
    return {
        "available": lib is not None,
        "logits_available": (lib is not None or provider is not None
                             or backend is not None),
        "path": str(_candidate_path()),
        "error": "disabled by EMAOPT_HOST_KERNEL" if disabled else _error,
        "internal": provider is not None,
        "internal_status": provider.status() if provider is not None else {},
        "numba": backend is not None and backend.available(),
        "numba_status": backend.status() if backend is not None else {},
        "stamp": dict(_stamp),
    }


def available() -> bool:
    """Whether the fused E-step can run: the C artifact, and only that.

    ``diag_estep`` needs the C artifact because no other arm implements the
    fused pass, so this is what gates it.  Callers that need the log-density
    alone must ask :func:`logits_available`, or an install served by an optional
    backend would be forced onto NumPy for a supported call.

    Re-reads ``EMAOPT_HOST_KERNEL`` on every call (cheap: an environment lookup)
    so a process can switch the compiled arms off mid-run -- that is how the
    reference arms in the tests stay NumPy.
    """
    setting = os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower()
    if setting in {"numpy", "0", "off", "false"}:
        return False
    return _load() is not None


def logits_available() -> bool:
    """Whether any arm can serve the ``(n, k)`` diagonal log-density."""
    setting = os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower()
    if setting in {"numpy", "0", "off", "false"}:
        return False
    return (_load() is not None or _internal() is not None
            or _numba() is not None)


def tied_available() -> bool:
    """Whether the maintainer provider has both tied E-step kernels built."""
    if os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower() in {
            "numpy", "0", "off", "false"}:
        return False
    provider = _internal()
    return bool(provider is not None and provider.tied_available())


def tied_logits(x: np.ndarray, means: np.ndarray, cov: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """``(n, k)`` tied-covariance log-density through the internal provider."""
    provider = _internal()
    if provider is None or not provider.tied_available():
        raise RuntimeError("tied internal kernel is not built")
    got = provider.tied_logits(x, means, cov, weights)
    if got is None:
        raise RuntimeError("tied internal kernel is not built")
    return got


def tied_estep(x: np.ndarray, means: np.ndarray, cov: np.ndarray,
               weights: np.ndarray | None = None, *,
               want_resp: bool = False) -> dict[str, Any]:
    """One tied E-step through the internal provider."""
    provider = _internal()
    if provider is None or not provider.tied_available():
        raise RuntimeError("tied internal kernel is not built")
    got = provider.tied_estep(x, means, cov, weights, want_resp=want_resp)
    if got is None:
        raise RuntimeError("tied internal kernel is not built")
    return got


def _threads(work: int | None = None) -> int:
    """Worker count: explicit override, then OpenMP's pin, then the affinity.

    ``OMP_NUM_THREADS`` is honoured because it is how callers and benches pin
    both arms alike.  Ignoring it let this kernel run 20 threads on a host the
    caller had pinned to one: measured at 3000x8, a 5-epoch fit took 5.95 ms at
    20 threads against 3.85 ms at one.

    ``work`` is accepted for call-site compatibility and deliberately ignored.
    It used to cap the pool at one thread per 1M ``n*k*d`` updates, which
    serialised the fused E-step exactly where threading pays: warm at eight
    threads, 3000x8 (192k updates) took 0.271 ms at the cap's single worker
    against 0.099 ms at eight, and 512x64 (524k) 0.194 against 0.098.  The C
    kernels keep per-thread accumulators and reduce in a fixed thread order, so
    the answer does not depend on the count, and the pool's spin-up is paid once
    per process rather than once per call.
    """
    raw = os.environ.get("EMAOPT_HOST_KERNEL_THREADS", "").strip()
    if not raw:
        raw = os.environ.get("OMP_NUM_THREADS", "").strip()
    count: int | None = None
    if raw:
        try:
            count = max(1, int(raw))
        except ValueError:
            count = None
    if count is None:
        try:
            count = max(1, len(os.sched_getaffinity(0)))
        except AttributeError:      # pragma: no cover - non-Linux
            count = max(1, os.cpu_count() or 1)
    return count


def _prepare(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
             weights: np.ndarray | None):
    """Contiguous float64 views, parameters transposed, k-vectors precomputed.

    The parameters go to the kernel as ``(d, k)`` so its inner loop (over the
    components) is a set of independent accumulators that vectorise without
    reassociating.  The transpose costs k*d*8 = 64 KB at (16, 512) and is
    dwarfed by the row pass; ``off`` folds the per-component constants.
    """
    x = np.ascontiguousarray(x, dtype=np.float64)
    means = np.ascontiguousarray(means, dtype=np.float64)
    covs = np.ascontiguousarray(covs, dtype=np.float64)
    k, d = means.shape
    inv = np.ascontiguousarray(1.0 / covs, dtype=np.float64)
    mean_inv = np.ascontiguousarray(means * inv, dtype=np.float64)
    quad_bias = np.ascontiguousarray((means * mean_inv).sum(axis=1),
                                     dtype=np.float64)
    logdet = np.ascontiguousarray(np.log(covs).sum(axis=1), dtype=np.float64)
    if weights is None:
        logw = np.zeros(k, dtype=np.float64)
    else:
        logw = np.ascontiguousarray(
            np.log(np.maximum(np.asarray(weights, dtype=np.float64), 1e-300)),
            dtype=np.float64)
    off = np.ascontiguousarray(-0.5 * (d * _LOG_2PI + logdet) + logw,
                               dtype=np.float64)
    return (x, np.ascontiguousarray(inv.T), np.ascontiguousarray(mean_inv.T),
            quad_bias, off)


#: The optional numba arm, imported lazily so a NumPy-only install never pays
#: for the import.  ``False`` = not probed yet.
_numba_arm: Any = False


def _numba() -> Any:
    """The numba backend module, or ``None`` when it is not in use.

    The environment is checked *before* the import: ``import numba`` is not
    cheap, and this is reached from ``status()`` and ``logits_available()``,
    which a caller may hit on every fit.
    """
    global _numba_arm
    if _numba_arm is False:
        if os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower() in {
                "numpy", "0", "off", "false"}:
            _numba_arm = None
        elif os.environ.get("EMAOPT_NUMBA", "").strip().lower() not in {
                "1", "on", "true", "yes", "numba"}:
            _numba_arm = None
        else:
            try:
                from .kernels import numba as backend

                _numba_arm = backend
            except Exception:        # pragma: no cover - numba absent
                _numba_arm = None
    return _numba_arm


def _internal() -> Any:
    """The optional internal provider, or ``None`` (see ``kernels.internal``)."""
    if os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower() in {
            "numpy", "0", "off", "false"}:
        return None
    from .kernels import internal

    return internal()



def _c_logits(lib: Any, x: np.ndarray, means: np.ndarray, covs: np.ndarray,
              weights: np.ndarray | None) -> np.ndarray:
    """The C artifact's ``(n, k)`` log-density."""
    x, invT, miT, bias, off = _prepare(x, means, covs, weights)
    n, d = x.shape
    k = means.shape[0]
    out = np.empty((n, k), dtype=np.float64)
    if _has_logits_direct:
        # The direct form needs meanT, not the expanded miT/bias pair.  Cast
        # here: ``_prepare`` normalises its own copy, so the caller's ``means``
        # may still be float32 or integer and ctypes requires float64.
        meanT = np.ascontiguousarray(means.T, dtype=np.float64)
        status_code = lib.egmm_diag_logits_direct(
            x, meanT, invT, off, out, n, d, k, _threads(n * d * k))
    else:
        status_code = lib.egmm_diag_logits(
            x, invT, miT, bias, off, out, n, d, k, _threads(n * d * k))
    if status_code != 0:
        raise RuntimeError(f"egmm_diag_logits failed (status {status_code})")
    return out


def logsumexp_rows(logits: np.ndarray) -> np.ndarray | None:
    """Row log-sum-exp through a compiled arm, or ``None`` when none is usable.

    One output per row, so no reduction crosses rows and every arm here threads
    without changing the answer.  Order: the C artifact, then the opt-in numba
    backend, then the internal provider.  Returns ``None`` rather than raising
    when nothing is available, was built before this entry point existed, or has
    been switched off with ``EMAOPT_HOST_KERNEL=numpy``; the caller keeps NumPy.

    Ratios at the eGMM scoring shapes, and the accuracy gap against NumPy, are
    regenerated by ``scripts/gen_hostkernels.py`` into
    ``evidence/host_kernels.json``.
    """
    if os.environ.get("EMAOPT_HOST_KERNEL", "").strip().lower() in {
            "numpy", "0", "off", "false"}:
        return None
    logits = np.ascontiguousarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        return None
    n, k = logits.shape
    lib = _load()
    if lib is not None and _has_logsumexp:
        out = np.empty(n, dtype=np.float64)
        # The row log-sum-exp is cheap per row, so the old one-thread-per-1M
        # updates cap gave this kernel a single worker at every scoring shape
        # (8192x16 = 131k updates): 0.600 ms against 0.164-0.185 with 4-8
        # workers.  It is a pure row map with no cross-row reduction, so there
        # is no correctness reason to keep it serial; let the caller's pin
        # decide, and let tiny row counts fall out of the OpenMP cost itself.
        code = lib.egmm_logsumexp_rows(logits.reshape(-1), out, n, k,
                                       _threads())
        if code == 0:
            return out
    backend = _numba()
    if backend is not None:
        got = backend.logsumexp(logits)
        if got is not None:
            return got
    provider = _internal()
    if provider is not None:
        got = provider.logsumexp(logits)
        if got is not None:
            return got
    return None


def diag_logits(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
                weights: np.ndarray | None = None) -> np.ndarray:
    """``(n, k)`` ``log N(x | mu_j, diag cov_j)``, log-weights added if given.

    Precedence is fixed: an internal provider when present, then the numba
    backend when it was asked for and the batch clears its work/width gate,
    then the C artifact.  The per-shape crossover lives in the record: internal
    wins the narrow row, C wins wide rows, and numba's wide rows use two BLAS
    GEMMs.  ``component_log_prob`` falls back to its BLAS expansion when no arm
    is available, which a bare call here cannot do.
    """
    means = np.asarray(means)
    lib = _load()
    provider = _internal() if means.ndim == 2 else None
    backend = _numba() if means.ndim == 2 else None
    work = 0
    if np.ndim(x) == 2 and means.ndim == 2:
        work = int(np.shape(x)[0]) * int(means.shape[0]) * int(means.shape[1])
    if provider is not None:
        got = provider.logits(x, means, covs, weights, work, lib is not None)
        if got is not None:
            return got
    if backend is not None:
        got = backend.logits(x, means, covs, weights, work)
        if got is not None:
            return got
    if lib is not None:
        return _c_logits(lib, x, means, covs, weights)
    raise RuntimeError(f"compiled host kernel unavailable: {_error}")


def diag_estep(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
               weights: np.ndarray | None = None, *,
               want_resp: bool = False) -> dict[str, Any]:
    """One fused E-step: statistics + objective, one pass over ``x``.

    Returns ``{'counts', 'sums', 'sq', 'nll'}`` -- and ``'resp'`` when
    ``want_resp`` -- exactly the quantities ``statistics_from`` produces.
    """
    lib = _load()
    if lib is None:
        raise RuntimeError(f"compiled host kernel unavailable: {_error}")
    x, invT, miT, bias, off = _prepare(x, means, covs, weights)
    n, d = x.shape
    k = means.shape[0]
    counts = np.empty(k, dtype=np.float64)
    sums = np.empty((k, d), dtype=np.float64)
    sq = np.empty((k, d), dtype=np.float64)
    resp = np.empty((n, k), dtype=np.float64) if want_resp else None
    log_norm = np.empty(n, dtype=np.float64)
    nll_sum = np.zeros(1, dtype=np.float64)
    # The three trailing buffers are optional in C, so they are passed as raw
    # addresses: ``ndpointer`` refuses ``None`` and ``c_void_p`` refuses arrays.
    status_code = lib.egmm_diag_estep(
        x, invT, miT, bias, off,
        n, d, k, _threads(n * d * k),
        counts, sums, sq,
        None if resp is None else resp.ctypes.data,
        log_norm.ctypes.data, nll_sum.ctypes.data)
    if status_code != 0:
        raise RuntimeError(f"egmm_diag_estep failed (status {status_code})")
    out: dict[str, Any] = {
        "counts": counts,
        "sums": sums.reshape(k, d),
        "sq": sq.reshape(k, d),
        # the C side accumulates sum(-log_norm) already: no second negation
        "nll": float(nll_sum[0]) / max(n, 1),
    }
    if resp is not None:
        out["resp"] = resp
    return out
