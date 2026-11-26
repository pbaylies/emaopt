"""Optional libmvec acceleration for the elementwise transcendental hot paths.

Evidence, measured on the development box: libmvec's
vector tanh beats scalar libm 7.69x (f64, 4-lane) and 14.51x (f32, 8-lane)
at machine-epsilon agreement with ``np.tanh``, which is the substrate the
FastICA E-step's 86%-of-iteration tanh was measured against
(README, "the E-step is compute-bound on tanh").

Design rules (telos: fail loudly where it matters, degrade where it does
not):

  * the shim is compiled once into the user cache and reused; any step
    (no compiler, non-Linux, old glibc, dlopen failure) leaves
    :func:`tanh` transparently on ``numpy``;
  * a freshly built shim is checked against ``np.tanh`` before it is ever
    trusted, and a mismatch disables it for the process — acceleration may
    never change results, only their wall time;
  * ``EMAOPT_DISABLE_LIBMVEC=1`` opts out entirely.

The vector entry points are called through C wrappers (``tanh4`` /
``tanh8f``): the libmvec functions take/return 4x-double / 8x-float vector
types ctypes cannot express, so the wrappers receive and return by pointer.
``probe_init`` resolves the IFUNC symbols; on a libmvec-less glibc it
returns nonzero and the shim stays inert.
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import numpy as np

_SRC = r"""
#include <dlfcn.h>
#include <stddef.h>
#include <string.h>

typedef double __v4f64 __attribute__((__vector_size__(32)));
typedef float  __v8f32 __attribute__((__vector_size__(32)));
typedef __v4f64 (*t4_t)(__v4f64);
typedef __v8f32 (*t8_t)(__v8f32);
static t4_t t4; static t8_t t8;
static t4_t c4; static t4_t l4;

int probe_init(void) {
    void *h = dlopen("libmvec.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libm.so.6", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return 1;
    t4 = (t4_t)dlsym(h, "_ZGVdN4v_tanh");
    t8 = (t8_t)dlsym(h, "_ZGVcN8v_tanhf");
    c4 = (t4_t)dlsym(h, "_ZGVdN4v_cosh");
    l4 = (t4_t)dlsym(h, "_ZGVdN4v_log");
    return (t4 && t8 && c4 && l4) ? 0 : 2;
}

/* log(cosh(x)) -- ICA's negentropy of one component.  Two vector
 * transcendentals per four lanes: measured 0.043 ms against NumPy's two
 * scalar passes at 0.167 ms for the 4000 x 4 history, agreeing to 8.9e-16. */
void logcosh4(const double *x, double *y, size_t n) {
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __v4f64 v; memcpy(&v, x + i, 32);
        v = l4(c4(v)); memcpy(y + i, &v, 32);
    }
    for (size_t i = n - n % 4; i < n; i++) {
        extern double cosh(double); extern double log(double);
        y[i] = log(cosh(x[i]));
    }
}

void tanh4(const double *x, double *y, size_t n) {
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __v4f64 v; memcpy(&v, x + i, 32);
        v = t4(v); memcpy(y + i, &v, 32);
    }
    for (size_t i = n - n % 4; i < n; i++) {
        extern double tanh(double); y[i] = tanh(x[i]);
    }
}

void tanh8f(const float *x, float *y, size_t n) {
    for (size_t i = 0; i + 8 <= n; i += 8) {
        __v8f32 v; memcpy(&v, x + i, 32);
        v = t8(v); memcpy(y + i, &v, 32);
    }
    for (size_t i = n - n % 8; i < n; i++) {
        extern float tanhf(float); y[i] = tanhf(x[i]);
    }
}
"""

_state: dict = {"lib": None, "tried": False}
#: Serializes the one-time shim build.  Without it two threads can both see
#: ``tried`` false, enter ``_build`` and race inside ``dlopen``; on this box
#: that deadlocked a 16-worker ICA fold in ``ctypes.CDLL``.
_build_lock = threading.Lock()


def _cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "emaopt"


def _build() -> ctypes.CDLL | None:
    tag = hashlib.sha256((_SRC + "|-O2|-mavx2|-lm").encode()).hexdigest()[:12]
    cache = _cache_dir() / f"libmvec_tanh_{tag}"
    so = cache / "tanh_shim.so"
    if not so.exists():
        if not Path("/proc/cpuinfo").exists() or "avx2" not in (
            Path("/proc/cpuinfo").read_text(errors="ignore").lower()
        ):
            # libmvec's vector entry points take YMM registers; on a
            # non-AVX2 CPU calling one is SIGILL, not a wrong answer, so the
            # probe must gate on the flag, not on runtime failure.
            return None
        cc = shutil.which(os.environ.get("CC", "cc")) or shutil.which("clang") \
            or shutil.which("gcc")
        if cc is None:
            return None
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", suffix=".c", delete=False) as handle:
            handle.write(_SRC)
            src = handle.name
        try:
            subprocess.run(
                [cc, "-O2", "-mavx2", "-fPIC", "-shared", src, "-o", str(so), "-lm"],
                check=True, capture_output=True, timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        finally:
            os.unlink(src)
    try:
        lib = ctypes.CDLL(str(so))
        if lib.probe_init() != 0:
            return None
    except OSError:
        return None
    # Trust but verify: the shim must reproduce np.tanh before it may
    # accelerate anything.
    probe = np.random.default_rng(0).standard_normal(4096)
    out = np.empty_like(probe)
    lib.tanh4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.tanh4(probe.ctypes.data, out.ctypes.data, probe.size)
    if not np.allclose(out, np.tanh(probe), rtol=0, atol=1e-12):
        return None
    # Same rule for the fused log-cosh: it may only accelerate if it agrees
    # with ``np.log(np.cosh(x))`` to machine epsilon.  It is *not* bit-identical
    # (libmvec and NumPy are different implementations: 8.9e-16 abs on this
    # box), which is why the callers that pin values exactly keep NumPy and
    # only the per-iteration diagnostics use this.
    lib.logcosh4.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.logcosh4.restype = None
    probe_out = np.empty_like(probe)
    lib.logcosh4(probe.ctypes.data, probe_out.ctypes.data, probe.size)
    if not np.allclose(probe_out, np.log(np.cosh(probe)), rtol=0, atol=1e-12):
        return None
    lib.tanh4.restype = None
    lib.tanh8f.restype = None
    lib.tanh8f.argtypes = lib.tanh4.argtypes
    return lib


def _lib() -> ctypes.CDLL | None:
    if _state["tried"]:
        return _state["lib"]
    with _build_lock:
        if not _state["tried"]:
            lib = None
            if os.environ.get("EMAOPT_DISABLE_LIBMVEC") != "1":
                lib = _build()
            _state["lib"] = lib
            _state["tried"] = True
    return _state["lib"]


def tanh(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """``np.tanh``-equivalent elementwise tanh, libmvec-accelerated when able."""
    lib = _lib()
    if out is None:
        out = np.empty_like(x)
    if lib is not None and x.dtype in (np.float64, np.float32) \
            and x.flags.c_contiguous and out.flags.c_contiguous:
        entry = lib.tanh4 if x.dtype == np.float64 else lib.tanh8f
        entry(x.ctypes.data, out.ctypes.data, x.size)
        return out
    np.tanh(x, out=out)
    return out


def logcosh(x: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """``np.log(np.cosh(x))`` elementwise, libmvec-accelerated when able.

    Agrees with NumPy to 8.9e-16 abs but is not bit-identical to it (different
    implementations); callers that need the NumPy bit pattern must not use this.
    """
    lib = _lib()
    if out is None:
        out = np.empty_like(x)
    if lib is not None and x.dtype == np.float64 \
            and x.flags.c_contiguous and out.flags.c_contiguous:
        lib.logcosh4(x.ctypes.data, out.ctypes.data, x.size)
        return out
    np.log(np.cosh(x, out=out), out=out)
    return out


def active() -> bool:
    """Whether the libmvec path is live in this process (for evidence rows)."""
    return _lib() is not None
