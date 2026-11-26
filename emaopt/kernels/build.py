"""Build the optional compiled kernels from an installed package.

:mod:`emaopt.hostkernels` loads ``kernels/libegmm_diag.so`` when it exists and
silently falls back to NumPy when it does not.  The *source* ships in the wheel;
the *artifact* does not, because ``-march=native`` bakes the build host's ISA
into the instruction stream -- a shared object built on one machine and copied
to a weaker one raises SIGILL inside a fit, which no caller can catch (T35).

So the artifact is built per machine::

    python -m emaopt.kernels.build            # compile, stamp, verify (C, default)
    python -m emaopt.kernels.build --check    # verify an existing artifact

Backends, and who each one is for:

``c`` (default)
    ``kernels/egmm_diag.c`` -> ``kernels/libegmm_diag.so``.  Needs a C compiler
    and OpenMP, nothing else.  This is the only backend a release user ever
    needs, and it is what the install instructions describe.
``cuda``
    ``kernels/egmm_diag_device.cu`` -> ``kernels/libegmm_diag_cuda.so``.  Needs
    ``nvcc``; the artifact is opt-in at runtime (``EMAOPT_DEVICE_ESTEP=1``).

The build writes a ``*.build.json`` stamp beside the artifact recording the
required ISA flags, the source and artifact hashes and the command.  The loader
compares the stamp with this CPU and refuses (falling back to NumPy) rather than
crashing.  Behind a read-only ``site-packages``, build elsewhere and point the
loader at it::

    python -m emaopt.kernels.build --target ~/.cache/emaopt/libegmm_diag.so
    export EMAOPT_HOST_KERNEL_PATH=~/.cache/emaopt/libegmm_diag.so

The full accuracy *and* speed gate for the fused statistics pass lives in
``scripts/gen_hostkernels.py`` in the source tree; this module checks that the artifact
the loader will use agrees with NumPy on both entry points -- the fused
log-density and the row log-sum-exp -- because a stale artifact missing the
second one still serves the first, and would otherwise pass verification while
the scoring path stayed on NumPy.

Nothing here imports a maintainer tool, and no artifact but the shipped
``.c``/``.cu`` is packaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["build", "write_stamp", "verify", "main"]

#: The default backend: the C kernel, which needs only a C compiler.
DEFAULT_BACKEND = "c"

LIB_NAME = "libegmm_diag.so"
_KERNEL_DIR = Path(__file__).resolve().parent
SOURCE = _KERNEL_DIR / "egmm_diag.c"
CUDA_SOURCE = _KERNEL_DIR / "egmm_diag_device.cu"
CUDA_LIB_NAME = "libegmm_diag_cuda.so"


def _isa_flags() -> tuple[str, ...]:
    from ..hostkernels import _ISA_FLAGS

    return _ISA_FLAGS


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:                     # pragma: no cover - non-Linux
        return set()
    for line in text.splitlines():
        if line.startswith("flags"):
            return set(line.split(":", 1)[1].split())
    return set()                        # pragma: no cover - unusual kernel


def build(target: Path | None = None, *, rebuild: bool = True) -> dict:
    """Compile the kernel into ``target`` (default: the installed package)."""
    target = Path(target) if target is not None else _KERNEL_DIR / LIB_NAME
    if not SOURCE.exists():
        raise SystemExit(
            f"kernel source missing: {SOURCE}. A wheel built before the source "
            "was packaged cannot build the host kernel; reinstall emaopt>=0.1.0 "
            "from a build that ships emaopt/kernels/*.c")
    if not rebuild and target.exists():
        return {"command": "(existing build reused)", "seconds": 0.0,
                "bytes": target.stat().st_size}
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.environ.get("CC", "cc"), "-O3", "-march=native", "-fopenmp",
        "-fPIC", "-shared", "-fno-math-errno",
        str(SOURCE), "-o", str(target), "-lm",
    ]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(
            "build failed (run with --verbose for the command):\n"
            + " ".join(cmd) + "\n" + proc.stderr)
    return {
        "command": " ".join(cmd),
        "seconds": round(time.perf_counter() - started, 3),
        "bytes": target.stat().st_size,
        "stderr": proc.stderr.strip(),
    }


def write_stamp(build_info: dict, target: Path | None = None) -> dict:
    """Record what the artifact was compiled for, next to the artifact."""
    target = Path(target) if target is not None else _KERNEL_DIR / LIB_NAME
    stamp = {
        "artifact": target.name,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": platform.machine(),
        "march": "native",
        "required_flags": sorted(_cpu_flags() & set(_isa_flags())),
        "command": build_info.get("command", ""),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": int(build_info.get("bytes", 0)),
    }
    target.with_name(target.name + ".build.json").write_text(
        json.dumps(stamp, indent=2) + "\n")
    return stamp


def _numpy_logits(x, means, covs, weights) -> np.ndarray:
    inv = 1.0 / covs
    quad = (x * x) @ inv.T - 2.0 * (x @ (means * inv).T)
    quad = quad + (means * (means * inv)).sum(axis=1)[None, :]
    out = (-0.5 * quad - 0.5 * (np.log(covs).sum(axis=1)[None, :]
                                + means.shape[1] * np.log(2.0 * np.pi)))
    if weights is not None:
        out = out + np.log(np.maximum(weights, 1e-300))[None, :]
    return out


def verify(target: Path | None = None) -> dict:
    """Check the artifact the loader would use against NumPy on both entry points.

    A stale artifact is the failure this exists to catch: one built before
    ``egmm_logsumexp_rows`` was added loads and serves the E-step fine, so a
    log-density-only check passes while the scoring path silently stays NumPy.
    The row log-sum-exp is therefore reported separately and a missing symbol
    fails verification with the rebuild command.
    """
    import importlib

    if target is not None:
        os.environ["EMAOPT_HOST_KERNEL_PATH"] = str(target)
    import emaopt.hostkernels as hostkernels

    importlib.reload(hostkernels)
    status = hostkernels.status()
    if not status["available"]:
        return {"ok": False, "error": status["error"],
                "path": status["path"]}
    rng = np.random.default_rng(0)
    n, k, d = 128, 4, 16
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = 0.5 + rng.random((k, d))
    weights = rng.random(k)
    weights /= weights.sum()
    lib = hostkernels._load()
    if lib is None:
        return {"ok": False, "error": status["error"],
                "path": status["path"]}
    # Verify the C artifact itself, not an optional provider that may win
    # ``diag_logits`` precedence when the environment names one.
    got = hostkernels._c_logits(lib, x, means, covs, weights)
    ref = _numpy_logits(x, means, covs, weights)
    diff = float(np.max(np.abs(got - ref)))
    # Row log-sum-exp: present at all is half the check (see the docstring).
    logits = rng.normal(-3.0, 1.5, size=(777, 16))
    lse = hostkernels.logsumexp_rows(logits)
    if lse is None:
        return {"ok": False, "max_abs_diff": diff,
                "logsumexp": "missing (artifact predates egmm_logsumexp_rows; "
                              "rebuild with `python -m emaopt.kernels.build "
                              "--backend c`)",
                "path": status["path"], "stamp": status.get("stamp", {})}
    lse_ref = np.log(np.exp(logits - logits.max(axis=1, keepdims=True)).sum(axis=1))
    lse_ref += logits.max(axis=1)
    lse_diff = float(np.max(np.abs(lse - lse_ref)))
    return {"ok": diff <= 1e-9 and lse_diff <= 1e-12,
            "max_abs_diff": diff, "logsumexp_max_abs_diff": lse_diff,
            "path": status["path"], "stamp": status.get("stamp", {})}


# --------------------------------------------------------------------------- #
# cuda backend: the diagonal E-step on a GPU, no torch
# --------------------------------------------------------------------------- #
def _nvcc() -> str:
    """The nvcc to use: ``$NVCC``, else PATH, else the toolkit's usual home."""
    explicit = os.environ.get("NVCC", "").strip()
    if explicit:
        return explicit
    found = shutil.which("nvcc")
    if found:
        return found
    return "/usr/local/cuda/bin/nvcc"


def build_cuda(target: Path | None = None) -> dict:
    """Compile ``egmm_diag_device.cu`` into a ctypes-loadable shared object.

    ``-arch=native`` is deliberate in the same way ``-march=native`` is for the
    host kernel: the artifact is built per machine, and the loader's probe
    launches a real kernel rather than trusting the file, so an artifact copied
    to a different GPU is rejected (with a rebuild instruction) instead of
    failing inside a fit.
    """
    target = Path(target) if target is not None else _KERNEL_DIR / CUDA_LIB_NAME
    if not CUDA_SOURCE.exists():
        raise SystemExit(
            f"CUDA kernel source missing: {CUDA_SOURCE}. A wheel built before "
            "the source was packaged cannot build it; reinstall emaopt from a "
            "build that ships emaopt/kernels/*.cu")
    nvcc = _nvcc()
    if shutil.which(nvcc) is None and not Path(nvcc).exists():
        raise SystemExit(
            f"nvcc not found ({nvcc}); install the CUDA toolkit or set NVCC=")
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [nvcc, "-O3", "-arch=native", "-Xcompiler", "-fPIC", "-shared",
           str(CUDA_SOURCE), "-o", str(target), "-lm"]
    started = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit("CUDA build failed (run with --verbose for the "
                         "command):\n" + " ".join(cmd) + "\n" + proc.stderr)
    return {"command": " ".join(cmd),
            "seconds": round(time.perf_counter() - started, 3),
            "bytes": target.stat().st_size,
            "stderr": proc.stderr.strip()}


def write_cuda_stamp(build_info: dict, target: Path | None = None) -> dict:
    """Record what the CUDA artifact was built for, beside the artifact."""
    target = Path(target) if target is not None else _KERNEL_DIR / CUDA_LIB_NAME
    stamp = {
        "artifact": target.name,
        "backend": "cuda",
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": platform.machine(),
        "cuda_arch": "native",
        "command": build_info.get("command", ""),
        "source_sha256": hashlib.sha256(CUDA_SOURCE.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "bytes": int(build_info.get("bytes", 0)),
    }
    target.with_name(target.name + ".build.json").write_text(
        json.dumps(stamp, indent=2) + "\n")
    return stamp


def _numpy_estep(x: np.ndarray, means: np.ndarray, covs: np.ndarray,
                 weights: np.ndarray | None) -> dict:
    """The E-step the device kernel must reproduce, in plain NumPy."""
    inv = 1.0 / covs
    quad = (x * x) @ inv.T - 2.0 * (x @ (means * inv).T)
    quad += np.einsum("kd,kd->k", means, means * inv)[None, :]
    np.maximum(quad, 0.0, out=quad)
    off = -0.5 * (x.shape[1] * np.log(2.0 * np.pi) + np.log(covs).sum(axis=1))
    if weights is not None:
        off = off + np.log(np.maximum(weights, 1e-300))
    logits = -0.5 * quad + off[None, :]
    peak = logits.max(axis=1, keepdims=True)
    lse = peak[:, 0] + np.log(np.exp(logits - peak).sum(axis=1))
    resp = np.exp(logits - lse[:, None])
    return {"counts": resp.sum(axis=0), "sums": resp.T @ x,
            "sq": resp.T @ (x * x), "nll": float(-lse.mean())}


def verify_cuda(target: Path | None = None) -> dict:
    """Probe the artifact and check both precisions against NumPy.

    Thresholds follow the measured gap, not an aspiration: float32 counts/sums/
    sq agree to ~1.2e-5 and nll to ~4e-10 on the probe shape, so 1e-4 and 1e-6
    are gates that still catch a layout or accumulation bug (those are order 1).
    """
    import importlib

    if target is not None:
        os.environ["EMAOPT_DEVICE_KERNEL_PATH"] = str(target)
    import emaopt.devicekernels as devicekernels

    importlib.reload(devicekernels)
    status = devicekernels.status()
    if not status["available"]:
        return {"ok": False, "error": status["error"],
                "path": status["path"]}
    rng = np.random.default_rng(0)
    n, k, d = 257, 5, 3
    x = rng.normal(size=(n, d))
    means = rng.normal(size=(k, d))
    covs = 0.5 + rng.random((k, d))
    weights = rng.random(k)
    weights /= weights.sum()
    ref = _numpy_estep(x, means, covs, weights)
    errors: dict[str, dict[str, float]] = {}
    for precision in ("float32", "float64"):
        got = devicekernels.estep(x, means, covs, weights, precision=precision)
        diffs = {key: float(np.max(np.abs(got[key] - ref[key])
                                    / np.maximum(1.0, np.abs(ref[key]))))
                 for key in ("counts", "sums", "sq")}
        diffs["nll"] = abs(got["nll"] - ref["nll"]) / max(1.0, abs(ref["nll"]))
        errors[precision] = diffs
    f32, f64 = errors["float32"], errors["float64"]
    ok = (all(f32[key] <= 1e-4 for key in ("counts", "sums", "sq"))
          and f32["nll"] <= 1e-6
          and all(f64[key] <= 1e-10 for key in ("counts", "sums", "sq"))
          and f64["nll"] <= 1e-12)
    return {"ok": ok, "errors": errors, "path": status["path"],
            "device": status.get("name", ""),
            "compute_capability": f"{status.get('major')}.{status.get('minor')}",
            "sms": status.get("sms")}



def _main_c(args: argparse.Namespace) -> int:
    target = args.target
    if not args.check:
        try:
            info = build(target, rebuild=not args.no_build)
        except PermissionError:
            print(
                "cannot write into the installed package (read-only site-packages).\n"
                "Build elsewhere and point the loader at it:\n"
                "  python -m emaopt.kernels.build --target ~/.cache/emaopt/"
                f"{LIB_NAME}\n"
                f"  export EMAOPT_HOST_KERNEL_PATH=~/.cache/emaopt/{LIB_NAME}",
                file=sys.stderr)
            return 2
        stamp = write_stamp(info, target)
        print(f"built {stamp['artifact']} in {info['seconds']} s "
              f"({stamp['bytes']} bytes, flags: {', '.join(stamp['required_flags']) or 'none'})")
    result = verify(target)
    if result["ok"]:
        print(f"verified against NumPy: log-density max |diff| = "
              f"{result['max_abs_diff']:.2e}, row log-sum-exp = "
              f"{result['logsumexp_max_abs_diff']:.2e}")
        print(f"artifact: {result['path']}")
        return 0
    detail = (result.get("error") or result.get("logsumexp")
              or f"max |diff| {result.get('max_abs_diff')}")
    print(f"FAILED verification: {detail}", file=sys.stderr)
    return 1


def _main_cuda(args: argparse.Namespace) -> int:
    target = args.target
    if not args.check:
        try:
            info = build_cuda(target)
        except PermissionError:
            print(
                "cannot write into the installed package (read-only site-packages).\n"
                "Build elsewhere and point the loader at it:\n"
                "  python -m emaopt.kernels.build --backend cuda --target "
                f"~/.cache/emaopt/{CUDA_LIB_NAME}\n"
                "  export EMAOPT_DEVICE_KERNEL_PATH="
                f"~/.cache/emaopt/{CUDA_LIB_NAME}",
                file=sys.stderr)
            return 2
        stamp = write_cuda_stamp(info, target)
        print(f"built {stamp['artifact']} in {info['seconds']} s "
              f"({stamp['bytes']} bytes, cuda_arch native)")
    result = verify_cuda(target)
    if result["ok"]:
        f32 = result["errors"]["float32"]
        f64 = result["errors"]["float64"]
        print(f"device: {result['device']} (sm_{result['compute_capability']}, "
              f"{result['sms']} SMs)")
        print(f"verified against NumPy: float32 counts {f32['counts']:.1e} "
              f"nll {f32['nll']:.1e}; float64 counts {f64['counts']:.1e} "
              f"nll {f64['nll']:.1e}")
        print(f"artifact: {result['path']}")
        return 0
    detail = (result.get("error")
              or json.dumps(result.get("errors")))
    print(f"FAILED verification: {detail}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m emaopt.kernels.build",
        description="Build and verify emaopt's optional compiled kernels.")
    parser.add_argument("--backend", choices=("c", "cuda"),
                        default=DEFAULT_BACKEND,
                        help="c builds the fused diagonal E-step from "
                             "kernels/egmm_diag.c (default; needs only a C "
                             "compiler); cuda builds the torch-free device "
                             "E-step")
    parser.add_argument("--target", type=Path, default=None,
                        help="artifact path (default: the installed emaopt package)")
    parser.add_argument("--check", action="store_true",
                        help="verify an existing artifact instead of building")
    parser.add_argument("--no-build", action="store_true",
                        help="reuse an existing artifact at --target")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.backend == "cuda":
        return _main_cuda(args)
    return _main_c(args)


if __name__ == "__main__":              # pragma: no cover - CLI
    raise SystemExit(main())
