"""Compiled kernels for the diagonal eGMM E-step.

Two kinds of ``.c`` have lived in this directory and they are not the same thing:

``egmm_diag.c``
    Shipped source.  The fused single-pass E-step (logits, softmax, counts,
    sums, sq and the NLL in one pass over the batch) plus
    ``egmm_logsumexp_rows``, the scoring pass that produces one output per row.
    It is packaged in the wheel, compiled per machine by ``python -m
    emaopt.kernels.build`` (default ``--backend c``: a C compiler is the only
    requirement) and loaded with ctypes by :mod:`emaopt.hostkernels`.

``numba.py``
    Optional, opt-in at run time with ``EMAOPT_NUMBA=1`` (``pip install
    "emaopt[numba]"``).  The row log-density, fused E-step and e-means epoch
    runner written for a JIT.  The row kernels use ``prange`` with one output
    per row, so threading cannot change the answer; the epoch runner is serial
    because its updates carry state.  It is the no-C-toolchain arm, and its
    work/width gates come from the record in ``evidence/host_kernels.json``.
    It never runs unless asked for.

Every kernel here is optional: without an artifact or an explicit opt-in the
NumPy path runs and the library behaves exactly as before.

Anything else that compiles these kernels lives *outside* this package (a
maintainer-side tree, reached through ``EMAOPT_INTERNAL_KERNELS``), so nothing
unreleased is named, imported or packaged here.  ``emaopt.kernels.internal()``
is the whole of that surface: a provider object, or ``None``.
"""

from __future__ import annotations

import os
import sys
from typing import Any

__all__ = ["internal"]

#: Resolved once per process: a provider module, or ``None``.
_INTERNAL: Any = False


def internal() -> Any:
    """The optional internal kernel provider, or ``None``.

    A release install never has one: ``EMAOPT_INTERNAL_KERNELS`` is unset, and
    the module it would name is not packaged.  This is how a maintainer's local
    build can supply arms that are deliberately absent from the wheel without
    the wheel naming them.

    Any failure -- no variable, a bad path, an import error, a provider that
    reports itself unavailable -- resolves to ``None`` and the NumPy/C path.
    """
    global _INTERNAL
    if _INTERNAL is not False:
        return _INTERNAL
    provider = None
    path = os.environ.get("EMAOPT_INTERNAL_KERNELS", "").strip()
    if path:
        try:
            if path not in sys.path:
                sys.path.insert(0, path)
            import importlib

            module = importlib.import_module("internal_kernels")
            provider = module if module.available() else None
        except Exception:               # pragma: no cover - environment
            provider = None
    _INTERNAL = provider
    return provider
