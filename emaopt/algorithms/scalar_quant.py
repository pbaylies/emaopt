"""Scalar quantization, optionally behind an orthogonal rotation.

A scalar quantizer codes each coordinate independently, so its distortion is a
function of each coordinate's *marginal* distribution.  With a uniform min-max
quantizer two things hurt:

* a heavy-tailed coordinate wastes levels on an empty tail, because the range is
  set by the extremes;
* a rotation is the only way to change the marginals, and it can only help
  coordinates that are coded independently -- it is an exact no-op for distances
  and for vector codebooks (which is why ``ProductQuantizer`` does not use one).

The measured ordering on 4000x16 t3 data at 3 bits (relative MSE) is
``hadamard 0.337 < learned 0.545 < random 0.575 < none 1.406``.  Hadamard wins
because a Hadamard rotation minimises the largest coordinate magnitude -- the
minimax spreading that a min-max quantizer wants -- while the learned
Procrustes rotation minimises *squared* error against a reconstruction that the
tails dominate, so it aligns with the outliers instead of taming them.  On
Gaussian data nothing changes (0.0907 without, 0.0898 with), which is the
control: rotation is buying tail-spreading, not compression.

``min_range`` is the same objective solved instead of assumed: it minimises
``sum_c (max_c - min_c)^2`` over the rotation group with a greedy Givens search.
Measured at 3 bits (relative MSE): on 4000x16 isotropic t3 it lands at 0.3488
against Hadamard's 0.3371 -- the fixed basis is already a good local optimum and
the greedy search does not improve on it -- while on data with a variance
hierarchy it reaches 0.2483 against 0.3605 for Hadamard and 0.3586 for no
rotation.  It does not pad, so it costs ``d * n_bits`` rather than
``next_power_of_two(d) * n_bits``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_EPS = 1e-12


def hadamard_matrix(d: int) -> np.ndarray:
    """Normalised Sylvester Hadamard matrix (orthogonal, entries ``+-1/sqrt(d)``)."""
    if d <= 0 or (d & (d - 1)):
        raise ValueError(f"hadamard_matrix needs a power-of-two dimension, got {d}")
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


class ScalarQuantizer:
    """Uniform per-coordinate quantizer with an optional orthogonal preconditioner.

    ``rotation``:

    * ``"hadamard"`` -- fixed orthogonal basis (default); a non-power-of-two
      dimension is zero-padded to the next power of two, so ``bits_per_vector``
      is ``next_power_of_two(d) * n_bits``,
    * ``"none"`` -- quantise the raw coordinates (the baseline),
    * ``"random"`` -- one seeded orthogonal draw,
    * ``"learned"`` -- alternate quantisation with the orthogonal Procrustes
      solve against the reconstruction (OPQ-style).  Beats no rotation, loses to
      Hadamard on heavy tails; see the module docstring for why.
    * ``"min_range"`` -- greedy Givens search minimising the summed squared
      coordinate ranges, the objective a min-max quantiser actually has.  No
      padding; costs ``d * n_bits`` bits.
    * ``"auto"`` -- Hadamard unless the feature-scale spread is above 2.0 and
      ``d <= 32``, where the measured variance-hierarchy win is large enough to
      pay for the greedy search.  Above 32 dimensions the search's ``O(d^2)``
      cost dominates and Hadamard is the only sane default.

    ``encode`` returns integer codes of shape ``(n, p)`` where ``p`` is the
    (possibly padded) rotated dimension; ``decode`` maps them back to ``(n, d)``.
    """

    name = "scalar_quantizer"
    ROTATIONS = ("hadamard", "none", "random", "learned", "min_range", "auto")
    #: ``auto``: above this dimension the Givens search costs more than it buys.
    AUTO_MIN_RANGE_MAX_D = 32
    #: ``auto``: max/median feature std above which min_range is preferred.
    AUTO_MIN_RANGE_SPREAD = 2.0

    def __init__(self, n_bits: int = 4, *, rotation: str = "hadamard",
                 n_iter: int = 15, seed: int = 0) -> None:
        if rotation not in self.ROTATIONS:
            raise ValueError(f"rotation must be one of {self.ROTATIONS}, got {rotation!r}")
        if int(n_bits) < 1:
            raise ValueError("n_bits must be >= 1")
        self.n_bits = int(n_bits)
        self.rotation = rotation
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.d_ = 0
        self.p_ = 0
        self.rotation_ = np.ones((1, 1))
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    # -- helpers ------------------------------------------------------------ #
    @property
    def levels(self) -> int:
        return 2 ** self.n_bits

    @staticmethod
    def _principal_scale_spread(data: np.ndarray, *, sample: int = 4096) -> float:
        """Largest principal-axis std over the median, a rotation-invariant
        anisotropy score.

        Raw column scales miss a variance hierarchy that has been rotated (the
        common case for embeddings): every column then mixes the dominant and
        weak directions.  The covariance eigenvalues are the variances along
        the actual axes, so their square roots give the scale hierarchy the
        min-range search can exploit.  Only called at d <= 32, where the
        eigendecomposition is negligible.
        """
        data = np.asarray(data, dtype=np.float64)
        if data.shape[0] > sample:
            data = data[np.random.default_rng(0).choice(
                data.shape[0], size=sample, replace=False)]
        # A constant offset is not a scale hierarchy.  Center before the second
        # moment, or a large common mean can mask (or manufacture) the spread
        # that `auto` is supposed to detect.
        data = data - data.mean(axis=0, keepdims=True)
        cov = data.T @ data / max(data.shape[0], 1)
        evals = np.maximum(np.linalg.eigvalsh(0.5 * (cov + cov.T)), 0.0)
        scale = np.sqrt(evals)
        median = float(np.median(scale))
        if median <= 0.0:
            return 1.0 if float(scale.max()) <= 0.0 else float("inf")
        return float(scale.max() / median)

    def _resolve_rotation(self, data: np.ndarray) -> str:
        """Resolve ``auto`` from the measured dimension and scale spread."""
        if self.rotation != "auto":
            return self.rotation
        if data.shape[1] > self.AUTO_MIN_RANGE_MAX_D:
            return "hadamard"
        spread = self._principal_scale_spread(data)
        return "min_range" if spread > self.AUTO_MIN_RANGE_SPREAD else "hadamard"

    def _initial_rotation(self, d: int, rotation: str | None = None) -> tuple[np.ndarray, int]:
        """Rotation of shape (p, d) with orthonormal columns, plus p."""
        rotation = self.rotation if rotation is None else rotation
        if rotation == "none":
            return np.eye(d), d
        if rotation == "hadamard":
            size = 1 << int(np.ceil(np.log2(max(d, 1))))
            return hadamard_matrix(size)[:, :d], size
        if rotation == "min_range":
            return np.eye(d), d
        rng = np.random.default_rng(self.seed)
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        return q, d

    def _minimise_ranges(self, data: np.ndarray, *, sweeps: int = 5,
                         angles: int = 7, step: float = np.pi / 8) -> None:
        """Greedy Givens search on ``sum_c (max_c - min_c)**2``.

        A uniform min-max quantiser's error is driven by each coordinate's span,
        so this is the rotation's real objective (the Hadamard basis is a fixed
        point of it on isotropic data, not a solution to it).  Each accepted move
        rotates one pair of coordinates by the best angle on a small grid; the
        span bookkeeping makes an accepted move O(d) rather than O(n*d).
        """
        rotated = data @ self.rotation_.T
        span = rotated.max(axis=0) - rotated.min(axis=0)
        base = float(np.sum(span**2))
        grid = np.linspace(-np.pi / 4, np.pi / 4, angles)
        d = rotated.shape[1]
        for _ in range(max(1, sweeps)):
            improved = False
            for i in range(d):
                for j in range(i + 1, d):
                    column_i = rotated[:, i]
                    column_j = rotated[:, j]
                    best = None
                    for theta in grid:
                        if abs(theta) <= 1e-15:
                            continue
                        cos, sin = float(np.cos(theta)), float(np.sin(theta))
                        candidate_i = cos * column_i - sin * column_j
                        candidate_j = sin * column_i + cos * column_j
                        span_i = float(candidate_i.max() - candidate_i.min())
                        span_j = float(candidate_j.max() - candidate_j.min())
                        score = (base - span[i] ** 2 - span[j] ** 2
                                 + span_i**2 + span_j**2)
                        if best is None or score < best[0] - 1e-12:
                            best = (score, cos, sin, candidate_i, candidate_j,
                                    span_i, span_j)
                    if best is not None and best[0] < base - 1e-12:
                        base, cos, sin, candidate_i, candidate_j, span_i, span_j = best
                        rotated[:, i] = candidate_i
                        rotated[:, j] = candidate_j
                        span[i], span[j] = span_i, span_j
                        row_i = self.rotation_[i].copy()
                        row_j = self.rotation_[j].copy()
                        self.rotation_[i] = cos * row_i - sin * row_j
                        self.rotation_[j] = sin * row_i + cos * row_j
                        improved = True
            if not improved:
                break
        self._set_levels(rotated)

    def _set_levels(self, rotated: np.ndarray) -> None:
        lo = rotated.min(axis=0)
        hi = rotated.max(axis=0)
        # a constant coordinate still needs a non-zero span to decode
        degenerate = (hi - lo) <= _EPS
        self.lo_ = lo
        self.hi_ = np.where(degenerate, lo + 1.0, hi)

    def _codes(self, rotated: np.ndarray) -> np.ndarray:
        assert self.lo_ is not None and self.hi_ is not None, "call fit() first"
        span = np.maximum(self.hi_ - self.lo_, _EPS)
        scaled = (rotated - self.lo_) / span * (self.levels - 1)
        return np.clip(np.round(scaled), 0, self.levels - 1).astype(np.int32)

    def _reconstruct(self, codes: np.ndarray) -> np.ndarray:
        assert self.lo_ is not None and self.hi_ is not None, "call fit() first"
        return self.lo_ + codes.astype(np.float64) / (self.levels - 1) * (self.hi_ - self.lo_)

    # -- lifecycle ---------------------------------------------------------- #
    def fit(self, data: np.ndarray) -> "ScalarQuantizer":
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2:
            raise ValueError("scalar quantisation expects a 2-D array")
        _, d = data.shape
        self.d_ = d
        resolution = self._resolve_rotation(data)
        self._resolved_rotation = resolution
        self.rotation_, self.p_ = self._initial_rotation(d, resolution)
        self._set_levels(data @ self.rotation_.T)
        if resolution == "min_range":
            self._minimise_ranges(data)
            return self
        if resolution != "learned":
            return self
        best = self.quantization_error(data)
        for _ in range(max(1, self.n_iter)):
            target = self._reconstruct(self._codes(data @ self.rotation_.T))
            # orthogonal Procrustes: maximise tr(R X^T target) over orthogonal R
            u, _, vt = np.linalg.svd(data.T @ target)
            self.rotation_ = vt.T @ u.T
            self.p_ = d
            self._set_levels(data @ self.rotation_.T)
            error = self.quantization_error(data)
            if error >= best - 1e-15:
                break
            best = error
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] != self.d_:
            raise ValueError(f"expected {self.d_} features, got {data.shape[1]}")
        return self._codes(data @ self.rotation_.T)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=int)
        if codes.shape[1] != self.p_:
            raise ValueError(f"expected {self.p_} codes per vector, got {codes.shape[1]}")
        return self._reconstruct(codes) @ self.rotation_

    # -- quality ------------------------------------------------------------ #
    def quantization_error(self, data: np.ndarray) -> float:
        """Mean squared error per row in the original space (absolute)."""
        data = np.asarray(data, dtype=np.float64)
        residual = data - self.decode(self.encode(data))
        return float(np.mean(np.sum(residual**2, axis=1)))

    def relative_error(self, data: np.ndarray) -> float:
        """Quantisation error divided by the mean row energy (scale-free)."""
        data = np.asarray(data, dtype=np.float64)
        energy = float(np.mean(np.sum(data**2, axis=1)))
        return self.quantization_error(data) / max(energy, _EPS)

    def bits_per_vector(self) -> float:
        return float(self.p_ * self.n_bits)

    def describe(self) -> dict[str, Any]:
        return {"optimizer": self.name, "n_bits": self.n_bits,
                "rotation": self.rotation,
                "resolved_rotation": getattr(self, "_resolved_rotation", self.rotation),
                "d": self.d_, "padded": self.p_}
