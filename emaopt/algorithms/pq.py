"""Product and residual quantization.

Product quantization splits the vector into ``m`` sub-vectors and learns an
independent codebook per subspace, so the effective codebook is the Cartesian
product (``ksub^m`` codewords for ``m * log2(ksub)`` bits).  Residual
quantization stacks stages, each coding the previous stage's residual.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import assign, squared_distances
from .kmeans import EMeans


class _SubspaceCodebookMixin:
    def _fit_codebook(self, x: np.ndarray, ksub: int, seed: int) -> dict[str, Any]:
        model = EMeans(ksub, nepochs=self.nepochs, batch_size=self.batch_size, seed=seed)
        return model.fit(x, nepochs=self.nepochs, seed=seed).params


class ProductQuantizer(_SubspaceCodebookMixin):
    """``d`` split into ``m`` subspaces of size ``d/m``, each with ``ksub`` codewords."""

    def __init__(
        self,
        d: int,
        m: int = 4,
        ksub: int = 16,
        *,
        nepochs: int = 5,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        if d % m != 0:
            raise ValueError(f"d={d} must be divisible by m={m}")
        if ksub < 1:
            raise ValueError("ksub must be >= 1")
        self.d = int(d)
        self.m = int(m)
        self.ksub = int(ksub)
        self.dsub = self.d // self.m
        self.nepochs = int(nepochs)
        self.batch_size = batch_size
        self.seed = int(seed)
        self.codebooks_: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "ProductQuantizer":
        data = np.asarray(data, dtype=np.float64)
        if data.shape[1] != self.d:
            raise ValueError(f"expected {self.d} features, got {data.shape[1]}")
        codebooks = np.zeros((self.m, self.ksub, self.dsub), dtype=np.float64)
        for j in range(self.m):
            block = data[:, j * self.dsub : (j + 1) * self.dsub]
            params = self._fit_codebook(block, self.ksub, seed=self.seed + j)
            codebooks[j] = params["centroids"]
        self.codebooks_ = codebooks
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        assert self.codebooks_ is not None, "call fit() first"
        data = np.asarray(data, dtype=np.float64)
        codes = np.empty((data.shape[0], self.m), dtype=np.int32)
        for j in range(self.m):
            block = data[:, j * self.dsub : (j + 1) * self.dsub]
            codes[:, j] = assign(block, self.codebooks_[j])
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        assert self.codebooks_ is not None, "call fit() first"
        codes = np.asarray(codes, dtype=int)
        out = np.empty((codes.shape[0], self.d), dtype=np.float64)
        for j in range(self.m):
            out[:, j * self.dsub : (j + 1) * self.dsub] = self.codebooks_[j][codes[:, j]]
        return out

    def quantization_error(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=np.float64)
        return float(np.mean(np.sum((data - self.decode(self.encode(data))) ** 2, axis=1)))

    def bits_per_vector(self) -> float:
        return float(self.m * np.ceil(np.log2(self.ksub)))


class ResidualQuantizer(_SubspaceCodebookMixin):
    """Stack of ``n_stages`` codebooks, each coding the previous residual."""

    def __init__(
        self,
        n_stages: int = 3,
        ksub: int = 16,
        *,
        nepochs: int = 5,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_stages = int(n_stages)
        self.ksub = int(ksub)
        self.nepochs = int(nepochs)
        self.batch_size = batch_size
        self.seed = int(seed)
        self.stages_: list[np.ndarray] = []

    def fit(self, data: np.ndarray) -> "ResidualQuantizer":
        data = np.asarray(data, dtype=np.float64)
        residual = data.copy()
        self.stages_ = []
        for s in range(self.n_stages):
            params = self._fit_codebook(residual, self.ksub, seed=self.seed + 17 * s)
            centroids = params["centroids"]
            self.stages_.append(centroids)
            residual = residual - centroids[assign(residual, centroids)]
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        residual = data.copy()
        codes = np.empty((data.shape[0], len(self.stages_)), dtype=np.int32)
        for s, centroids in enumerate(self.stages_):
            codes[:, s] = assign(residual, centroids)
            residual = residual - centroids[codes[:, s]]
        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        codes = np.asarray(codes, dtype=int)
        out = np.zeros((codes.shape[0], self.stages_[0].shape[1]), dtype=np.float64)
        for s, centroids in enumerate(self.stages_):
            out += centroids[codes[:, s]]
        return out

    def quantization_error(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=np.float64)
        return float(np.mean(np.sum((data - self.decode(self.encode(data))) ** 2, axis=1)))

    def bits_per_vector(self) -> float:
        return float(len(self.stages_) * np.ceil(np.log2(self.ksub)))
