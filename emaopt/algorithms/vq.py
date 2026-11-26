"""Vector quantization and hierarchical codebooks.

Codebook learning is k-means with a different name for the same statistics
(``N_k``, ``S_k``, ``c_k = S_k / N_k``), so the quantizer is a thin layer over
:class:`~emaopt.algorithms.kmeans.KMeans` that adds the measurements an index
builder actually cares about: quantization error, code usage, dead codewords and
the imbalance ratio.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import FitResult, State, assign, squared_distances
from .kmeans import EMeans, KMeans


class VectorQuantizer(KMeans):
    """A codebook trained by the k-means state update (EMA by default)."""

    name = "vq"

    def encode(self, data: np.ndarray, parameters: dict[str, Any] | None = None) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        centroids = (parameters or {}).get("centroids")
        if centroids is None:
            raise ValueError("encode needs parameters from decode() or a fitted state")
        return assign(data, centroids)

    def decode_codes(self, codes: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        return np.asarray(parameters["centroids"])[np.asarray(codes, dtype=int)]

    def quantization_error(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        data = np.asarray(data, dtype=np.float64)
        dist = squared_distances(data, parameters["centroids"])
        return float(dist.min(axis=1).mean())

    def code_usage(self, data: np.ndarray, parameters: dict[str, Any]) -> dict[str, float]:
        """Dead-codeword count and size imbalance (1.0 = perfectly balanced)."""
        codes = self.encode(data, parameters)
        counts = np.bincount(codes, minlength=self.k).astype(float)
        mean = counts.mean()
        return {
            "dead": float(np.sum(counts == 0)),
            "imbalance": float(counts.max() / max(mean, 1e-12)),
            "perplexity": float(np.exp(-np.sum((counts / counts.sum()) *
                                               np.log(counts / counts.sum() + 1e-300)))),
        }

    def bits_per_vector(self) -> float:
        return float(np.log2(self.k))


class HierarchicalCodebook:
    """Two-stage codebook: coarse codeword plus a residual codeword.

    Storage is ``ceil(log2(k_coarse)) + ceil(log2(k_fine))`` bits per vector for
    ``k_coarse * k_fine`` effective codewords, which is the point of a
    hierarchical quantizer.  Each stage is an independent e-means run, so the
    residual stage sees a distribution whose scale has already been reduced by
    the coarse stage.
    """

    def __init__(
        self,
        n_coarse: int = 16,
        n_fine: int = 16,
        *,
        nepochs: int = 5,
        batch_size: int | None = None,
        seed: int = 0,
    ) -> None:
        self.n_coarse = int(n_coarse)
        self.n_fine = int(n_fine)
        self.nepochs = int(nepochs)
        self.batch_size = batch_size
        self.seed = seed
        self.coarse_: dict[str, Any] | None = None
        self.fine_: dict[str, Any] | None = None

    def fit(self, data: np.ndarray) -> "HierarchicalCodebook":
        data = np.asarray(data, dtype=np.float64)
        coarse = EMeans(self.n_coarse, nepochs=self.nepochs, batch_size=self.batch_size,
                        seed=self.seed)
        coarse_fit = coarse.fit(data, nepochs=self.nepochs, seed=self.seed)
        self.coarse_ = coarse_fit.params
        residual = data - coarse_fit.params["centroids"][
            assign(data, coarse_fit.params["centroids"])
        ]
        fine = EMeans(self.n_fine, nepochs=self.nepochs, batch_size=self.batch_size,
                      seed=self.seed + 1)
        fine_fit = fine.fit(residual, nepochs=self.nepochs, seed=self.seed + 1)
        self.fine_ = fine_fit.params
        return self

    def encode(self, data: np.ndarray) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        assert self.coarse_ is not None and self.fine_ is not None, "call fit() first"
        coarse = assign(data, self.coarse_["centroids"])
        residual = data - self.coarse_["centroids"][coarse]
        fine = assign(residual, self.fine_["centroids"])
        return np.stack([coarse, fine], axis=1)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        assert self.coarse_ is not None and self.fine_ is not None, "call fit() first"
        codes = np.asarray(codes, dtype=int)
        return self.coarse_["centroids"][codes[:, 0]] + self.fine_["centroids"][codes[:, 1]]

    def quantization_error(self, data: np.ndarray) -> float:
        data = np.asarray(data, dtype=np.float64)
        return float(np.mean(np.sum((data - self.decode(self.encode(data))) ** 2, axis=1)))

    def bits_per_vector(self) -> float:
        return float(np.ceil(np.log2(self.n_coarse)) + np.ceil(np.log2(self.n_fine)))
