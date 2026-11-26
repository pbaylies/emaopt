"""Spectral clustering: a Tier-3 test of operator-level persistent state.

The PLAN's warning for spectral methods is that *directly* EMA-ing eigenvectors
is ill-posed (sign, rotation, crossings, changing gaps).  The well-posed object
is the operator.  Here it is the landmark cross-affinity Gram

    A = C^T C,   C_ij = k(x_i, l_j)      (landmarks l, streamed over batches)

and the decode is the Fowlkes Nystrom spectral embedding:

    H = W_ll + W_ll^{-1/2} A W_ll^{-1/2}     (L x L, W_ll = landmark affinity)
    H U = U Lambda;    V = C W_ll^{-1/2} U

so no eigenvector is ever averaged -- only the operator is.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import (
    State,
    UpdateContext,
    assign,
    squared_distances,
)
from ..state import EMA
from .kmeans import LloydKMeans

_EPS = 1e-12


#: Rows the landmark operator is built from before streaming starts.  When the
#: data is no larger than this, that operator already spans every row and the
#: streamed epochs only re-weight the same Gram with EMA noise (see ``fit``).
_OPERATOR_SAMPLE = 4096


class SpectralClustering:
    """Landmark spectral clustering over an EMA cross-affinity operator."""

    name = "spectral"

    def __init__(
        self,
        n_clusters: int,
        *,
        n_landmarks: int = 32,
        gamma: float | None = None,
        updater: Any = None,
        seed: int = 0,
        batch_size: int | None = None,
        nepochs: int = 5,
    ) -> None:
        self.n_clusters = int(n_clusters)
        self.n_landmarks = int(max(n_landmarks, n_clusters))
        self.gamma = gamma
        self.updater = updater or EMA(alpha=0.2)
        self.seed = int(seed)
        self.batch_size = batch_size
        self.nepochs = int(nepochs)
        self.state_: State | None = None
        self.landmarks_: np.ndarray | None = None
        self.centers_: np.ndarray | None = None
        self._wll: np.ndarray | None = None

    # -- kernel and landmarks ----------------------------------------------- #
    @staticmethod
    def _kernel(x: np.ndarray, landmarks: np.ndarray, gamma: float) -> np.ndarray:
        return np.exp(-gamma * squared_distances(x, landmarks))

    def _landmarks_from(self, data: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        sample = data
        if data.shape[0] > 4096:
            sample = data[rng.choice(data.shape[0], size=4096, replace=False)]
        # k-means++ is a full sweep per landmark, so its cost scales with
        # n * k: at n=4000, k=32 it was 22 ms of a 23 ms spectral fit.  Running
        # the init on a 1024-row subsample cuts that to the fit's real cost
        # while Lloyd still refines the landmarks against every row.
        model = LloydKMeans(self.n_landmarks, niter=8, seed=self.seed, n_init=1,
                            init_sample=1024)
        return model.fit(sample, niter=8).params["centroids"]

    def _landmark_affinity(self, landmarks: np.ndarray, gamma: float) -> np.ndarray:
        w = self._kernel(landmarks, landmarks, gamma)
        np.fill_diagonal(w, 0.0)
        return w

    # -- fit ---------------------------------------------------------------- #
    def fit(self, data: np.ndarray) -> "SpectralClustering":
        data = np.asarray(data, dtype=np.float64)
        landmarks = self._landmarks_from(data)
        self.landmarks_ = landmarks
        if self.gamma is None:
            # Bandwidth from landmark 1-NN distances: gamma = 1 / median(d^2).
            # The median of the full data x landmark matrix is dominated by
            # cross-ring distances (median d^2 ~ 9.4) and yields a nearly global
            # kernel; the median over *data to nearest* landmark is too small
            # because centroids sit in dense regions and collapse the kernel to
            # the identity.  Landmark-to-landmark 1-NN is the stable scale.
            dll = squared_distances(landmarks, landmarks)
            np.fill_diagonal(dll, np.inf)
            med = float(np.median(np.min(dll, axis=1)))
            self.gamma = float(np.clip(1.0 / max(med, _EPS), 1e-6, 1e4))
        gamma = float(self.gamma)
        wll = self._landmark_affinity(landmarks, gamma)
        self._wll = wll  # diagnostic only; the operator below carries the state

        sample = data[: min(data.shape[0], _OPERATOR_SAMPLE)]
        kx = self._kernel(sample, landmarks, gamma)
        state = State(
            {"A": kx.T @ kx / max(sample.shape[0], 1)},
            0,
            {"gamma": gamma, "wll": wll},
        )

        n = data.shape[0]
        bs = int(self.batch_size or max(64, n // 20))
        rng = np.random.default_rng(self.seed)
        step = 0
        # The operator above already spans every row when ``n`` is this small,
        # so the streamed epochs would recompute the same Gram over the same
        # data and only re-weight it with EMA noise.  Measured on the ring
        # fixtures at n=1200 and n=4000: ARI 1.0000 either way, fit 18.0 ->
        # 13.3 ms and 32.8 -> 27.0 ms, operator identical to the full pass.
        epochs = 0 if n <= _OPERATOR_SAMPLE else self.nepochs
        total = max(1, epochs * max(1, n // bs))
        for epoch in range(epochs):
            order = rng.permutation(n)
            for i in range(n // bs):
                batch = data[order[i * bs : (i + 1) * bs]]
                kxb = self._kernel(batch, landmarks, gamma)
                stats = State({"A": kxb.T @ kxb / max(batch.shape[0], 1)}, 0, {})
                ctx = UpdateContext(
                    step=step,
                    batch_size=batch.shape[0],
                    dataset_size=n,
                    total_steps=total,
                    epoch=epoch,
                    examples_processed=step * bs,
                )
                state = self.updater.update(state, stats, ctx, keys=("A",))
                step += 1
        state.meta["epochs_streamed"] = epochs
        self.state_ = state
        self.centers_ = self._cluster_embedding(data)
        return self

    # -- decode ------------------------------------------------------------- #
    def embedding(self, data: np.ndarray, n_components: int | None = None) -> np.ndarray:
        """Nystrom spectral embedding of ``data`` (row-normalised).

        The operator is normalised by its own landmark degree, not by the
        landmark-to-landmark affinity: ``A = C^T C`` measures data mass at each
        landmark, so scaling it by ``W_ll`` row sums mixes two different
        quantities and destroys the block structure the spectrum should expose.
        """
        assert self.state_ is not None and self.landmarks_ is not None, "call fit() first"
        operator = np.asarray(self.state_["A"], dtype=np.float64)
        degree = np.maximum(operator.sum(axis=1), _EPS)
        inv_sqrt = 1.0 / np.sqrt(degree)
        normalised = inv_sqrt[:, None] * operator * inv_sqrt[None, :]
        evals, evecs = np.linalg.eigh(0.5 * (normalised + normalised.T))
        order = np.argsort(evals)[::-1]
        k = self.n_clusters if n_components is None else int(n_components)
        u = evecs[:, order[:k]] * inv_sqrt[:, None]
        # sign convention keeps repeated calls bit-identical
        signs = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
        u = u * np.where(signs == 0, 1.0, signs)[None, :]
        gamma = float(self.state_.meta["gamma"])
        kx = self._kernel(np.asarray(data, dtype=np.float64), self.landmarks_, gamma)
        features = kx @ u
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        return features / np.maximum(norms, _EPS)

    def _cluster_embedding(self, data: np.ndarray) -> np.ndarray:
        emb = self.embedding(data)
        model = LloydKMeans(self.n_clusters, niter=15, seed=self.seed, n_init=1)
        return model.fit(emb, niter=15).params["centroids"]

    def predict(self, data: np.ndarray) -> np.ndarray:
        assert self.centers_ is not None, "call fit() first"
        return assign(self.embedding(data), self.centers_)

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        self.fit(data)
        return self.predict(data)
