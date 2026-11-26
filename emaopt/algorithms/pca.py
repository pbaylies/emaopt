"""PCA: streaming covariance and the Oja control.

PCA is the PLAN's Tier-3 control.  Covariance PCA with an EMA covariance is
almost *too* easy -- online PCA is well established, so the interesting question
is not "can it be done" but "does the e-means-style state update beat the
established streaming method".  ``OjaPCA`` is that established method here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import (
    DataSpec,
    FitConfig,
    FitResult,
    State,
    StatisticalOptimizer,
    fit,
)
from ..state import EMA, Replace
from ..state.sufficient import GramianStat

_EPS = 1e-12


class PCACovariance(StatisticalOptimizer):
    """Persistent second moment ``C = E[x x^T]``; eigenvectors decode components."""

    name = "pca_covariance"
    state_keys = ("C",)
    supports_streaming = True

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        return int(d * d * 8 + d * 8)

    def stat(self, *, mode: str = "sum", alpha: Any = 1.0, center: bool = True) -> GramianStat:
        """The same covariance as a mergeable, blockable sufficient statistic."""
        return GramianStat(
            n_components=self.n_components, center=center, mode=mode, alpha=alpha
        )

    def __init__(self, n_components: int | None = None, *, updater: Any = None,
                 sample_size: int = 4096, seed: int = 0) -> None:
        super().__init__(updater or Replace())
        self.n_components = n_components
        self.sample_size = int(sample_size)
        self.seed = int(seed)

    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        n, d = data.shape
        if (
            config is not None
            and isinstance(self.updater, Replace)
            and config.batch_size in (None, "full")
            and int(config.nepochs) == 1
        ):
            # One exact full-batch Replace fit replaces this state with the
            # full-data second moment before anything decodes it, so the sample
            # covariance (4096 x d Gramian plus the row sampling, ~3 ms at
            # d=128) is work that is thrown away.
            return State({"C": np.zeros((d, d))}, 0, {"n_features": d})
        rng = np.random.default_rng(self.seed)
        sample = data
        if n > self.sample_size:
            sample = data[rng.choice(n, size=self.sample_size, replace=False)]
        cov = (sample.T @ sample) / max(sample.shape[0], 1)
        return State({"C": cov}, 0, {"n_features": d})

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        return State({"C": (batch.T @ batch) / max(batch.shape[0], 1)}, 0, {})

    def decode(self, state: State) -> dict[str, Any]:
        cov = np.asarray(state["C"], dtype=np.float64)
        d = cov.shape[0]
        k = d if self.n_components is None else int(min(self.n_components, d))
        evals, evecs = np.linalg.eigh(0.5 * (cov + cov.T))
        order = np.argsort(evals)[::-1][:k]
        total = max(float(evals.sum()), _EPS)
        mean = np.asarray(state.meta.get("mean", np.zeros(d)), dtype=np.float64)
        return {
            "components": evecs[:, order],
            "eigenvalues": np.maximum(evals[order], 0.0),
            "explained_variance_ratio": np.maximum(evals[order], 0.0) / total,
            "mean": mean,
        }

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared reconstruction error of the retained subspace.

        Computed chunk by chunk from ``||c||^2 - 2 p.p + p^T (W^T W) p`` with
        ``p = c W``, the algebraic expansion of ``||c - c W W^T||^2``.  The
        block loop never materialises an ``n x d`` centred copy or residual, and
        skips the centring subtract entirely when the mean is zero (a streaming
        second moment carries none).  Measured at 50k x 128, k=4: 3.9 ms against
        27.1 ms for the whole-array form (13.7 ms with a non-zero mean).  The
        ``W^T W`` term is what keeps it exact if a caller passes components
        that are not orthonormal; for :meth:`decode`'s eigenvectors it is the
        identity and costs ``k^2 d`` flops.  Clipping at zero keeps round-off
        from reporting a negative squared error when the subspace is exact.

        Reconstruction is about the stored mean when one is present, which is
        what makes :meth:`from_data` agree with a centred PCA; a streaming
        second moment carries no mean and reconstructs about the origin exactly
        as before.
        """
        data = np.asarray(data, dtype=np.float64)
        comps = np.asarray(parameters["components"], dtype=np.float64)
        mean = np.asarray(parameters.get("mean", np.zeros(data.shape[1])), dtype=np.float64)
        gram = comps.T @ comps
        shift = bool(np.any(mean))
        chunk = 8192
        total = 0.0
        for start in range(0, data.shape[0], chunk):
            block = data[start : start + chunk]
            c = block - mean if shift else block
            p = c @ comps
            total += float(
                np.einsum("ij,ij->i", c, c).sum()
                - 2.0 * np.einsum("ij,ij->i", p, p).sum()
                + np.einsum("ij,ij->i", p @ gram, p).sum()
            )
        return float(max(total, 0.0) / max(data.size, 1))

    @classmethod
    def from_data(
        cls, data: np.ndarray, n_components: int | None = None, *, chunk: int | None = None
    ) -> tuple["PCACovariance", dict[str, Any]]:
        """Centred PCA in closed form, in bounded memory.

        The mean is taken first and each row block is centred before its Gramian
        is accumulated.  The algebraically equivalent ``E[x x^T] - mu mu^T``
        loses the variance of offset data to cancellation: at a mean of 1e8 the
        subtraction is between numbers of size 1e16 and a unit variance falls
        below their spacing.  The extra mean pass is one scalar reduction per
        block and keeps the identity without the cancellation.

        ``chunk=None`` picks a cache-friendly block: at most 4096 rows and about
        16 MB.  The old fixed 65536-row default made a 33.6 MB centred block at
        d=64 and a 67 MB one at d=128; that overflowed cache.  In the final run
        it was 1.7x slower at d=64, 1.5x at d=128, and 1.3x at d=512 than the
        adaptive block.  A caller can still pass an explicit row count when its
        memory/IO trade-off differs.
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        n, d = data.shape
        # Accumulate in float64 even for a float16/float32 memmap; mean() would
        # otherwise keep a low-precision accumulator for the whole data pass.
        mean = np.asarray(data.mean(axis=0, dtype=np.float64), dtype=np.float64)
        if chunk is None:
            chunk = min(4096, max(1, 16_000_000 // (8 * max(int(d), 1))))
        else:
            chunk = max(1, int(chunk))
        stat = GramianStat(n_components=n_components, center=False)
        for start in range(0, n, chunk):
            block = np.asarray(data[start : start + chunk], dtype=np.float64)
            stat.update(block - mean)
        model = cls(n_components)
        params = stat.decode()
        params["mean"] = mean
        return model, params

    @classmethod
    def from_blocks(
        cls,
        blocks: Any,
        n_components: int | None = None,
        *,
        center: bool = True,
        mean: np.ndarray | None = None,
    ) -> tuple["PCACovariance", dict[str, Any]]:
        """Blocked PCA over any iterable of row blocks (no full-data copy).

        This is the ONLINE_PLAN phase-3 port: the reduction is
        :class:`~emaopt.state.sufficient.GramianStat`, so the same code path
        also merges across workers and streams a memmap that never fits.

        Passing ``mean=`` centres every block before accumulation, which is the
        stable form for offset data.  Without it a one-pass centered stat must
        use the raw moments and subtract ``mu mu^T`` at decode; that is exact in
        algebra but not in floating point once the mean is far from zero.
        """
        if mean is not None:
            mean = np.asarray(mean, dtype=np.float64).ravel()
            center = False
        stat = GramianStat(n_components=n_components, center=center)
        for block in blocks:
            b = np.asarray(block, dtype=np.float64)
            stat.update(b - mean if mean is not None else b)
        model = cls(n_components)
        params = stat.decode()
        if mean is not None:
            params["mean"] = mean
        return model, params

    @staticmethod
    def transform(data: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
        """Project ``data`` onto the retained components (centred)."""
        data = np.asarray(data, dtype=np.float64)
        mean = np.asarray(parameters.get("mean", np.zeros(data.shape[1])), dtype=np.float64)
        return (data - mean) @ parameters["components"]

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int = 1,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory)
        return fit(self, data, cfg, eval_data=eval_data)


class OjaPCA:
    """Oja's streaming rule for the leading eigenvector (established control)."""

    name = "oja"

    def __init__(self, lr: float = 0.05, *, niter: int = 5, seed: int = 0) -> None:
        self.lr = float(lr)
        self.niter = int(niter)
        self.seed = int(seed)

    def fit(self, data: np.ndarray) -> dict[str, Any]:
        data = np.asarray(data, dtype=np.float64)
        n, d = data.shape
        rng = np.random.default_rng(self.seed)
        w = rng.normal(size=d)
        w /= max(np.linalg.norm(w), _EPS)
        history: list[float] = []
        for _ in range(self.niter):
            order = rng.permutation(n)
            for i in order:
                x = data[i]
                proj = float(x @ w)
                w = w + self.lr * (proj * x - proj * proj * w)
                nrm = np.linalg.norm(w)
                if nrm > 0:
                    w = w / nrm
            cov = (data.T @ data) / n
            history.append(float(w @ cov @ w))
        cov = (data.T @ data) / n
        evals, evecs = np.linalg.eigh(cov)
        top = evecs[:, np.argmax(evals)]
        return {
            "component": w,
            "rayleigh": float(w @ cov @ w),
            "cosine_to_pc1": float(abs(w @ top)),
            "history": history,
        }

    def describe(self) -> dict[str, Any]:
        return {"optimizer": self.name, "lr": self.lr, "niter": self.niter}
