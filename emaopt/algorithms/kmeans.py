"""k-means: Lloyd's algorithm (Replace) and e-means (EMA), one implementation.

Both are the *same* optimizer.  They differ only in the state-update law and
the compute budget:

* Lloyd: ``batch = n``, ``Replace``  -> ``S_j <- s_j``, ``N_j <- n_j``.
* e-means: mini-batches, ``EMA``     -> ``S_j <- (1-a) S_j + a s_j``.

Setting ``batch_size = n`` and ``alpha = 1`` makes one e-means update exactly
one Lloyd iteration (the paper's stated special case, pinned by a test here).
e-means earns its advantage from everything else: the batch and step chosen by
the fitted scaling laws, cosine annealing of the step, proportional-split
revival of dead clusters, and a final exact-mean pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core import (
    DataSpec,
    label_sums,
    FitConfig,
    FitResult,
    State,
    StatisticalOptimizer,
    UpdateContext,
    assign,
    fit,
    squared_distances,
)
from ..state import EMA, Replace
from ..state.schedules import CosineSchedule

# Fitted scaling laws from the e-means paper (arXiv:2611.09501), as published on
# the project page.  B = c_B k^b_k n^b_n T^b_T and
# alpha0 = min(c_a k^a_k B^a_B n^a_n T^a_T, 1).
BATCH_LAW = {"default": (0.035, 0.13, 0.88, 0.81), "quality": (0.0044, 0.13, 0.88, 0.81)}
ALPHA_LAW = {"c": 0.824, "k": -0.46, "B": 0.90, "n": -0.45, "T": -0.11}

# Reference e-means initial cluster usage N_j (base.py _INIT_USAGE): the
# starting centroids are decoded exactly (sums = c * usage) but carry no
# pseudo-observation mass into the EMA.
_INIT_USAGE = 2.0 ** -16

# batch_law="auto" picks the quality law at or below this dimension and the
# throughput law above it (reference base._AUTO_QUALITY_MAX_D: the throughput
# batch makes ~8x fewer updates and costs 11-20 points of win rate at d <= 16).
AUTO_QUALITY_MAX_D = 16

#: Feature-scale spread above which ``init="auto"`` refuses the coreset seed.
#: A coreset is a uniform draw, so it represents a feature in proportion to the
#: variance that feature carries.  With all features at one scale that costs
#: nothing (every isotropic sweep cell measures 1.10-1.30); with per-feature
#: scales spanning ``e^2`` -- the aniso cell -- the coreset seed measured 181.6
#: held-out MSE against the full-data k-means++ seed's 175.6 (+3.4%), while the
#: rule's wins are all on evenly scaled data.  The threshold sits in the gap
#: (aniso 4.09, everything else <= 1.53) rather than on a fitted optimum, and
#: ``feature_scale_spread`` costs one sampled std.
AUTO_INIT_MAX_SCALE_SPREAD = 2.0

#: The compiled epoch runner is a maintainer-side provider
#: (``EMAOPT_INTERNAL_KERNELS``); a release install has none, so the per-batch
#: loop is the shipped path and the constants that gate the provider live with
#: it rather than here.
_INTERNAL_RUNNER: Any = False


def _internal_runner() -> Any:
    """The optional epoch runner, or ``None``.

    The maintainer-side provider comes first; when it is absent, the opt-in
    numba backend supplies the same flat-array contract.  Both are explicit
    opt-ins, so a release install still takes the Python loop.
    """
    global _INTERNAL_RUNNER
    if _INTERNAL_RUNNER is False:
        from ..kernels import internal

        provider = internal()
        runner = (provider.epoch_runner()
                  if provider is not None else None)
        if runner is None:
            from ..kernels import numba as backend

            runner = backend.epoch_runner()
        _INTERNAL_RUNNER = runner
    return _INTERNAL_RUNNER


def _round_power_of_two(value: float) -> int:
    """Round a positive value to the nearest power of two (reference recipe)."""
    if value <= 1:
        return 1
    low = 2 ** int(np.floor(np.log2(value)))
    high = 2 * low
    return low if value / low < high / value else high


def batch_size_from_law(
    n: int, k: int, nepochs: int = 5, *, batch_law: str = "default", d: int | None = None
) -> int:
    """Batch size from the published e-means scaling law, clipped to ``[1, n]``.

    Law-derived batches are rounded to the nearest power of two and capped at
    half the data, matching the reference release's ``_resolve_recipe``.
    ``batch_law="auto"`` follows the reference dimension rule: the
    ``"quality"`` law at ``d <= 16``, the throughput ``"default"`` law above it
    (``d`` unknown -> throughput, the pre-parity behavior).  A law named
    explicitly is used as given.
    """
    if batch_law in ("auto", None):
        key = "quality" if d is not None and d <= AUTO_QUALITY_MAX_D else "default"
    else:
        key = str(batch_law)
    try:
        c, b_k, b_n, b_t = BATCH_LAW[key]
    except KeyError as exc:
        raise KeyError(f"unknown batch law {batch_law!r}; available: {sorted(BATCH_LAW)}") from exc
    raw = c * (k ** b_k) * (n ** b_n) * (nepochs ** b_t)
    batch = _round_power_of_two(raw)
    # A floor was tried here (max(batch, 256)) and reverted: it made the 3k x 8
    # blob case faster and *better* (30.6 -> 4.3 ms, MSE ratio 1.0257 ->
    # 1.0017), but the alpha law is coupled to the batch size, and with the
    # maternal-init fixture (2000 x 8, k=16) the floored batch landed at 43.0
    # against the reference's 20.3 (0.94x before the floor).  The law is the
    # published recipe; changing one side of it is not a local edit.
    if key == "default":
        batch = min(batch, _round_power_of_two(n / 2))
    else:
        batch = min(batch, n // 2)
    return int(max(1, min(batch, n)))


def alpha0_from_law(n: int, k: int, batch_size: int, nepochs: int = 5) -> float:
    """Initial step from the published e-means scaling law, clipped to ``[1e-4, 1]``.

    The ``1e-4`` floor is the reference step-law clamp, not the paper's
    (its fitted exponent product can otherwise drift small on tiny batches).
    """
    raw = (
        ALPHA_LAW["c"]
        * (k ** ALPHA_LAW["k"])
        * (batch_size ** ALPHA_LAW["B"])
        * (n ** ALPHA_LAW["n"])
        * (nepochs ** ALPHA_LAW["T"])
    )
    return float(np.clip(raw, 1e-4, 1.0))


class KMeans(StatisticalOptimizer):
    """k-means as measurement -> state update -> decode -> objective.

    State: per-cluster vector sums ``S`` (k, d) and counts ``N`` (k,).
    Decode: ``c_j = S_j / N_j``.  With the ``Replace`` updater this is Lloyd;
    with any averaging updater it is a mini-batch k-means.
    """

    name = "kmeans"
    state_keys = ("sums", "counts")
    supports_streaming = True

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        k = self.k if k is None else int(k)
        extra = 2 if self.finalize else 1
        return int(k * d * 8 * extra + k * 8 * (extra + 1))

    def __init__(
        self,
        k: int,
        *,
        updater: Any = None,
        seed: int = 0,
        init: str = "kmeans++",
        init_centroids: np.ndarray | None = None,
        empty_cluster: str = "keep",
        revival: bool = False,
        revival_threshold: float = 0.1,
        finalize: bool = False,
        finalize_shrink: float = 0.0,
        init_sample: int | None = None,
        init_coreset: int | None = None,
        init_draws: int = 1,
        chunk: int = 4096,
        revival_geometry: str = "isotropic",
    ) -> None:
        super().__init__(updater or Replace())
        self.k = int(k)
        self.seed = int(seed)
        self.init = init
        #: Which branch ``init="auto"`` resolved to on the last initialization.
        self._resolved_init = init
        self.init_centroids = (
            None if init_centroids is None else np.asarray(init_centroids, dtype=np.float64)
        )
        #: Diagnostic for ``init="auto"``: the measured feature-scale spread.
        self._init_spread: float | None = None
        if int(init_draws) < 1:
            raise ValueError("init_draws must be >= 1")
        self.init_draws = int(init_draws)
        if self.init == "maternal" and self.init_centroids is None:
            raise ValueError("init='maternal' needs init_centroids (k, d)")
        self.empty_cluster = empty_cluster
        self.init_sample = init_sample
        #: Coreset size for ``init="coreset"`` (rows Lloyd clusters before the
        #: real fit grows them).  ``None`` uses ``max(4k, 512)``; the sweep put a
        #: 4x larger coreset ahead of k-means++ at 20k x 32 k=64 (34.96 against
        #: 37.06, 95 against 156 ms).
        self.init_coreset = init_coreset
        self.revival = bool(revival)
        self.revival_threshold = float(revival_threshold)
        self.finalize = bool(finalize)
        #: Empirical-Bayes shrinkage applied at finalize (s-means recipe):
        #: ``n/(n+tau)*final_mean + tau/(n+tau)*smoothed``.  0 keeps the
        #: reference e-means behavior of taking the final-pass mean outright.
        self.finalize_shrink = float(finalize_shrink)
        self.chunk = int(chunk)
        if revival_geometry not in ("isotropic", "anisotropic"):
            raise ValueError(
                f"unknown revival_geometry {revival_geometry!r}; "
                "available: isotropic, anisotropic"
            )
        self.revival_geometry = revival_geometry
        self._rng = np.random.default_rng(seed)

    # -- initialization ----------------------------------------------------- #
    def _init_centroids(self, data: np.ndarray) -> np.ndarray:
        n, d = data.shape
        k = self.k
        init = self.init
        if init == "auto":
            from .smeans import coreset_size, feature_scale_spread

            # Reset per call: a model reused on a second dataset must not report
            # the first dataset's spread.
            self._init_spread = None
            # Measured crossover (5 seeds):
            # k-means++ init costs O(n k d), so at 3000 x 8 it is 0.83 ms against
            # the coreset's 1.55 and scores 73.5 held-out MSE against 75.3;
            # at 20000 x 32 and 50000 x 16 the coreset is 4.7-11.3x cheaper to
            # build (2.4 ms against 25.6) with held-out MSE equal within noise
            # (413.2 vs 416.4 and 169.7 vs 169.3).  The rule below picks the
            # coreset only once the data is at least 8 coresets wide, which is
            # the measured side of that crossover; the exact multiple is a
            # plateau, not an optimum, and sits between the two shapes tested.
            # An explicit ``init_coreset`` moves the crossover with the coreset
            # the fit will actually cluster, so the threshold follows it.
            m = int(self.init_coreset or coreset_size(n, k))
            init = "coreset" if n >= 8 * m else "kmeans++"
            if init == "coreset":
                # Distortion guard: the rule above is a *cost* rule, and a
                # coreset draw is only a good seed while feature scales are
                # comparable.  On the anisotropic sweep cell (per-feature
                # scales spanning e^2) it measured 3.4% worse held out than
                # k-means++ on the full data, so the coreset is refused there
                # and the fit pays the k-means++ init it would have paid before.
                spread = feature_scale_spread(data)
                self._init_spread = spread
                if spread > AUTO_INIT_MAX_SCALE_SPREAD:
                    init = "kmeans++"
        self._resolved_init = init
        if init == "maternal":
            given = np.asarray(self.init_centroids, dtype=np.float64)
            if given.shape != (k, d):
                raise ValueError(
                    f"init_centroids shape {given.shape} != (k, d) {(k, d)}"
                )
            return given.copy()
        if init == "random":
            idx = self._rng.choice(n, size=k, replace=n < k)
            return data[idx].astype(np.float64, copy=True)
        if init in ("d1.5", "d15", "smeans"):
            # Multi-round subsampled D^1.5 seeding (s-means).
            from .smeans import d15_seed

            return d15_seed(data, k, self._rng)
        if init == "pddp":
            # Deterministic PCA-partition seed; opt-in because the exact
            # per-node direction costs O(n d^2) and k-means++ remains the
            # stronger default on easy isotropic shapes.  kM-PDDP steering
            # uses the exact one-dimensional two-means cut.
            from .pddp import pddp_seed

            return pddp_seed(data, k, seed=self.seed)
        if init == "coreset-guard":
            # The cheap seed, with the draw guarded: two candidates (one uniform,
            # one stratified) judged on a <4096-row slice they did not see.  This
            # is what removes the +52% tail the five-draw study found without
            # falling back to a full-data k-means++ init.
            from .smeans import guarded_coreset_seed

            # A private RNG: the guard's draws must not consume the stream the
            # fallback k-means++ would use, or the fallback is a *different*
            # k-means++ than the pinned arm and the comparison stops being fair.
            seed, acceptable = guarded_coreset_seed(
                data, k, np.random.default_rng(self.seed ^ 0x5EED),
                coreset=self.init_coreset,
                draws=max(2, int(self.init_draws)), check_rows=8192,
                verify=True,
            )
            if acceptable:
                self._resolved_init = init
                return seed
            # The guarded draw lost to a slice k-means++, so pay the full-data
            # k-means++ this path exists to avoid.  Rare, and it bounds the tail.
            init = "kmeans++"
            self._resolved_init = init
        if init in ("coreset", "coreset-lloyd"):
            # Lloyd on a coreset, then grown into the full set by this fit.
            # ``init_draws > 1`` draws several coresets and keeps the one whose
            # full-data inertia is lowest, because the draw -- not the size --
            # is what the size sweep found noisy.
            from .smeans import coreset_lloyd_seed, multi_coreset_seed

            if int(self.init_draws) > 1:
                return multi_coreset_seed(data, k, self._rng,
                                          coreset=self.init_coreset,
                                          draws=int(self.init_draws))
            return coreset_lloyd_seed(data, k, self._rng,
                                      coreset=self.init_coreset)
        if init not in ("kmeans++", "kmeanspp"):
            raise ValueError(f"unknown init {init!r}")
        # k-means++ on the full data by default.  A small fixed subsample looks
        # cheaper but can miss whole clusters; with no restart policy that leaves
        # a centroid parked on an empty region forever (measured: MSE 11.0 vs
        # sklearn's 7.9 on 16 blobs).
        sample = data
        if self.init_sample is not None and n > int(self.init_sample):
            sample = data[self._rng.choice(n, size=int(self.init_sample), replace=False)]
        m = sample.shape[0]
        centroids = np.empty((k, d), dtype=np.float64)
        centroids[0] = sample[self._rng.integers(m)]
        # row norms are constants of `sample`: compute once, reuse for all
        # k greedy sweeps (see squared_distances x_norm).
        sample_norm = np.einsum("ij,ij->i", sample, sample)
        closest = squared_distances(sample, centroids[:1],
                                    x_norm=sample_norm).ravel()
        # Greedy k-means++ (the variant sklearn and Faiss use): draw several
        # candidates per step and keep the one that most reduces the potential.
        # Plain D^2 sampling picks the same cluster twice often enough that a
        # 16-blob problem can end up with a missing centroid.
        trials = min(m, 2 + int(np.log(max(k, 2))))
        for j in range(1, k):
            total = float(closest.sum())
            if total <= 0.0 or not np.isfinite(total):
                centroids[j] = sample[self._rng.integers(m)]
                continue
            candidates = np.unique(self._rng.choice(m, size=trials, p=closest / total))
            d2 = squared_distances(sample, sample[candidates], x_norm=sample_norm)  # (m, trials)
            potentials = np.minimum(closest[:, None], d2).sum(axis=0)
            best = int(np.argmin(potentials))
            centroids[j] = sample[candidates[best]]
            closest = np.minimum(closest, d2[:, best])
        return centroids

    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        if isinstance(data_spec, DataSpec):
            raise TypeError("KMeans initialization needs the data matrix, not a DataSpec")
        data = np.asarray(data_spec, dtype=np.float64)
        centroids = self._init_centroids(data)
        meta = {"fallback": centroids.copy(), "n_features": data.shape[1],
                "init": self._resolved_init}
        if self._init_spread is not None:
            meta["init_spread"] = self._init_spread
        if self.revival_geometry == "anisotropic":
            # The split axis needs the data's per-feature scale.  A <=4096-row
            # probe estimated once is what both the Python loop and the compiled
            # epoch runner can see (the runner has no batch at revival time).
            probe = data
            if data.shape[0] > 4096:
                probe = data[self._rng.choice(data.shape[0], size=4096, replace=False)]
            meta["feature_scale"] = np.asarray(probe.std(axis=0), dtype=np.float64)
        accumulators: dict[str, Any] = {}
        if self.finalize:
            accumulators["final_sums"] = np.zeros_like(centroids)
            accumulators["final_counts"] = np.zeros(self.k, dtype=np.float64)
            accumulators["final_complete"] = False
        # Sufficient-statistic seeding with a negligible prior weight:
        # sums = centroids * INIT_USAGE, counts = INIT_USAGE (reference
        # e_means base.py:25), so decode() recovers the initial centroids
        # exactly while an EMA fit gives them essentially zero inertia.
        # Seeding counts at 1 -- the old scheme -- hands the starting
        # centroids the mass of a real observation: at batch ~ k each
        # centroid is dragged toward its (random) start for ~ 1/alpha
        # updates, costing ~1.9x held-out MSE vs the reference at d=8,
        # k=16, n=2000 (measured 38.4 vs 20.3; now 20.5).
        sums = centroids * _INIT_USAGE
        counts = np.full(self.k, _INIT_USAGE)
        return State({"sums": sums, "counts": counts}, 0, meta, {}, accumulators)

    # -- four stages -------------------------------------------------------- #
    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        centroids = self.decode(state)["centroids"]
        dist = squared_distances(batch, centroids, chunk=self.chunk)
        labels = np.argmin(dist, axis=1)
        # np.add.at is one scatter per element; label_sums is one linear
        # pass per column (the reduction this library already documents for
        # from_labels), ~7x cheaper on these shapes (measured 2026-11-14:
        # 2.3 ms vs 1.7 ms/col-set at batch 1024 x d 512; GEMM one-hots are
        # faster again but reorder the float sum, so not here).
        sums = label_sums(batch, labels, self.k)
        counts = np.bincount(labels, minlength=self.k).astype(np.float64)
        own = dist[np.arange(batch.shape[0]), labels]
        inertia = float(own.sum())
        if self.empty_cluster == "reseed" and batch.shape[0] >= self.k:
            labels, sums, counts = self._reseed(batch, dist, labels, sums, counts)
            own = dist[np.arange(batch.shape[0]), labels]
        meta = {"inertia": inertia}
        if self.revival_geometry == "anisotropic":
            # Per-cluster distortion, the donor weight s-means couples to usage.
            meta["cluster_sse"] = np.bincount(labels, weights=own, minlength=self.k)
        return State({"sums": sums, "counts": counts}, 0, meta)

    def _reseed(
        self,
        batch: np.ndarray,
        dist: np.ndarray,
        labels: np.ndarray,
        sums: np.ndarray,
        counts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Give empty clusters the worst-fitting point of a non-singleton cluster.

        This is the standard Lloyd empty-cluster policy: an empty centroid has no
        gradient of its own, so it must be moved explicitly or it stays parked
        outside the data for the rest of the run.
        """
        empty = np.flatnonzero(counts == 0.0)
        if empty.size == 0:
            return labels, sums, counts
        own = dist[np.arange(batch.shape[0]), labels]
        order = np.argsort(-own)
        used = np.zeros(batch.shape[0], dtype=bool)
        labels = labels.copy()
        sums = sums.copy()
        counts = counts.copy()
        for j in empty:
            for i in order:
                source = labels[i]
                if used[i] or counts[source] <= 1.0:
                    continue
                counts[source] -= 1.0
                sums[source] -= batch[i]
                labels[i] = int(j)
                counts[j] = 1.0
                sums[j] = batch[i]
                used[i] = True
                break
        return labels, sums, counts

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        new = super().update_state(state, statistics, ctx)
        if self.revival and new["sums"].shape[1] > 2:
            new = self._revive(new, statistics, ctx)
        return new.inherit_pockets(state)

    def begin_fit(self, state: State, config: FitConfig, n: int, batch_size: int,
                  total_steps: int) -> None:
        # the exact-mean finish accumulates over the whole *last epoch*, which is
        # what makes it a negligible extra cost instead of another full pass.
        state.runtime["final_epoch"] = int(config.nepochs) - 1

    def on_batch(self, state: State, batch: np.ndarray, ctx: UpdateContext) -> None:
        if not self.finalize:
            return
        final_epoch = state.runtime.get(
            "final_epoch", state.meta.get("_final_epoch", -1))
        if ctx.epoch != final_epoch:
            return
        centroids = self.decode(state)["centroids"]
        labels = assign(batch, centroids, chunk=self.chunk)
        state.accumulators["final_sums"] += label_sums(batch, labels, self.k)
        state.accumulators["final_counts"] += np.bincount(labels, minlength=self.k)
        if ctx.step + 1 >= ctx.total_steps:
            state.accumulators["final_complete"] = True

    def decode(self, state: State) -> dict[str, Any]:
        counts = np.asarray(state["counts"], dtype=np.float64)
        sums = np.asarray(state["sums"], dtype=np.float64)
        # the smoothed (EMA) centroid, kept for the shrinkage blend below
        smoothed = sums / np.maximum(counts, 1e-12)[:, None]
        complete = state.accumulators.get(
            "final_complete", state.meta.get("_final_complete", False))
        if self.finalize and complete:
            counts_src = (state.accumulators["final_counts"]
                          if "final_counts" in state.accumulators
                          else state.meta["_final_counts"])
            sums_src = (state.accumulators["final_sums"]
                        if "final_sums" in state.accumulators
                        else state.meta["_final_sums"])
            final_counts = np.asarray(counts_src, dtype=np.float64)
            final_sums = np.asarray(sums_src, dtype=np.float64)
            use = final_counts > 0.5
            counts = np.where(use, final_counts, counts)
            sums = np.where(use[:, None], final_sums, sums)
        safe = np.maximum(counts, 1e-12)
        centroids = sums / safe[:, None]
        if self.finalize_shrink > 0.0 and self.finalize and complete:
            from .smeans import shrink_final

            centroids = shrink_final(centroids, smoothed, counts,
                                     self.finalize_shrink)
        empty = counts <= 1e-12
        if np.any(empty):
            if self.empty_cluster == "keep":
                centroids[empty] = state.meta.get("fallback", centroids)[empty]
            elif self.empty_cluster == "zero":
                centroids[empty] = 0.0
            else:
                raise ValueError(f"unknown empty_cluster policy {self.empty_cluster!r}")
        total = float(counts.sum())
        return {
            "centroids": centroids,
            "counts": counts,
            "weights": counts / total if total > 0 else np.full(self.k, 1.0 / self.k),
        }

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Mean squared distance to the nearest centroid (the k-means MSE)."""
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            return float("nan")
        centroids = parameters["centroids"]
        total = 0.0
        for start in range(0, data.shape[0], max(self.chunk, 1024)):
            block = data[start : start + max(self.chunk, 1024)]
            total += float(squared_distances(block, centroids, chunk=self.chunk).min(axis=1).sum())
        return total / data.shape[0]

    # -- closed form from labels -------------------------------------------- #
    @classmethod
    def from_labels(
        cls, data: np.ndarray, labels: np.ndarray, *, n_clusters: int | None = None
    ) -> tuple["KMeans", dict[str, Any]]:
        """Centroids of a labelled dataset: one reduction, no Lloyd iterations.

        Cluster labels make the centroid update a group-by: the mean of every
        label's rows.  It is the same decode the streaming fit performs, applied
        to statistics a single pass can produce.
        """
        data = np.asarray(data, dtype=np.float64)
        labels = np.asarray(labels).astype(np.int64).ravel()
        if data.ndim != 2:
            raise ValueError("data must be 2-D")
        if labels.size != data.shape[0]:
            raise ValueError(
                f"labels have {labels.size} entries for {data.shape[0]} rows"
            )
        k = int(n_clusters if n_clusters is not None else labels.max() + 1)
        if labels.max() >= k:
            raise ValueError(f"label {int(labels.max())} exceeds n_clusters={k}")
        model = cls(k)
        counts = np.bincount(labels, minlength=k).astype(np.float64)
        sums = label_sums(data, labels, k)
        empty = counts < 0.5
        if np.any(empty):
            counts[empty] = 1e-9
            sums[empty] = data.mean(axis=0)[None, :] * 1e-9
        meta = {"fallback": np.repeat(data.mean(axis=0)[None, :], k, axis=0)}
        params = model.decode(State({"sums": sums, "counts": counts}, 0, meta))
        return model, params

    # -- e-means extras ----------------------------------------------------- #
    def _revive(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        """Split a large cluster to revive clusters whose mass has collapsed.

        The donor is sampled with probability proportional to size and its mass
        donor's mean is preserved).  The threshold is ``revival_threshold``
        times the steady-state mean cluster mass (``batch_size / k``, the
        value ``mean(usage)`` converges to), matching the reference release's
        ``unused_threshold`` semantics -- 0.1 by default, not the paper text's
        "thousandth of an average cluster" the first build used.
        """
        # Test on the state arrays themselves and copy only when a split is
        # actually going to happen: revival is enabled by default, so this
        # ran (and allocated two copies) on every one of the 640 updates in
        # a small-n fit without ever finding a dead cluster.
        counts = np.asarray(state["counts"], dtype=np.float64)
        avg = ctx.batch_size / max(1, self.k)
        threshold = max(1e-12, self.revival_threshold * avg)
        if not np.any(counts < threshold):
            return state
        counts = counts.copy()
        sums = np.array(state["sums"], dtype=np.float64)
        dead = np.flatnonzero(counts < threshold)
        active = np.flatnonzero(counts >= threshold)
        if active.size < 2:
            return state
        inertia = float(statistics.meta.get("inertia", 0.0))
        scale = math.sqrt(max(inertia / (max(ctx.batch_size, 1) * sums.shape[1]), 1e-12))
        cluster_sse = statistics.meta.get("cluster_sse")
        if cluster_sse is not None and self.revival_geometry == "anisotropic":
            # s-means donor weight: distortion * usage^0.35, so a large cluster
            # that already fits well is not the first one split.
            weight = np.asarray(cluster_sse, dtype=np.float64)[active] * np.power(
                counts[active], 0.35)
            total = float(weight.sum())
            probs = weight / total if total > 0.0 else counts[active] / counts[active].sum()
        else:
            probs = counts[active] / counts[active].sum()
        feature_scale = state.meta.get("feature_scale")
        for j in dead:
            donor = int(self._rng.choice(active, p=probs))
            if counts[donor] <= 2.0 * threshold:
                continue
            mu = sums[donor] / counts[donor]
            half = counts[donor] / 2.0
            if self.revival_geometry == "anisotropic" and feature_scale is not None:
                from .smeans import anisotropic_split

                distortion = (float(cluster_sse[donor])
                              if cluster_sse is not None else inertia)
                lo, hi = anisotropic_split(mu, counts[donor], distortion,
                                           feature_scale, self._rng)
            else:
                direction = self._rng.normal(size=sums.shape[1])
                norm = float(np.linalg.norm(direction))
                if norm <= 0.0:
                    continue
                delta = scale * direction / norm
                lo, hi = mu - delta, mu + delta
            sums[donor] = lo * half
            sums[j] = hi * half
            counts[donor] = half
            counts[j] = half
        return State({"sums": sums, "counts": counts}, state.step, dict(state.meta))

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "k": self.k,
                "init": self.init,
                "empty_cluster": self.empty_cluster,
                "revival": self.revival,
                "revival_geometry": self.revival_geometry,
                "finalize": self.finalize,
                "finalize_shrink": self.finalize_shrink,
            }
        )
        return out

    # -- convenience -------------------------------------------------------- #
    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int = 5,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(
            batch_size=batch_size,
            nepochs=nepochs,
            seed=seed,
            track_memory=track_memory,
        )
        return fit(self, data, cfg, eval_data=eval_data)


class LloydKMeans(KMeans):
    """Conventional k-means: one full-dataset rebuild per iteration.

    Restarts (``n_init``, best objective wins) and empty-cluster reseeding are
    part of the reference algorithm as every library implements it; without them
    this baseline is weaker than the thing it is supposed to bound.
    """

    name = "lloyd"
    supports_streaming = False

    def __init__(
        self,
        k: int,
        *,
        niter: int = 25,
        n_init: int = 3,
        seed: int = 0,
        init: str = "kmeans++",
        empty_cluster: str = "reseed",
        init_sample: int | None = None,
        tol: float | None = None,
        rel_tol: float | None = 1e-6,
    ) -> None:
        super().__init__(k, updater=Replace(), seed=seed, init=init,
                         empty_cluster=empty_cluster, init_sample=init_sample)
        self.niter = int(niter)
        self.n_init = int(n_init)
        self.tol = tol
        self.rel_tol = rel_tol

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"niter": self.niter, "n_init": self.n_init,
                    "tol": self.tol, "rel_tol": self.rel_tol})
        return out

    def resolve_batch_size(self, config: FitConfig, n: int, *, d: int | None = None) -> int:
        return n

    def fit(
        self,
        data: np.ndarray,
        *,
        niter: int | None = None,
        n_init: int | None = None,
        seed: int | None = None,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        data = np.asarray(data, dtype=np.float64)
        iterations = int(niter or self.niter)
        restarts = int(n_init if n_init is not None else self.n_init)
        base_seed = self.seed if seed is None else int(seed)
        # A five-iteration run is a fixed unit test contract, not a convergence
        # budget; checking at that scale would make the check the whole run.
        stop_tol = self.tol if iterations >= 10 else None
        stop_rel = self.rel_tol if iterations >= 10 else None
        best: FitResult | None = None
        for restart in range(max(1, restarts)):
            self.seed = base_seed + restart
            self._rng = np.random.default_rng(self.seed)
            cfg = FitConfig(
                batch_size=None,
                nepochs=iterations,
                seed=self.seed,
                track_memory=track_memory,
                tol=stop_tol,
                rel_tol=stop_rel,
                # Lloyd's objective is a second full distance pass.  Scoring
                # every other full pass halves that cost; the convergence check
                # still sees a two-iteration change, which is what the relative
                # tolerance is for.
                eval_every_updates=2,
            )
            result = fit(self, data, cfg, eval_data=eval_data)
            result.extras["restart"] = restart
            if best is None or result.objective < best.objective:
                best = result
        self.seed = base_seed
        self._rng = np.random.default_rng(base_seed)
        assert best is not None
        best.extras["n_init"] = restarts
        return best


class EMeans(KMeans):
    """e-means: EMA centroid statistics with the paper's compute-saving extras."""

    name = "emeans"

    def __init__(
        self,
        k: int,
        *,
        nepochs: int = 5,
        batch_size: int | None = None,
        alpha0: float | None = None,
        batch_law: str = "auto",
        alpha_end_ratio: float = 0.1,
        revival: bool = True,
        revival_threshold: float = 0.1,
        finalize: bool = True,
        finalize_shrink: float = 0.0,
        revival_geometry: str = "isotropic",
        #: ``auto`` is the fast opt-in: once the data is at least eight
        #: coresets wide, the seeded Lloyd-on-coreset path replaces the
        #: O(n k d) full-data k-means++ sweep.  It stays opt-in because the
        #: five-draw study failed D1/D2 (worst +52%); use it when the measured
        #: init cost matters and the quality tail is acceptable.
        init: str = "kmeans++",
        init_centroids: np.ndarray | None = None,
        init_coreset: int | None = None,
        init_draws: int = 1,
        max_points_per_centroid: int | None = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(
            k,
            updater=EMA(alpha=1.0),
            seed=seed,
            init=init,
            init_centroids=init_centroids,
            init_coreset=init_coreset,
            init_draws=init_draws,
            revival=revival,
            revival_threshold=revival_threshold,
            finalize=finalize,
            finalize_shrink=finalize_shrink,
            revival_geometry=revival_geometry,
        )
        self.nepochs = int(nepochs)
        self.batch_size = batch_size
        self.alpha0 = alpha0
        self.batch_law = batch_law
        self.alpha_end_ratio = float(alpha_end_ratio)
        self.max_points_per_centroid = max_points_per_centroid
        self._resolved_batch = batch_size
        self._resolved_alpha0 = alpha0

    def resolve_batch_size(self, config: FitConfig, n: int, *, d: int | None = None) -> int:
        """An explicit batch wins, then the constructor, then the fitted law.

        ``"auto"`` (the ``FitConfig`` default) means "use this optimizer's own
        resolution", which is the law unless the constructor was given a size.
        ``"full"`` is the explicit one-batch-per-pass request; ``None`` keeps its
        long-standing meaning here (defer to the optimizer), because
        ``EMeans.fit`` uses it to ask for the law.
        """
        requested = config.batch_size
        if requested == "full":
            bs = int(n)
        elif requested is None or requested == "auto":
            if self.batch_size is not None:
                bs = int(np.clip(self.batch_size, 1, n))
            else:
                bs = batch_size_from_law(n, self.k, config.nepochs,
                                         batch_law=self.batch_law, d=d)
        else:
            bs = int(np.clip(int(requested), 1, n))
        self._resolved_batch = bs
        return bs

    def begin_fit(self, state: State, config: FitConfig, n: int, batch_size: int,
                  total_steps: int) -> None:
        # keep the finalization accumulator window from the base class
        super().begin_fit(state, config, n, batch_size, total_steps)
        alpha0 = self.alpha0 if self.alpha0 is not None else alpha0_from_law(
            n, self.k, batch_size, config.nepochs
        )
        self._resolved_alpha0 = alpha0
        # the step is annealed along the planned horizon, so the schedule is
        # only knowable once the compute budget is fixed.
        self.updater = EMA(
            schedule=CosineSchedule(alpha0, alpha_end_ratio=self.alpha_end_ratio)
        )
        self._alpha_cache: tuple[Any, np.ndarray] | None = None

    # -- compiled epoch runner ---------------------------------------------- #
    def _alpha_grid(self, total_steps: int, batch_size: int,
                    dataset_size: int) -> np.ndarray:
        """The whole horizon's step sizes, computed once.

        Same values as asking the schedule per update -- ``CosineSchedule``
        provides a vector form and is pinned equal to its scalar form by a
        test -- but the scalar loop costs 1.3 ms over 640 updates, which is more
        than the kernel it feeds.
        """
        key = (int(total_steps), int(batch_size), int(dataset_size))
        if self._alpha_cache is not None and self._alpha_cache[0] == key:
            return self._alpha_cache[1]
        schedule = getattr(self.updater, "schedule", None)
        grid = getattr(schedule, "alpha_grid", None)
        total = max(int(total_steps), 1)
        if grid is not None:
            values = np.ascontiguousarray(grid(total, batch_size, dataset_size),
                                          dtype=np.float64)
        else:
            values = np.ascontiguousarray(
                [float(self.updater.alpha(UpdateContext(
                    step=step, batch_size=int(batch_size),
                    dataset_size=int(dataset_size), total_steps=total)))
                 for step in range(total)], dtype=np.float64)
        self._alpha_cache = (key, values)
        return values

    def epoch_runner_ready(self, n: int, d: int, batch_size: int,
                           per_epoch: int, total_steps: int) -> bool:
        """Whether the compiled runner can take this fit's whole epochs.

        The runner advances ``examples`` a whole epoch at a time and cannot
        call back per batch, so ``fit`` only offers it when nothing else would
        be lost.  A release install has no runner and this is ``False``.
        """
        if per_epoch < 1 or batch_size < 1:
            return False
        runner = _internal_runner()
        return runner is not None and runner.ready(
            n, d, self.k, batch_size, per_epoch, total_steps)

    def run_epoch(self, state: State, data: np.ndarray, order: np.ndarray,
                  batch_size: int, per_epoch: int, epoch: int,
                  total_steps: int, step0: int) -> int:
        """Run one epoch's updates through the compiled kernel, in place.

        The kernel takes the same steps in the same order as ``measure`` +
        ``update_state``: decode, fold, assign, accumulate, then test for a dead
        cluster.  Revival is the one thing the kernel cannot do (it draws from
        the model's RNG), so it stops and hands control back for that step.
        """
        runner = _internal_runner()
        if runner is None:
            raise RuntimeError("no internal epoch runner is built")

        k = self.k
        d = int(data.shape[1])
        flat = np.require(data, dtype=np.float64, requirements="C").reshape(-1)
        sums = np.require(state["sums"], dtype=np.float64,
                          requirements="C").reshape(-1)
        counts = np.require(state["counts"], dtype=np.float64,
                            requirements="C").reshape(-1)
        index = np.ascontiguousarray(order, dtype=np.int64)
        alphas = self._alpha_grid(total_steps, batch_size, int(data.shape[0]))
        fallback = np.require(state.meta["fallback"], dtype=np.float64,
                              requirements="C").reshape(-1)
        # ``centroids`` is scratch whose layout depends on the kernel: the tiled
        # form decodes it transposed as ``(d, k)`` so its inner ``j`` loop is
        # contiguous.  Nothing outside the call reads it.
        centroids = np.empty(k * d, dtype=np.float64)
        acc = np.zeros(k, dtype=np.float64)
        use_tiled = runner.tiled
        inertia = np.zeros(1, dtype=np.float64)
        scratch = np.zeros(k * d, dtype=np.float64)
        scratch_counts = np.zeros(k, dtype=np.float64)
        final_epoch = int(state.runtime.get("final_epoch", self.nepochs - 1))
        do_final = bool(self.finalize and epoch == final_epoch
                        and "final_sums" in state.accumulators)
        if do_final:
            final_sums = np.require(state.accumulators["final_sums"],
                                    dtype=np.float64,
                                    requirements="C").reshape(-1)
            final_counts = np.require(state.accumulators["final_counts"],
                                      dtype=np.float64, requirements="C")
        else:
            final_sums = scratch
            final_counts = scratch_counts
        # ``update_state`` revives only when the state is wider than two
        # features -- the reference turns revival off at d <= 2 -- so the kernel
        # must not revive there either.  That guard is not cosmetic: at d=2 with
        # a small batch, every count starts below the threshold, so reviving
        # would split clusters the shipped path leaves alone (measured: a
        # ProductQuantizer block fell from 0.0002 to 0.55 quantization error).
        revive = bool(self.revival and d > 2)
        threshold = (max(1e-12, self.revival_threshold * batch_size / max(1, k))
                     if revive else -1.0)
        done = 0
        while done < per_epoch:
            # The kernel reads ``order[s * batch + i]``, so a resumed segment
            # gets the epoch's rows sliced at the update it stopped on -- without
            # that it would re-read the epoch's first rows after every revival,
            # and revival fires from the very first update (counts start at
            # 2**-16, below 0.1 * batch / k).
            args = [flat, index[done * batch_size:], alphas, sums, counts,
                    centroids, fallback]
            if use_tiled:
                args.append(acc)
            args += [final_sums, final_counts, d, k, batch_size,
                     per_epoch - done, step0 + done,
                     1 if do_final else 0, threshold, inertia]
            completed = runner.call(args, tiled=use_tiled)
            done += completed
            if done < per_epoch:
                # a cluster fell below the revival threshold at this step
                ctx = UpdateContext(
                    step=step0 + done, batch_size=batch_size,
                    dataset_size=int(data.shape[0]), total_steps=total_steps,
                    epoch=epoch)
                revived = self._revive(
                    State({"sums": sums.reshape(k, d), "counts": counts},
                          0, dict(state.meta)),
                    State({}, 0, {"inertia": float(inertia[0])}), ctx)
                np.copyto(sums, np.ravel(revived["sums"]))
                np.copyto(counts, np.ravel(revived["counts"]))
        state.values["sums"] = sums.reshape(k, d)
        state.values["counts"] = counts
        if do_final and step0 + per_epoch >= total_steps:
            state.accumulators["final_complete"] = True
        return per_epoch

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update(
            {
                "nepochs": self.nepochs,
                "resolved_batch_size": self._resolved_batch,
                "resolved_alpha0": self._resolved_alpha0,
                "batch_law": self.batch_law,
            }
        )
        return out
