"""LDA topic models: batch variational EM reference versus persistent statistics.

LDA is another control: online variational Bayes (Hoffman et al.) already keeps
minibatch state, so "is online LDA possible" is not the question.  The question
is whether the *e-means-style* state update -- folding the word-topic counts in
with a calibrated EMA -- changes the compute/quality frontier.

State: ``lam`` (K, V) word-topic counts.  Decode: ``beta = lam / sum(lam)``.
Measurement: variational E-step on a mini-batch, then ``lam_b = eta + sum_d phi_d``.
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
    UpdateContext,
    fit,
)
from ..state import EMA, Replace
from ..state.sufficient import SufficientStat

_EPS = 1e-12


def digamma(x: np.ndarray) -> np.ndarray:
    """Digamma with recurrence + asymptotic series (no SciPy dependency)."""
    x = np.asarray(x, dtype=np.float64)
    result = np.zeros_like(x)
    acc = x.copy()
    mask = acc < 6.0
    while np.any(mask):
        result = np.where(mask, result - 1.0 / np.maximum(acc, _EPS), result)
        acc = np.where(mask, acc + 1.0, acc)
        mask = acc < 6.0
    inv = 1.0 / acc
    inv2 = inv * inv
    series = np.log(acc) - 0.5 * inv - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    return result + series


def lda_e_step(
    counts: np.ndarray, beta: np.ndarray, alpha: float, iters: int
) -> tuple[np.ndarray, np.ndarray]:
    """Variational E-step shared by the optimizer and :class:`LDAStat`.

    Returns ``(gamma (B, K), phi (B, K, V))`` for a batch of documents.
    """
    counts = np.asarray(counts, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    b, v = counts.shape
    k = beta.shape[0]
    gamma = alpha + counts.sum(axis=1)[:, None] / k
    safe_beta = np.maximum(beta, _EPS)
    # ``exp(log beta + log eg)`` is ``beta * eg``: the log/exp round trip was
    # recomputing values already in hand, at B*K*V exps per inner iteration
    # (21.6M exps for a 10-epoch fit at 600 x 6 x 300).  ``eg`` is an exp, so it
    # is positive and the multiplicative form is the same quantity.
    weights = np.empty((b, k, v), dtype=np.float64)
    phi = np.empty((b, k, v), dtype=np.float64)
    denom = np.empty((b, 1, v), dtype=np.float64)
    eg = np.empty((b, k), dtype=np.float64)
    for _ in range(int(iters)):
        # In-place through the inner loop: the old form allocated three fresh
        # B*K*V temporaries per iteration (8.6 MB each at this shape).
        np.exp(digamma(gamma) - digamma(gamma.sum(axis=1, keepdims=True)), out=eg)
        np.multiply(eg[:, :, None], safe_beta[None, :, :], out=weights)
        np.sum(weights, axis=1, keepdims=True, out=denom)
        np.maximum(denom, _EPS, out=denom)
        np.divide(weights, denom, out=phi)
        gamma = alpha + np.einsum("bkv,bv->bk", phi, counts)
    return gamma, phi


class LDAStat(SufficientStat):
    """Word-topic counts as a sufficient statistic.

    Values are the smoothed M-step counts ``eta + sum_d phi_dv`` -- the same
    array ``LDA.measure`` returns -- and the topic matrix ``beta`` is the
    derived parameter, normalised from the accumulated counts.
    """

    name = "lda_stat"
    keys = ("lam",)

    def __init__(
        self,
        n_topics: int,
        vocab: int,
        *,
        eta: float = 0.1,
        alpha: float = 0.1,
        e_step_iters: int = 20,
        mode: str = "sum",
        decay: float | Any = 1.0,
        act_every: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(mode=mode, alpha=decay, act_every=act_every, seed=seed)
        self.n_topics = int(n_topics)
        self.vocab = int(vocab)
        self.eta = float(eta)
        self.doc_alpha = float(alpha)
        self.e_step_iters = int(e_step_iters)

    def prepare(self, lam: np.ndarray | None = None) -> "LDAStat":
        """Seed the E-step's ``beta``; the accumulator itself starts empty.

        ``values["lam"]`` holds *raw* word-topic counts, with the ``eta``
        smoothing applied once at decode.  Adding ``eta`` per block (the
        optimizer's per-measurement convention) would make the statistic depend
        on how the data was blocked, and merging partitions would then add the
        prior once per partition -- measured as a 0.3 count difference on a
        4-way split.  The seed is a parameter, not a measurement.
        """
        if lam is None:
            rng = np.random.default_rng(self.seed)
            lam = self.eta + rng.random((self.n_topics, self.vocab)) * 0.1
        lam = np.asarray(lam, dtype=np.float64)
        self.params["lam"] = np.array(lam, copy=True)
        self.params["beta"] = lam / np.maximum(lam.sum(axis=1, keepdims=True), _EPS)
        return self

    def beta(self) -> np.ndarray:
        if "lam" in self.values:
            lam = self.eta + np.asarray(self.values["lam"], dtype=np.float64)
        else:
            lam = np.asarray(self.params["lam"], dtype=np.float64)
        return lam / np.maximum(lam.sum(axis=1, keepdims=True), _EPS)

    def block_statistic(self, block: np.ndarray, weight: float) -> dict[str, np.ndarray]:
        if "lam" not in self.values or "beta" not in self.params:
            self.prepare()
        # the E-step runs against the last *acted* beta, so frozen-parameter
        # accumulation is a true statistic and merges exactly
        beta = np.asarray(self.params["beta"], dtype=np.float64)
        _, phi = lda_e_step(block, beta, self.doc_alpha, self.e_step_iters)
        return {"lam": np.einsum("bkv,bv->kv", phi, block)}

    def act(self) -> None:
        if "lam" not in self.values:
            return
        counts = np.asarray(self.values["lam"], dtype=np.float64)
        self.params["lam"] = self.eta + counts
        self.params["beta"] = self.beta()
        self._dirty = False

    def decode(self) -> dict[str, Any]:
        self._maybe_act()
        if "lam" in self.values:
            counts = np.asarray(self.values["lam"], dtype=np.float64)
            lam = self.eta + counts
        else:
            lam = np.asarray(self.params["lam"], dtype=np.float64)
        return {"beta": self.beta(), "lam": lam, "topics": self.n_topics}

    def extra_config(self) -> dict[str, Any]:
        return {"n_topics": self.n_topics, "vocab": self.vocab, "eta": self.eta,
                "doc_alpha": self.doc_alpha, "e_step_iters": self.e_step_iters}


class LDA(StatisticalOptimizer):
    """Smoothed LDA with variational E-step and persistent word-topic counts."""

    name = "lda"
    state_keys = ("lam",)
    supports_streaming = True

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        k = self.n_topics if k is None else int(k)
        return int(k * d * 8 * 2)

    def stat(self, vocab: int, *, mode: str = "sum", alpha: Any = 1.0) -> LDAStat:
        """Word-topic counts as a mergeable sufficient statistic."""
        return LDAStat(self.n_topics, int(vocab), eta=self.eta, alpha=self.alpha,
                       e_step_iters=self.e_step_iters, mode=mode, decay=alpha,
                       seed=self.seed)

    def __init__(
        self,
        n_topics: int,
        *,
        eta: float = 0.1,
        alpha: float = 0.1,
        e_step_iters: int = 20,
        updater: Any = None,
        seed: int = 0,
    ) -> None:
        super().__init__(updater or Replace())
        self.n_topics = int(n_topics)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.e_step_iters = int(e_step_iters)
        self.seed = int(seed)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: np.ndarray | DataSpec, config: Any = None) -> State:
        data = np.asarray(data_spec, dtype=np.float64)
        vocab = data.shape[1]
        rng = np.random.default_rng(self.seed)
        lam = self.eta + rng.random((self.n_topics, vocab)) * 0.1
        return State({"lam": lam}, 0, {"n_topics": self.n_topics, "vocab": vocab})

    def decode(self, state: State) -> dict[str, Any]:
        lam = np.asarray(state["lam"], dtype=np.float64)
        beta = lam / np.maximum(lam.sum(axis=1, keepdims=True), _EPS)
        return {"beta": beta, "lam": lam, "topics": self.n_topics}

    # -- variational E-step ------------------------------------------------- #
    def e_step(self, counts: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(gamma (B, K), phi (B, K, V))`` for a batch of documents."""
        return lda_e_step(counts, beta, self.alpha, self.e_step_iters)

    def measure(self, batch: np.ndarray, state: State) -> State:
        batch = np.asarray(batch, dtype=np.float64)
        beta = self.decode(state)["beta"]
        _, phi = self.e_step(batch, beta)
        # M-step: weight each word type by how often it occurs (phi is a
        # per-token distribution, so unweighted summation makes uniform beta a
        # fixed point and no amount of data can move it).
        lam_b = self.eta + np.einsum("bkv,bv->kv", phi, batch)
        return State({"lam": lam_b}, 0, {})

    def objective(self, data: np.ndarray, parameters: dict[str, Any]) -> float:
        """Per-word negative log-likelihood (nats); ``exp`` of this is perplexity."""
        data = np.asarray(data, dtype=np.float64)
        beta = parameters["beta"]
        gamma, _ = self.e_step(data, beta)
        theta = gamma / np.maximum(gamma.sum(axis=1, keepdims=True), _EPS)
        word_prob = np.maximum(theta @ beta, _EPS)
        nll = -float(np.sum(data * np.log(word_prob)))
        return nll / max(float(data.sum()), _EPS)

    def perplexity(self, data: np.ndarray, parameters: dict[str, Any] | None = None) -> float:
        parameters = parameters or {}
        return float(np.exp(self.objective(np.asarray(data, dtype=np.float64), parameters)))

    def fit(
        self,
        data: np.ndarray,
        *,
        batch_size: int | None = None,
        nepochs: int = 10,
        seed: int = 0,
        eval_data: np.ndarray | None = None,
        track_memory: bool = False,
    ) -> FitResult:
        cfg = FitConfig(batch_size=batch_size, nepochs=nepochs, seed=seed,
                        track_memory=track_memory)
        return fit(self, data, cfg, eval_data=eval_data)

    def describe(self) -> dict[str, Any]:
        out = super().describe()
        out.update({"n_topics": self.n_topics, "eta": self.eta, "alpha": self.alpha})
        return out
