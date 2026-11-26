"""The four-stage abstraction the whole library is built on.

    measurement -> state update -> parameter decode -> objective -> evidence

A conventional iterative statistical algorithm re-estimates a statistic from the
current batch and throws the previous estimate away.  Every optimizer in
``emaopt`` instead separates:

``measure(batch, state)``
    reduce the batch to a *statistic* (per-cluster sums/counts, word-topic
    counts, ``X^T U``, higher-order gradients, ...).

``update_state(state, statistics, ctx)``
    fold the statistic into persistent state with a replaceable
    :class:`~emaopt.state.base.StateUpdater` (replace, EMA, running mean,
    bias-corrected EMA, Robbins-Monro, adaptive).

``decode(state)``
    turn state into parameters (centroids, means/covariances, a codebook, a
    dictionary, a topic matrix, an unmixing matrix).

``objective(data, parameters)``
    score parameters on data.  This is the only thing that becomes evidence.

Swapping the updater changes the optimization law without touching the
algorithm, which is what makes the research question ("does persistent state
change the compute/quality frontier?") cheap to ask.
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# numerical helpers
# --------------------------------------------------------------------------- #
def squared_distances(x: Array, centroids: Array, *, chunk: int = 4096,
                      x_norm: Array | None = None) -> Array:
    """Squared Euclidean distances ``(n, k)`` computed in row chunks.

    Uses ``|x|^2 - 2 x.c + |c|^2`` and clips at zero so that round-off cannot
    produce negative distances (which would poison ``argmin`` ties).
    ``x_norm`` may carry precomputed ``einsum('ij,ij->i', x, x)``; the row
    norms are constants of ``x``, so callers that sweep the same rows many
    times (k-means++ greedy init) pass them to avoid one full memory pass per
    sweep (measured: the norm einsum is ~half of the distance cost).
    """
    x = np.asarray(x, dtype=np.float64)
    c = np.asarray(centroids, dtype=np.float64)
    if c.ndim != 2 or x.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError(f"shape mismatch: x{x.shape} centroids{c.shape}")
    out = np.empty((x.shape[0], c.shape[0]), dtype=np.float64)
    c_norm = np.einsum("ij,ij->i", c, c)
    if x_norm is not None:
        for start in range(0, x.shape[0], chunk):
            block = x[start : start + chunk]
            d = c_norm[None, :] - 2.0 * (block @ c.T)
            d += np.asarray(x_norm)[start : start + chunk, None]
            np.maximum(d, 0.0, out=d)
            out[start : start + chunk] = d
        return out
    for start in range(0, x.shape[0], chunk):
        block = x[start : start + chunk]
        d = c_norm[None, :] - 2.0 * (block @ c.T)
        d += np.einsum("ij,ij->i", block, block)[:, None]
        np.maximum(d, 0.0, out=d)
        out[start : start + chunk] = d
    return out


def assign(x: Array, centroids: Array, *, chunk: int = 4096) -> Array:
    """Nearest-centroid label for every row of ``x``."""
    return np.argmin(squared_distances(x, centroids, chunk=chunk), axis=1)


#: Feature count at which one flat bincount beats one bincount per column.
#: Measured interleaved on this box: 3.09x faster at d=512, 1.10x at d=64,
#: 0.78x at d=8 (building the flat index costs more than it saves with few
#: columns), so the per-column form stays for narrow data.
FLAT_BINCOUNT_MIN_D = 32


def _flat_label_reduce(data: Array, labels: Array, k: int) -> Array:
    """One weighted bincount over ``(label, column)`` bins.

    The per-column form walks the whole column once per feature -- 64 or 512
    passes over the data.  Flattening the bin to ``label*d + j`` makes it one
    pass, and it is bit-identical: a fixed ``(label, j)`` bin still sees its
    rows in row order, so the float accumulation order does not move.
    """
    d = data.shape[1]
    idx = np.asarray(labels, dtype=np.int64)[:, None] * d + np.arange(
        d, dtype=np.int64)
    return np.bincount(idx.ravel(), weights=np.asarray(data).ravel(),
                       minlength=k * d).reshape(k, d)


def label_sums(data: Array, labels: Array, k: int) -> Array:
    """Per-label column sums ``(k, d)``.

    ``np.add.at`` is a scatter and runs at a few tens of millions of updates per
    second, which is the wrong tool when the construction is applied to a million
    rows of 512 features (the LAION recipe's last stage).  ``bincount`` with
    weights is one pass per column, and for wide data one pass over all
    ``(label, column)`` bins is cheaper again (3.09x at d=512, 1.10x at d=64).
    """
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    # Wide data pays for the flat form outright; tiny batches pay for it in
    # call overhead instead (a 24-row, 8-column e-means batch was making
    # eight bincount calls per update).  The flat bin is bit-identical.
    if data.shape[1] >= FLAT_BINCOUNT_MIN_D or data.shape[0] <= 256:
        return _flat_label_reduce(data, labels, k)
    out = np.empty((k, data.shape[1]), dtype=np.float64)
    for j in range(data.shape[1]):
        out[:, j] = np.bincount(labels, weights=data[:, j], minlength=k)[:k]
    return out


def label_squares(data: Array, labels: Array, k: int) -> Array:
    """Per-label column sums of squares ``(k, d)``, same reduction rule."""
    data = np.asarray(data, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if data.shape[1] >= FLAT_BINCOUNT_MIN_D:
        return _flat_label_reduce(data * data, labels, k)
    out = np.empty((k, data.shape[1]), dtype=np.float64)
    for j in range(data.shape[1]):
        out[:, j] = np.bincount(labels, weights=data[:, j] ** 2, minlength=k)[:k]
    return out


def softmax_rows(logits: Array) -> Array:
    """Row-wise softmax that is safe for very negative logits."""
    m = np.max(logits, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    e = np.exp(logits - m)
    s = np.sum(e, axis=1, keepdims=True)
    return e / np.maximum(s, np.finfo(np.float64).tiny)


#: ``n*k`` elements above which a compiled row kernel pays its dispatch.
#: Below the gate the dispatch itself is the cost, so NumPy stays.  Regenerate
#: the gate with ``scripts/gen_hostkernels.py``; the ratios it measures are
#: recorded in ``evidence/host_kernels.json``.
_LOGSUMEXP_KERNEL_GATE = 32_768


def logsumexp_rows(logits: Array) -> Array:
    """Numerically stable ``log(sum(exp(logits)))`` along axis 1.

    The ``exp`` is applied into the shifted temporary instead of a new one: the
    same operations in the same order (so the result is bit-identical to the
    three-temporary form, which is what keeps an objective history comparable),
    with one fewer ``n*k`` allocation.  Measured on the maintainer box
    at the eGMM scoring shapes: 3.02 -> 2.25 ms
    at 20000 x 16, 9.56 -> 5.84 at 20000 x 64, 21.89 -> 16.61 at 200000 x 8.
    That is the whole of this function's cost on the scoring path, which
    ``cProfile`` puts at 6 of a 16 ms fit's 16 ms after the seed change.
    """
    logits = np.asarray(logits)
    if (logits.ndim == 2 and logits.dtype == np.float64
            and logits.size >= _LOGSUMEXP_KERNEL_GATE):
        # One output per row, so the scoring pass threads without a reduction.
        # The compiled arms come first (C, then the opt-in numba backend, then an
        # internal provider); NumPy is the floor.  ``EMAOPT_LOGSUMEXP=numpy``
        # turns the compiled pass off for an A/B, which is how the streaming
        # bench measures the scoring kernel on its own (``EMAOPT_HOST_KERNEL``
        # would also disable the E-step).
        off = os.environ.get("EMAOPT_LOGSUMEXP", "").strip().lower() in {
            "numpy", "0", "off", "false"}
        if not off:
            from . import hostkernels as _host

            got = _host.logsumexp_rows(logits)
            if got is not None:
                return got
    m = np.max(logits, axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    shifted = logits - m
    np.exp(shifted, out=shifted)
    return (m + np.log(shifted.sum(axis=1, keepdims=True))).ravel()


# --------------------------------------------------------------------------- #
# state and context
# --------------------------------------------------------------------------- #
#: Bumped when the on-disk shape of :meth:`State.to_dict` changes incompatibly.
STATE_VERSION = 1


def _encode(obj: Any) -> Any:
    """JSON-safe form of a state payload; arrays keep their dtype."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): _encode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _decode(obj: Any) -> Any:
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.asarray(obj["__ndarray__"], dtype=obj.get("dtype") or np.float64)
        return {k: _decode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v) for v in obj]
    return obj


def _deep(obj: Any) -> Any:
    """Recursive copy: arrays are copied, dicts and lists are rebuilt."""
    if isinstance(obj, np.ndarray):
        return np.array(obj, copy=True)
    if isinstance(obj, dict):
        return {k: _deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep(v) for v in obj]
    return obj


@dataclass
class State:
    """Persistent sufficient-statistic state, in three separate pockets.

    ``values``
        the statistics the updater folds in -- the only thing decoded.
    ``meta``
        durable data that must travel with the state but is not decoded:
        fallback means, a model's own constants, recursion internals.
    ``runtime``
        run-scoped bookkeeping (the finalization epoch, the last epoch seen).
        A resumed or re-planned fit may reset this without touching the state.
    ``accumulators``
        live arrays a ``finalize`` pass adds to between updates (the exact
        statistics of the last epoch, and the flag that says they are complete).

    They used to share ``meta``, which meant a shallow ``copy()`` could alias a
    live accumulator and a serialized state could not be read back without
    knowing every private key (T43).  ``to_dict``/``from_dict`` are versioned and
    keep arrays as arrays; ``copy()`` is deep in all four pockets.
    """

    values: dict[str, Array]
    step: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)
    accumulators: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Array:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def keys(self) -> tuple[str, ...]:
        return tuple(self.values)

    def replace_values(self, **updates: Array) -> "State":
        values = dict(self.values)
        values.update(updates)
        return State(values, self.step, dict(self.meta), dict(self.runtime),
                     dict(self.accumulators))

    def copy(self) -> "State":
        return State(
            _deep(self.values),
            self.step,
            _deep(self.meta),
            _deep(self.runtime),
            _deep(self.accumulators),
        )

    def inherit_pockets(self, other: "State") -> "State":
        """Carry ``runtime`` and ``accumulators`` over from ``other``.

        An updater owns ``values`` and its own recursion ``meta``; it has no
        business reconstructing the fit's bookkeeping.  Every law builds a fresh
        ``State`` from the values it blended, so without this the accumulators a
        finalizer adds to would be dropped on the next update -- which is exactly
        what the split exposed (T43).  Shallow, so the arrays stay shared and the
        finalizer's in-place adds land in the loop's state.
        """
        runtime = dict(other.runtime)
        runtime.update(self.runtime)
        self.runtime = runtime
        accumulators = dict(other.accumulators)
        accumulators.update(self.accumulators)
        self.accumulators = accumulators
        return self

    def to_dict(self) -> dict[str, Any]:
        """A JSON-safe, versioned snapshot; :meth:`from_dict` restores it."""
        return {
            "version": STATE_VERSION,
            "step": int(self.step),
            "values": _encode(self.values),
            "meta": _encode(self.meta),
            "runtime": _encode(self.runtime),
            "accumulators": _encode(self.accumulators),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "State":
        version = int(payload.get("version", 0))
        if version != STATE_VERSION:
            raise ValueError(
                f"unsupported State version {version}; expected {STATE_VERSION}"
            )
        values = {k: np.asarray(v, dtype=np.float64)
                  for k, v in _decode(payload.get("values") or {}).items()}
        return cls(
            values=values,
            step=int(payload.get("step", 0)),
            meta=_decode(payload.get("meta") or {}),
            runtime=_decode(payload.get("runtime") or {}),
            accumulators=_decode(payload.get("accumulators") or {}),
        )


@dataclass(frozen=True)
class UpdateContext:
    """Everything an updater or schedule is allowed to condition on."""

    step: int
    batch_size: int
    dataset_size: int
    total_steps: int = 1
    epoch: int = 0
    examples_processed: int = 0

    @property
    def progress(self) -> float:
        """Fraction of the planned optimization horizon consumed, in (0, 1]."""
        return min(1.0, (self.step + 1) / max(1, self.total_steps))


@dataclass(frozen=True)
class DataSpec:
    """Problem shape, for optimizers that can be initialized without data."""

    n: int
    d: int
    k: int = 0
    seed: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


class Schedule(Protocol):
    """Maps update context to a step size."""

    def alpha(self, ctx: UpdateContext) -> float:  # pragma: no cover - protocol
        ...


# --------------------------------------------------------------------------- #
# optimizer interface
# --------------------------------------------------------------------------- #
class StatisticalOptimizer:
    """Base class: measurement -> state update -> decode -> objective."""

    name: str = "statistical_optimizer"
    state_keys: tuple[str, ...] = ()
    #: Whether a smaller batch is still the same algorithm.  Lloyd is the
    #: canonical False: its rebuild *is* the full pass, so the budget resolver
    #: records the pressure but cannot make it stream.  Everything with an
    #: incremental statistic (e-means, the mixture, ICA, PCA, NMF, ...) is True.
    supports_streaming: bool = False

    def __init__(self, updater: Any = None, **config: Any) -> None:
        if updater is None:
            from .state import make_updater

            updater = make_updater("replace")
        self.updater = updater
        self.config = dict(config)

    # -- lifecycle ---------------------------------------------------------- #
    def initialize(self, data_spec: Array | DataSpec, config: Any = None) -> State:
        raise NotImplementedError

    def begin_fit(self, state: State, config: "FitConfig", n: int, batch_size: int,
                  total_steps: int) -> None:
        """Hook for optimizers that derive schedules from problem shape."""

    def resolve_batch_size(self, config: "FitConfig", n: int, *, d: int | None = None) -> int:
        """Batch size for one update.  Optimizers with a law override this.

        The base implementation has no law, so ``"auto"`` falls back to the
        historical mini-batch and ``None``/``"full"`` mean one batch per pass.
        """
        bs = config.batch_size
        if bs is None or bs == "full":
            return int(n)
        if bs == "auto":
            return int(min(256, n))
        return int(min(max(1, int(bs)), n))

    def estimate_state_bytes(self, d: int, k: int | None = None) -> int:
        """Bytes the persistent statistic will hold, for the budget resolver.

        Zero means "small relative to the data"; subclasses with a known state
        (``k*d`` sums, a ``d*d`` Gramian, ...) override it.  This is a planning
        estimate, not a measurement.
        """
        return 0

    def as_stat(self, **kwargs: Any) -> Any:
        """Expose this optimizer through the sufficient-stat protocol."""
        from .state.sufficient import OptimizerStat

        return OptimizerStat(self, **kwargs)

    def identification_min_batch(self, d: int) -> int:
        """Smallest update that can identify this optimizer's own state.

        A statistic estimated from fewer observations than it has free
        parameters cannot be estimated at all, so a batch below this floor
        spends compute on a number that is mostly noise.  The default (0) says
        the optimizer has no such floor; :class:`GaussianMixture` overrides it
        for the full covariance family.  :func:`fit` raises a smaller request to
        the floor and reports both numbers, so the change is visible rather
        than silent.
        """
        return 0

    # -- the four stages ---------------------------------------------------- #
    def measure(self, batch: Array, state: State) -> State:
        raise NotImplementedError

    def update_state(self, state: State, statistics: State, ctx: UpdateContext) -> State:
        keys = self.state_keys or tuple(statistics.values)
        new = self.updater.update(state, statistics, ctx, keys=keys)
        return new.inherit_pockets(state)

    def on_batch(self, state: State, batch: Array, ctx: UpdateContext) -> State | None:
        """Hook called after ``update_state``; used for finalization accumulators.

        Functional like :meth:`update_state`: return a new ``State`` to replace
        the one in the loop, or ``None`` to mutate in place.  Returning a state
        is the honest form for an accumulator that is rebuilt rather than added
        to; in-place is what the shipped finalizers do, because copying every
        accumulator on every batch would cost more than the accumulation
        (T43).
        """
        return None

    def decode(self, state: State) -> dict[str, Any]:
        raise NotImplementedError

    def objective(self, data: Array, parameters: dict[str, Any]) -> float:
        raise NotImplementedError

    # -- diagnostics -------------------------------------------------------- #
    def describe(self) -> dict[str, Any]:
        return {
            "optimizer": self.name,
            "updater": getattr(self.updater, "name", type(self.updater).__name__),
            "updater_config": self.updater.config() if hasattr(self.updater, "config") else {},
            "config": dict(self.config),
        }


# --------------------------------------------------------------------------- #
# fit driver
# --------------------------------------------------------------------------- #
@dataclass
class FitConfig:
    """Compute budget for :func:`fit`.

    ``batch_size`` accepts three kinds of value:

    ``"auto"`` (the default)
        The optimizer's own resolution wins: its constructor argument if it has
        one, otherwise its fitted law (e-means' ``batch_law``).  Optimizers with
        no opinion get the historical 256-point mini-batch.
    ``None`` or ``"full"``
        One batch per pass, i.e. the conventional rebuild-from-scratch pass.
    ``int``
        That exact batch, clipped to the dataset.

    The default used to be the literal ``256``, which silently overrode the
    e-means batch law: the README's runner example scored 92.43 on 100k x 32 /
    k=64 where the same model at its own law scores 46.99.  ``"auto"`` makes the
    documented behaviour the default; recorded fits pass an explicit size, so no
    ledger number moves.

    ``nepochs`` is the pass budget; the primary metric downstream is objective
    versus *examples processed*, not epochs.
    """

    batch_size: int | str | None = "auto"
    nepochs: int = 5
    seed: int = 0
    shuffle: bool = True
    #: Whether a trailing partial batch is discarded.  NOTE: this default (True)
    #: disagrees with ``stream.iter_blocks``/``stat_fit`` (False) and with
    #: ``GaussianMixture.fit`` (False except for the ``full`` family), so
    #: ``core.fit(gmm, x)`` runs up to one update per epoch fewer than
    #: ``gmm.fit(x)``.  The measured cost of that mismatch was a 120-vs-125
    #: update difference and a 1.40-nat endpoint gap.  Unifying every call site
    #: on one ``plan_batches()`` is queued as T38.
    drop_last: bool = True
    max_updates: int | None = None
    track_memory: bool = False
    eval_every_updates: int | None = None
    #: Measured memory budget for the batch-vs-stream resolution (ONLINE_PLAN
    #: phase 2).  "auto" probes MemAvailable/cgroup at call time; a
    #: :class:`~emaopt.budget.MemoryBudget` overrides it; None disables it.
    budget: Any = "auto"
    #: Raise a batch smaller than the optimizer's ``identification_min_batch``
    #: (see :meth:`StatisticalOptimizer.identification_min_batch`).  Turn it off
    #: to measure the under-identified path deliberately, where the state guard
    #: is what keeps the fit finite.
    enforce_identifiability: bool = True
    #: Stop when the objective changes by at most this much between successive
    #: evaluations (absolute, in the objective's own units).  ``None`` runs the
    #: full budget.  A shipped epoch budget should be a cap, not a cost: GMM's
    #: 60-epoch default otherwise runs 12-15x the reference's converged count at
    #: a NLL that differs by ~0.001 nats, which is a stopping-rule difference,
    #: not a kernel one.
    tol: float | None = None
    #: Stop when the objective changes by at most this fraction of its current
    #: value.  This is the scale-free form of ``tol``; an absolute tolerance is
    #: wrong for k-means, whose MSE scales with the data.  ``None`` disables it.
    rel_tol: float | None = None


@dataclass
class FitResult:
    optimizer: str
    params: dict[str, Any]
    history: list[dict[str, Any]]
    updates: int
    examples_processed: int
    wall_time: float
    cpu_time: float
    peak_memory: int
    objective: float
    extras: dict[str, Any] = field(default_factory=dict)

    def curve(self) -> tuple[Array, Array]:
        """``(examples_processed, objective)`` including the initial point."""
        xs = np.array([h["examples_processed"] for h in self.history], dtype=float)
        ys = np.array([h["objective"] for h in self.history], dtype=float)
        return xs, ys

    def summary(self) -> dict[str, Any]:
        out = {
            "optimizer": self.optimizer,
            "objective": float(self.objective),
            "updates": int(self.updates),
            "examples_processed": int(self.examples_processed),
            "wall_time": float(self.wall_time),
            "cpu_time": float(self.cpu_time),
            "peak_memory": int(self.peak_memory),
        }
        out.update(self.extras)
        return out


def _iter_batches(
    data: Array,
    batch_size: int,
    nepochs: int,
    rng: np.random.Generator,
    shuffle: bool,
    drop_last: bool,
    max_updates: int | None,
) -> Iterator[tuple[int, int, Array]]:
    from .plan import plan_batches

    n = data.shape[0]
    per_epoch = plan_batches(n, batch_size, nepochs,
                             drop_last=drop_last).per_epoch
    emitted = 0
    # One batch per pass *is* the dataset, so permuting it only to copy every
    # row cannot change a full-batch statistic beyond float summation order.
    # The copy is not free: 16.2 ms measured for a 50k x 128 gather, on a fit
    # whose whole training pass is ~36 ms (PCA's 50k row was paying it).
    whole = per_epoch == 1 and batch_size >= n
    for epoch in range(nepochs):
        if whole:
            yield epoch, 0, data
            emitted += 1
            if max_updates is not None and emitted >= max_updates:
                return
            continue
        order = rng.permutation(n) if shuffle else np.arange(n)
        for i in range(per_epoch):
            idx = order[i * batch_size : (i + 1) * batch_size]
            if idx.size == 0:
                continue
            yield epoch, i, data[idx]
            emitted += 1
            if max_updates is not None and emitted >= max_updates:
                return


def evaluate(optimizer: StatisticalOptimizer, state: State, data: Array) -> float:
    """Decode parameters once and score them on ``data``."""
    return float(optimizer.objective(data, optimizer.decode(state)))


def fit(
    optimizer: StatisticalOptimizer,
    data: Array,
    config: FitConfig | None = None,
    *,
    eval_data: Array | None = None,
    callback: Callable[[State, UpdateContext], None] | None = None,
) -> FitResult:
    """Run the measurement -> update -> decode -> objective loop.

    The objective is evaluated once per epoch (and once at the end) on
    ``eval_data`` (defaults to ``data``).  Records carry ``examples_processed``
    so that objective-versus-compute curves can be built without epochs.

    ``data`` is always materialised: ``np.asarray(data, dtype=float64)`` copies a
    memmap or a path-like buffer into RAM before the first batch.  That is the
    price of the row-permuting loop and it is why :func:`emaopt.stream.stat_fit`
    exists -- it folds blocks from a memmap without ever holding the corpus.  A
    caller who passes a shard path here has chosen the batch view deliberately.

    ``callback(state, ctx)`` runs after every update, between the state update
    and the evaluation, so a handler sees exactly the state the objective will
    score.  It used to receive a third ``nan`` argument that nothing read (T44).
    """
    config = config or FitConfig()
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("data must be a 2-D array (n, d)")
    held_out = eval_data is not None
    if eval_data is None:
        eval_data = data
    eval_data = np.asarray(eval_data, dtype=np.float64)

    n = data.shape[0]
    d = int(data.shape[1])
    batch_size = optimizer.resolve_batch_size(config, n, d=d)
    # The optimizer's answer, before the budget resolver may shrink it.  Records
    # keep this (a number) rather than the raw ``FitConfig.batch_size``, which
    # may be the "auto" sentinel.
    requested_batch = int(batch_size)
    plan = None
    if config.budget is not None and config.budget is not False:
        from .budget import MemoryBudget, resolve_compute

        memory = (
            config.budget
            if isinstance(config.budget, MemoryBudget)
            else MemoryBudget.probe()
        )
        k = int(getattr(optimizer, "k", 0) or getattr(optimizer, "n_components", 0) or 0)
        plan = resolve_compute(
            n,
            d,
            k,
            stat_bytes=int(optimizer.estimate_state_bytes(d, k=k or None)),
            budget=memory,
            supports_streaming=bool(getattr(optimizer, "supports_streaming", False)),
            requested_batch=batch_size,
            nepochs=config.nepochs,
        )
        if plan.mode == "stream":
            # The resolver only streams optimizers that declare a streaming
            # form, so this never silently changes Lloyd's full-pass semantics.
            batch_size = max(1, min(batch_size, plan.block_size))
    min_batch = (
        int(optimizer.identification_min_batch(d))
        if config.enforce_identifiability
        else 0
    )
    if min_batch > 0 and batch_size < min_batch:
        # Never larger than the dataset: below that the whole data is one batch
        # and the state guard is the thing that keeps the fit honest.
        batch_size = int(min(min_batch, n))
    from .plan import plan_batches

    batches = plan_batches(n, batch_size, config.nepochs,
                           drop_last=config.drop_last,
                           max_updates=config.max_updates)
    batch_size = batches.batch_size
    per_epoch = batches.per_epoch
    total_steps = batches.total_steps
    eval_every = config.eval_every_updates or per_epoch

    rng = np.random.default_rng(config.seed)
    state = optimizer.initialize(data, config)
    optimizer.begin_fit(state, config, n, batch_size, total_steps)

    if config.track_memory:
        tracemalloc.start()
    wall0, cpu0 = time.perf_counter(), time.process_time()
    history: list[dict[str, Any]] = []
    examples = 0
    updates = 0
    # Evaluation runs inside the timed region (it is part of the loop), so it is
    # accumulated separately: objective-per-wall-clock is the primary outcome and
    # a scoring pass on a GMM costs about as much as a training pass.  Arms that
    # evaluate at different cadences -- or a device arm whose streamed objective
    # costs nothing -- are not comparable against the undivided number.
    eval_s = 0.0
    last_loss = float("nan")
    # Parameters from the in-loop evaluation, kept so the post-loop
    # decode does not repeat an eigendecomposition for the same state.
    eval_params: dict[str, Any] | None = None
    converged_at: int | None = None
    def _score(current: State, done_updates: int, epoch_index: int,
               done_examples: int) -> bool:
        """Decode, score on ``eval_data`` and record; ``True`` when converged."""
        nonlocal eval_params, last_loss, eval_s, converged_at
        _eval0 = time.perf_counter()
        scored = optimizer.decode(current)
        loss = float(optimizer.objective(eval_data, scored))
        eval_s += time.perf_counter() - _eval0
        eval_params = scored
        last_loss = loss
        history.append(
            {
                "update": done_updates,
                "epoch": epoch_index,
                "examples_processed": done_examples,
                "objective": loss,
                "wall_time": time.perf_counter() - wall0,
            }
        )
        if len(history) > 1:
            change = abs(history[-2]["objective"] - loss)
            absolute = config.tol is not None and change <= config.tol
            relative = (config.rel_tol is not None
                        and change <= config.rel_tol * max(abs(loss), 1e-12))
            if absolute or relative:
                # The budget is a cap: if the objective has stopped moving,
                # spending the rest of it buys < ``tol`` nats per point.  The
                # record above is the converged state's own score, so it is
                # reused as the final objective (no extra pass).
                converged_at = done_updates
                return True
        return False

    # An optimizer may offer a compiled epoch runner (a maintainer-side
    # provider, absent from any release install).  It is eligible only when
    # nothing the per-batch loop does
    # would be lost: no per-batch callback, evaluation exactly at epoch
    # boundaries, no max_updates cap that could cut an epoch short, and full
    # batches -- the runner advances ``examples`` a whole epoch at a time.
    runner = None
    candidate = getattr(optimizer, "run_epoch", None)
    if (candidate is not None and callback is None
            and eval_every == per_epoch and config.max_updates is None
            and total_steps == per_epoch * config.nepochs
            and per_epoch * batch_size <= n):
        ready = getattr(optimizer, "epoch_runner_ready", None)
        if ready is not None and ready(n, d, batch_size, per_epoch, total_steps):
            runner = candidate
    try:
        if runner is not None:
            for epoch in range(config.nepochs):
                order = rng.permutation(n) if config.shuffle else np.arange(n)
                step0 = updates
                updates += runner(state, data, order, batch_size, per_epoch,
                                  epoch, total_steps, step0)
                examples += per_epoch * batch_size
                if updates % eval_every == 0 or updates >= total_steps:
                    if _score(state, updates, epoch, examples):
                        break
        else:
            for epoch, i, batch in _iter_batches(
                data, batch_size, config.nepochs, rng, config.shuffle,
                config.drop_last, config.max_updates,
            ):
                ctx = UpdateContext(
                    step=updates,
                    batch_size=int(batch.shape[0]),
                    dataset_size=n,
                    total_steps=total_steps,
                    epoch=epoch,
                    examples_processed=examples,
                )
                statistics = optimizer.measure(batch, state)
                state = optimizer.update_state(state, statistics, ctx)
                state.step = updates + 1
                state = optimizer.on_batch(state, batch, ctx) or state
                examples += int(batch.shape[0])
                updates += 1
                if callback is not None:
                    callback(state, ctx)
                if updates % eval_every == 0 or updates == total_steps:
                    if _score(state, updates, epoch, examples):
                        break
    finally:
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        peak = 0
        if config.track_memory:
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()

    scored_last = bool(history) and history[-1]["update"] == updates
    params = eval_params if (eval_params is not None and scored_last) \
        else optimizer.decode(state)
    if scored_last and not math.isnan(last_loss):
        # The in-loop evaluation that closed the last update already scored
        # exactly this state, so scoring it again here spends a second full-data
        # objective pass on identical parameters.  That pass is not free: a
        # GMM's full-data NLL is about a training pass and PCA's reconstruction
        # objective was 2/3 of its 50k x 128 wall clock, which is why a one-epoch
        # fit used to show double the reference cost.  Only score when the loop
        # did not end on an evaluated update (e.g. ``max_updates`` mid-epoch).
        final = last_loss
        final_eval_s = 0.0
    else:
        # Score the parameters already decoded above rather than calling
        # ``evaluate``, which would decode the same state a second time.  The
        # decode is not free: for PCA it is a ``d x d`` eigendecomposition, so
        # every fit paid for one extra eigh purely to produce the record.
        _final0 = time.perf_counter()
        final = float(optimizer.objective(eval_data, params))
        final_eval_s = time.perf_counter() - _final0
    if not history or history[-1]["objective"] != final:
        history.append(
            {
                "update": updates,
                "epoch": config.nepochs - 1,
                "examples_processed": examples,
                "objective": final,
                "wall_time": wall,
            }
        )
    extras = {
        "n": int(n),
        "d": d,
        "batch_size": batch_size,
        "batch_size_requested": requested_batch,
        "batch_size_input": config.batch_size,
        "identification_min_batch": min_batch,
        "nepochs": config.nepochs,
        "epochs_completed": examples / max(1, n),
        "updates_per_epoch": per_epoch,
        "state_keys": list(state.values),
        "seed": config.seed,
        "compute_plan": None if plan is None else plan.to_dict(),
        # ``wall_time`` is total elapsed *including in-loop evaluation*; the
        # split below is what a compute-per-quality claim should use.
        "eval_time": float(eval_s),
        "train_time": float(max(0.0, wall - eval_s)),
        "final_eval_time": float(final_eval_s),
        "tol": config.tol,
        "rel_tol": config.rel_tol,
        "converged": converged_at is not None,
        "converged_at_update": converged_at,
        # What ``objective`` / ``history[*]["objective"]`` *is*.  A prequential
        # number (device ``eval_mode="stream"``) and a full-data NLL share the
        # key, so a ratio across kinds is meaningless (T39); records carry the
        # kind so a consumer can refuse it.
        "objective_kind": "held_out" if held_out else "full_data",
    }
    for record in history:
        record.setdefault("objective_kind",
                          "held_out" if held_out else "full_data")
    extras.update(optimizer.describe())
    return FitResult(
        optimizer=optimizer.name,
        params=params,
        history=history,
        updates=updates,
        examples_processed=examples,
        wall_time=wall,
        cpu_time=cpu,
        peak_memory=int(peak),
        objective=float(final),
        extras=extras,
    )
