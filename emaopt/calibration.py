"""Post-hoc calibration and evaluation for predictive probabilities.

The important distinction is what information is available:

* ``TemperatureScaler`` is supervised.  It fits one temperature on held-out
  ``(probabilities, labels)`` pairs by minimizing negative log-likelihood.  It
  changes confidence but not the argmax, so accuracy is preserved.
* ``PriorCorrector`` is unsupervised.  It needs a target marginal (for example,
  known cluster proportions under label shift) and rescales probabilities so
  their mean matches it.  It is the Saerens-style prior correction used for
  mixture posteriors.
* ``CalibratedProbabilities`` is the common interface.  Its default
  ``method="auto"`` chooses temperature scaling when held-out labels are
  supplied, prior correction when only a target prior is supplied, and raises
  otherwise.  It never calibrates against a model's own argmax.
* ``calibration_report`` is the evaluation surface.  It reports proper scores,
  the base-rate control, AUROC/average precision, and a reliability curve.  A
  calibration number without a discrimination number and a constant baseline is
  not a result.

Unsupervised mixture responsibilities are *not* automatically calibrated by
these tools.  Temperature scaling without labels has no calibration target, and
prior correction only fixes marginal proportions, not per-example confidence.
Calibration is therefore opt-in; the estimators themselves do not silently
calibrate their responsibilities.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_EPS = 1e-300
_BINNINGS = {"equal_width", "quantile", "equal_mass"}


def _as_probabilities(probabilities: Any) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("probabilities must be a 2-D array with at least two columns")
    if p.shape[0] == 0:
        raise ValueError("probabilities must contain at least one row")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("probabilities must be finite and non-negative")
    total = p.sum(axis=1, keepdims=True)
    if np.any(total <= 0.0):
        raise ValueError("each probability row must have positive mass")
    return p / total


def _as_labels(labels: Any, n: int, k: int) -> np.ndarray:
    y = np.asarray(labels)
    if y.shape != (n,):
        raise ValueError(f"labels must have shape ({n},), got {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integers")
    y = y.astype(np.int64)
    if np.any(y < 0) or np.any(y >= k):
        raise ValueError(f"labels must be in [0, {k})")
    return y


def _as_binary_labels(labels: Any, n: int) -> np.ndarray:
    y = np.asarray(labels)
    if y.dtype == np.bool_:
        y = y.astype(np.int64)
    return _as_labels(y, n, 2)


def _as_scores(scores: Any, *, n: int | None = None) -> np.ndarray:
    """Return a 1-D score vector, accepting binary two-column probabilities."""
    value = np.asarray(scores, dtype=np.float64)
    if value.ndim == 2:
        if value.shape[1] != 2:
            raise ValueError(
                "binary scores must be one-dimensional or have exactly two columns"
            )
        value = value[:, 1]
    if value.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if value.size == 0:
        raise ValueError("scores must contain at least one value")
    if not np.all(np.isfinite(value)):
        raise ValueError("scores must be finite")
    if n is not None and value.size != n:
        raise ValueError(f"scores must have length {n}, got {value.size}")
    return value


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    z = np.exp(z)
    return z / np.maximum(z.sum(axis=1, keepdims=True), _EPS)


def _normalise_vector(x: Any) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    if v.size < 2 or not np.all(np.isfinite(v)) or np.any(v < 0.0):
        raise ValueError("prior must be a finite, non-negative vector with >= 2 entries")
    total = float(v.sum())
    if total <= 0.0:
        raise ValueError("prior must have positive mass")
    return v / total


def negative_log_likelihood(probabilities: Any, labels: Any) -> float:
    """Mean multiclass negative log-likelihood in nats."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    return float(-np.log(np.maximum(p[np.arange(p.shape[0]), y], _EPS)).mean())


def brier_score(probabilities: Any, labels: Any) -> float:
    """Mean multiclass Brier score (lower is better)."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    onehot = np.zeros_like(p)
    onehot[np.arange(p.shape[0]), y] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def _normalise_binning(binning: str) -> str:
    if binning not in _BINNINGS:
        raise ValueError("binning must be 'equal_width', 'quantile' or 'equal_mass'")
    return "quantile" if binning == "equal_mass" else binning


def _bin_indices(
    confidence: np.ndarray, n_bins: int, binning: str
) -> tuple[np.ndarray, int]:
    """Return bin indices and the number of non-empty-width bins."""
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    if confidence.size == 0:
        raise ValueError("at least one confidence value is required")
    binning = _normalise_binning(binning)
    if binning == "equal_width":
        boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        # Equal-mass bins keep resolution when probabilities are saturated.
        boundaries = np.unique(
            np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
        )
    if boundaries.size < 2:
        return np.zeros(confidence.size, dtype=np.int64), 1
    which = np.digitize(confidence, boundaries[1:-1], right=False)
    return np.clip(which, 0, boundaries.size - 2), int(boundaries.size - 1)


def _curve_rows(
    confidence: np.ndarray, outcome: np.ndarray, n_bins: int, binning: str
) -> tuple[float, float, list[dict[str, float]]]:
    which, actual_bins = _bin_indices(confidence, n_bins, binning)
    total = float(confidence.size)
    ece = 0.0
    mce = 0.0
    rows: list[dict[str, float]] = []
    for b in range(actual_bins):
        mask = which == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_confidence = float(confidence[mask].mean())
        observed_rate = float(outcome[mask].mean())
        gap = abs(observed_rate - mean_confidence)
        ece += (count / total) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "bin": b,
                "count": count,
                "mean_probability": mean_confidence,
                "observed_rate": observed_rate,
                "gap": gap,
            }
        )
    return float(ece), float(mce), rows


def reliability_curve(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "quantile",
) -> dict[str, Any]:
    """Top-label reliability curve with counts, ECE and MCE.

    ``binning="quantile"`` is the default because equal-width bins collapse to
    a single populated bin when GMM/eGMM probabilities are near 0 or 1.  Use
    ``binning="equal_width"`` to reproduce the standard textbook ECE.
    """
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    ece, mce, rows = _curve_rows(confidence, correct, n_bins, binning)
    return {
        "n": int(confidence.size),
        "n_bins": int(n_bins),
        "binning": _normalise_binning(binning),
        "nonempty_bins": len(rows),
        "ece": ece,
        "mce": mce,
        "bins": rows,
    }


def binary_reliability_curve(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "quantile",
) -> dict[str, Any]:
    """Reliability curve for the positive class of a binary score."""
    scores = _as_scores(probabilities)
    y = _as_binary_labels(labels, scores.size)
    ece, mce, rows = _curve_rows(scores, y.astype(np.float64), n_bins, binning)
    return {
        "n": int(scores.size),
        "n_bins": int(n_bins),
        "binning": _normalise_binning(binning),
        "nonempty_bins": len(rows),
        "positive_rate": float(y.mean()),
        "ece": ece,
        "mce": mce,
        "bins": rows,
    }


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int, binning: str = "equal_width") -> float:
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    value, _, _ = _curve_rows(confidence, correct, n_bins, binning)
    return value


def expected_calibration_error(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float:
    """Top-label expected calibration error (ECE), in [0, 1].

    Equal-width bins are the standard default.  Use ``binning="quantile"`` (or
    ``"equal_mass"``) when predicted probabilities are concentrated near 0 or 1;
    that is what :func:`calibration_report` does by default.
    """
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    return _ece(p, y, n_bins, binning)


def classwise_expected_calibration_error(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float:
    """Balanced classwise ECE: the mean one-vs-rest ECE over all classes.

    Each class receives equal weight, so a rare class is not hidden by a
    dominant class.  This is a diagnostic, not a replacement for top-label ECE.
    """
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    values = []
    for c in range(p.shape[1]):
        outcome = (y == c).astype(np.float64)
        value, _, _ = _curve_rows(p[:, c], outcome, n_bins, binning)
        values.append(value)
    return float(np.mean(values))


def maximum_calibration_error(
    probabilities: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float:
    """Largest top-label reliability gap; useful as a worst-bin audit."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    confidence = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    _, value, _ = _curve_rows(confidence, correct, n_bins, binning)
    return value


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks in sorted-value order, with ties sharing their mean rank."""
    n = int(values.size)
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    average = (starts + 1 + ends) / 2.0
    group = np.repeat(np.arange(starts.size), ends - starts)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = average[group]
    return ranks


def auroc_score(probabilities: Any, labels: Any) -> float:
    """ROC AUC for a binary score.

    ``probabilities`` may be a 1-D positive-class score or a two-column
    probability matrix.  Ties receive half credit.  A single-class sample has no
    ranking question and returns 0.5.
    """
    scores = _as_scores(probabilities)
    y = _as_binary_labels(labels, scores.size)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = _average_ranks(scores)
    return float(
        (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    )


def average_precision_score(probabilities: Any, labels: Any) -> float:
    """Average precision for a binary score; ties are handled as one threshold."""
    scores = _as_scores(probabilities)
    y = _as_binary_labels(labels, scores.size)
    n_pos = int(y.sum())
    if n_pos == 0:
        return 0.0
    if n_pos == y.size:
        return 1.0
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_y = y[order]
    starts = np.r_[0, np.flatnonzero(ordered_scores[1:] != ordered_scores[:-1]) + 1]
    ends = np.r_[starts[1:], y.size]
    true_positives = np.cumsum(ordered_y)[ends - 1].astype(np.float64)
    precision = true_positives / ends.astype(np.float64)
    recall = true_positives / float(n_pos)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def macro_auroc_score(probabilities: Any, labels: Any) -> float:
    """Mean one-vs-rest AUROC, useful for multiclass reports."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    values = []
    for c in range(p.shape[1]):
        outcome = (y == c).astype(np.int64)
        if outcome.min() != outcome.max():
            values.append(auroc_score(p[:, c], outcome))
    return float(np.mean(values)) if values else 0.5


def macro_average_precision_score(probabilities: Any, labels: Any) -> float:
    """Mean one-vs-rest average precision, useful for multiclass reports."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels, p.shape[0], p.shape[1])
    values = []
    for c in range(p.shape[1]):
        outcome = (y == c).astype(np.int64)
        if outcome.min() != outcome.max():
            values.append(average_precision_score(p[:, c], outcome))
    return float(np.mean(values)) if values else 0.0


def base_rate_baseline(
    labels: Any, *, n_classes: int | None = None
) -> np.ndarray:
    """Constant class-prior predictor fitted from labels.

    Use calibration-split labels for an honest deployment baseline.  Passing
    evaluation labels gives an oracle-marginal reference, which is still useful
    as a control but should be labelled as such.
    """
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if y.dtype == np.bool_:
        y = y.astype(np.int64)
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be integers")
    if y.size == 0:
        raise ValueError("labels must contain at least one value")
    y = y.astype(np.int64)
    if np.any(y < 0):
        raise ValueError("labels must be non-negative")
    if n_classes is None:
        n_classes = int(y.max()) + 1
    if n_classes < 2 or np.any(y >= n_classes):
        raise ValueError("labels must be in [0, n_classes)")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError("labels must contain at least one value")
    return counts / total


def _constant_probabilities(prior: Any, n: int, k: int) -> np.ndarray:
    value = _normalise_vector(prior)
    if value.size != k:
        raise ValueError("prior length must match the probability columns")
    return np.tile(value, (n, 1))


def _golden_section(function, lo: float, hi: float, iterations: int = 60) -> float:
    """Minimize a scalar unimodal-ish function on [lo, hi] without SciPy."""
    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc, fd = function(c), function(d)
    for _ in range(iterations):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = function(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = function(d)
    return 0.5 * (lo + hi)


class TemperatureScaler:
    """Single-temperature calibration fitted on held-out labels.

    ``transform`` computes ``softmax(log(p) / T)``.  The temperature is chosen
    by minimizing held-out NLL over a log-spaced grid plus a golden-section
    refinement.  ``T > 1`` softens an overconfident model; ``T < 1`` sharpens an
    underconfident one.  The argmax is unchanged, so calibration cannot improve
    ranking or accuracy.

    NLL is the default criterion because it is a proper scoring rule.  ``ece``
    is available for a deployment that is specifically judged on top-label
    calibration, but it can improve ECE while worsening NLL.
    """

    def __init__(
        self,
        *,
        min_temperature: float = 0.05,
        max_temperature: float = 1_000.0,
        grid: int = 81,
        refine: bool = True,
        criterion: str = "nll",
        ece_bins: int = 15,
    ) -> None:
        if min_temperature <= 0.0 or max_temperature <= min_temperature:
            raise ValueError("temperature bounds must satisfy 0 < min < max")
        if grid < 3:
            raise ValueError("grid must be at least 3")
        if criterion not in {"nll", "ece"}:
            raise ValueError("criterion must be 'nll' or 'ece'")
        if ece_bins < 1:
            raise ValueError("ece_bins must be positive")
        self.min_temperature = float(min_temperature)
        self.max_temperature = float(max_temperature)
        self.grid = int(grid)
        self.refine = bool(refine)
        self.criterion = criterion
        self.ece_bins = int(ece_bins)
        self.temperature_: float | None = None
        self.nll_: float | None = None
        self.ece_: float | None = None

    def fit(self, probabilities: Any, labels: Any) -> "TemperatureScaler":
        p = _as_probabilities(probabilities)
        y = _as_labels(labels, p.shape[0], p.shape[1])
        logp = np.log(np.maximum(p, _EPS))
        rows = np.arange(p.shape[0])

        def nll(log_temperature: float) -> float:
            logits = logp / math.exp(log_temperature)
            logits -= logits.max(axis=1, keepdims=True)
            log_norm = np.log(np.maximum(np.exp(logits).sum(axis=1), _EPS))
            return float((log_norm - logits[rows, y]).mean())

        def ece(log_temperature: float) -> float:
            logits = logp / math.exp(log_temperature)
            return _ece(_softmax(logits), y, self.ece_bins)

        objective = nll if self.criterion == "nll" else ece
        lo = math.log(self.min_temperature)
        hi = math.log(self.max_temperature)
        candidates = np.linspace(lo, hi, self.grid)
        losses = np.array([objective(float(v)) for v in candidates])
        best = int(losses.argmin())
        log_t = float(candidates[best])
        # ECE is piecewise-constant; golden-section refinement assumes a smooth
        # objective, so only refine the NLL fit.
        if self.refine and self.criterion == "nll" and 0 < best < self.grid - 1:
            step = float(candidates[1] - candidates[0])
            log_t = _golden_section(
                objective, max(lo, log_t - step), min(hi, log_t + step)
            )
        self.temperature_ = float(math.exp(log_t))
        self.nll_ = float(nll(log_t))
        self.ece_ = float(ece(log_t))
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        if self.temperature_ is None:
            raise RuntimeError("TemperatureScaler must be fitted before transform")
        p = _as_probabilities(probabilities)
        return _softmax(np.log(np.maximum(p, _EPS)) / self.temperature_)

    def fit_transform(self, probabilities: Any, labels: Any) -> np.ndarray:
        return self.fit(probabilities, labels).transform(probabilities)


class PriorCorrector:
    """Rescale probabilities so their mean matches a target marginal.

    This is prior/proportion correction, not confidence calibration.  It is the
    right tool when eGMM's component proportions are known or externally
    constrained, and it works without labels.  ``target_prior`` may be supplied
    at construction time or to :meth:`fit`.
    """

    def __init__(
        self,
        target_prior: Any | None = None,
        *,
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> None:
        if max_iter < 1:
            raise ValueError("max_iter must be positive")
        self.target_prior = (
            None if target_prior is None else _normalise_vector(target_prior)
        )
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.target_prior_: np.ndarray | None = None
        self.source_prior_: np.ndarray | None = None
        self.correction_source_: np.ndarray | None = None
        self.iterations_ = 0

    def fit(
        self, probabilities: Any, target_prior: Any | None = None
    ) -> "PriorCorrector":
        p = _as_probabilities(probabilities)
        target = target_prior if target_prior is not None else self.target_prior
        if target is None:
            raise ValueError("a target_prior is required for prior correction")
        target = _normalise_vector(target)
        if target.size != p.shape[1]:
            raise ValueError("target_prior length must match the probability columns")
        # Find a correction source ``s`` such that the adjusted mean equals
        # ``target``.  Updating ``s`` by the multiplicative mean error converges
        # to that fixed point; iterating on the observed mean itself does not.
        initial_prior = np.maximum(p.mean(axis=0), _EPS)
        source = initial_prior.copy()
        safe_target = np.maximum(target, _EPS)
        for iteration in range(1, self.max_iter + 1):
            q = p * (target / source)[None, :]
            q /= np.maximum(q.sum(axis=1, keepdims=True), _EPS)
            observed = q.mean(axis=0)
            if float(np.max(np.abs(observed - target))) <= self.tol:
                break
            source *= np.maximum(observed / safe_target, 1e-6)
        self.target_prior_ = target
        self.source_prior_ = initial_prior
        self.correction_source_ = np.maximum(source, _EPS)
        self.iterations_ = iteration
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        if self.target_prior_ is None or self.correction_source_ is None:
            raise RuntimeError("PriorCorrector must be fitted before transform")
        p = _as_probabilities(probabilities)
        if p.shape[1] != self.target_prior_.size:
            raise ValueError("probability columns do not match the fitted prior")
        q = p * (self.target_prior_ / self.correction_source_)[None, :]
        return q / np.maximum(q.sum(axis=1, keepdims=True), _EPS)


class CalibratedProbabilities:
    """One fit/transform interface for supervised or prior calibration.

    ``method="auto"`` (the default) chooses temperature scaling when held-out
    labels are supplied and prior correction when only a target prior is
    supplied.  If neither is available it raises: responsibilities cannot be
    calibrated against their own argmax.  The selected method and the reason are
    recorded in ``method_`` and ``reason_``.

    ``method="temperature"`` requires labels; ``method="prior"`` requires a
    target prior and no labels.  The fitted calibrator is available as
    ``calibrator_`` for inspection.
    """

    def __init__(
        self,
        method: str = "auto",
        *,
        target_prior: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if method not in {"auto", "temperature", "prior"}:
            raise ValueError("method must be 'auto', 'temperature' or 'prior'")
        self.method = method
        self.target_prior = target_prior
        self.kwargs = dict(kwargs)
        self.method_: str | None = None
        self.reason_: str | None = None
        self.calibrator_: TemperatureScaler | PriorCorrector | None = None

    def fit(
        self, probabilities: Any, labels: Any | None = None
    ) -> "CalibratedProbabilities":
        if self.method == "auto":
            if labels is not None:
                selected = "temperature"
                reason = "held-out labels provided"
            elif self.target_prior is not None:
                selected = "prior"
                reason = "target prior provided; no labels"
            else:
                raise ValueError(
                    "auto calibration needs held-out labels for temperature "
                    "scaling or target_prior for prior correction; "
                    "responsibilities are not self-calibrating"
                )
        else:
            selected = self.method
            reason = "explicit method"

        if selected == "temperature":
            if labels is None:
                raise ValueError("temperature calibration requires held-out labels")
            self.calibrator_ = TemperatureScaler(**self.kwargs).fit(
                probabilities, labels
            )
        else:
            if self.target_prior is None:
                raise ValueError("prior calibration requires target_prior")
            self.calibrator_ = PriorCorrector(self.target_prior, **self.kwargs).fit(
                probabilities
            )
        self.method_ = selected
        self.reason_ = reason
        return self

    def transform(self, probabilities: Any) -> np.ndarray:
        if self.calibrator_ is None:
            raise RuntimeError("CalibratedProbabilities must be fitted before transform")
        return self.calibrator_.transform(probabilities)

    def fit_transform(
        self, probabilities: Any, labels: Any | None = None
    ) -> np.ndarray:
        return self.fit(probabilities, labels).transform(probabilities)


def auto_calibrate(
    probabilities: Any,
    labels: Any | None = None,
    *,
    target_prior: Any | None = None,
    **kwargs: Any,
) -> CalibratedProbabilities:
    """Fit the appropriate calibrator with one call.

    Labels select NLL-fitted temperature scaling.  Without labels, a target
    prior selects prior correction.  With neither, this raises rather than
    returning an identity map that looks calibrated.
    """
    return CalibratedProbabilities(
        method="auto", target_prior=target_prior, **kwargs
    ).fit(probabilities, labels)


def _metric_suite(
    probabilities: np.ndarray, labels: np.ndarray, n_bins: int, binning: str
) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(
            probabilities, labels, n_bins=n_bins, binning=binning
        ),
        "classwise_ece": classwise_expected_calibration_error(
            probabilities, labels, n_bins=n_bins, binning=binning
        ),
        "mce": maximum_calibration_error(
            probabilities, labels, n_bins=n_bins, binning=binning
        ),
        "brier": brier_score(probabilities, labels),
        "nll": negative_log_likelihood(probabilities, labels),
    }


def _discrimination(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    if probabilities.shape[1] == 2:
        return {
            "kind": "binary",
            "auroc": auroc_score(probabilities, labels),
            "average_precision": average_precision_score(probabilities, labels),
            "positive_rate": float(np.asarray(labels).mean()),
        }
    return {
        "kind": "macro_ovr",
        "auroc": macro_auroc_score(probabilities, labels),
        "average_precision": macro_average_precision_score(probabilities, labels),
    }


def calibration_report(
    uncalibrated: Any,
    calibrated: Any,
    labels: Any,
    *,
    n_bins: int = 15,
    binning: str = "quantile",
    baseline: Any | None = None,
    baseline_labels: Any | None = None,
) -> dict[str, Any]:
    """Before/after report with scores, discrimination and a base-rate control.

    ``binning="quantile"`` is the default so a saturated confidence distribution
    does not collapse to one equal-width bin.  ``baseline`` may be a prior vector
    or a full probability matrix.  If it is omitted, the baseline is the
    constant class prior from ``baseline_labels`` when supplied, otherwise from
    the evaluation labels; the latter is an oracle-marginal reference and is
    labelled ``"evaluation_labels"`` in ``settings``.
    """
    before = _as_probabilities(uncalibrated)
    after = _as_probabilities(calibrated)
    y = _as_labels(labels, before.shape[0], before.shape[1])
    if after.shape != before.shape:
        raise ValueError("uncalibrated and calibrated probabilities must have equal shape")
    binning = _normalise_binning(binning)
    n, k = before.shape

    if baseline is not None:
        value = np.asarray(baseline, dtype=np.float64)
        if value.ndim == 1:
            prior = _normalise_vector(value)
            base = _constant_probabilities(prior, n, k)
            baseline_source = "provided_prior"
        elif value.ndim == 2:
            base = _as_probabilities(value)
            if base.shape != before.shape:
                raise ValueError("baseline probabilities must match the report shape")
            prior = base.mean(axis=0)
            baseline_source = "provided_probabilities"
        else:
            raise ValueError("baseline must be a prior vector or a probability matrix")
    else:
        source_labels = y if baseline_labels is None else baseline_labels
        prior = base_rate_baseline(source_labels, n_classes=k)
        base = _constant_probabilities(prior, n, k)
        baseline_source = (
            "evaluation_labels" if baseline_labels is None else "baseline_labels"
        )

    report: dict[str, Any] = {
        "before": _metric_suite(before, y, n_bins, binning),
        "after": _metric_suite(after, y, n_bins, binning),
        "baseline": _metric_suite(base, y, n_bins, binning),
        "discrimination": {
            "before": _discrimination(before, y),
            "after": _discrimination(after, y),
            "baseline": _discrimination(base, y),
        },
        "settings": {
            "n": int(n),
            "classes": int(k),
            "n_bins": int(n_bins),
            "binning": binning,
            "baseline_source": baseline_source,
            "baseline_prior": [float(v) for v in prior],
        },
    }
    if k == 2:
        report["reliability"] = {
            "before": reliability_curve(
                before, y, n_bins=n_bins, binning=binning
            ),
            "after": reliability_curve(
                after, y, n_bins=n_bins, binning=binning
            ),
            "baseline": reliability_curve(
                base, y, n_bins=n_bins, binning=binning
            ),
            "binary_positive": {
                "before": binary_reliability_curve(
                    before, y, n_bins=n_bins, binning=binning
                ),
                "after": binary_reliability_curve(
                    after, y, n_bins=n_bins, binning=binning
                ),
                "baseline": binary_reliability_curve(
                    base, y, n_bins=n_bins, binning=binning
                ),
            },
        }
    else:
        report["reliability"] = {
            "before": reliability_curve(
                before, y, n_bins=n_bins, binning=binning
            ),
            "after": reliability_curve(
                after, y, n_bins=n_bins, binning=binning
            ),
            "baseline": reliability_curve(
                base, y, n_bins=n_bins, binning=binning
            ),
        }
    return report


__all__ = [
    "CalibratedProbabilities",
    "PriorCorrector",
    "TemperatureScaler",
    "auto_calibrate",
    "auroc_score",
    "average_precision_score",
    "base_rate_baseline",
    "binary_reliability_curve",
    "brier_score",
    "calibration_report",
    "classwise_expected_calibration_error",
    "expected_calibration_error",
    "macro_auroc_score",
    "macro_average_precision_score",
    "maximum_calibration_error",
    "negative_log_likelihood",
    "reliability_curve",
]
