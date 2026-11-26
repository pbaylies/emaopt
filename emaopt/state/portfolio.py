"""PortfolioEMA: online selection among step sizes instead of a fitted law.

Every schedule so far commits to one step-size law before the run.  This updater
does the opposite: it runs ``m`` EMA states at fixed, log-spaced rates in
parallel (they share the measurement, so the extra cost is ``m`` array updates,
not ``m`` E-steps) and lets exponential weights decide how much of each state
the answer contains.

The loss is the expert's *one-step-ahead* prediction error of the incoming batch
statistic, relative to the statistic's own scale::

    L_i(t) = ||S_i(t-1) - s_t||^2 / ||s_t||^2
    w_i(t) ∝ exp(-eta * sum_{u<=t} L_i(u))

so an expert whose state consistently anticipated the next batch's counts/sums
keeps weight, and one that lags loses it.  The reported step is the weight of
the experts, which is what makes this a *selection* rule rather than a tunable:
there is no parameter that has to be right about the problem, only an ensemble
that has to contain a good rate.

Two honest caveats, both measured rather than argued:

* the loss is computed against the statistic being damped, so this is a
  feedback loop in the H6 sense -- the guard says such a loop is a risk, not
  that it is useless, and the schedule sweep reports which it is here;
* the returned state is a convex combination of the experts' states.  That is
  exact for counts and sums (both are linear in the batch) and is a mixture
  moment for ``sq``/``scatter``; it is a moment, not a model average.
"""

from __future__ import annotations

import numpy as np

from ..core import State, UpdateContext
from .base import StateUpdater


class PortfolioEMA(StateUpdater):
    name = "portfolio"

    def __init__(
        self,
        alphas: tuple[float, ...] = (0.04, 0.08, 0.16, 0.32),
        *,
        eta: float = 1.0,
        clip: float = 4.0,
    ) -> None:
        alphas = tuple(float(a) for a in alphas)
        if not alphas:
            raise ValueError("alphas must not be empty")
        if any(not 0.0 < a <= 1.0 for a in alphas):
            raise ValueError("every alpha must lie in (0, 1]")
        if eta <= 0.0:
            raise ValueError("eta must be positive")
        self.alphas = alphas
        self.eta = float(eta)
        self.clip = float(clip)
        self._weights = np.full(len(alphas), 1.0 / len(alphas))

    def alpha(self, ctx: UpdateContext) -> float:
        """The mixture's effective rate, for diagnostics."""
        return float(np.dot(self._weights, np.asarray(self.alphas)))

    def update(
        self,
        previous: State,
        statistics: State,
        ctx: UpdateContext,
        keys: tuple[str, ...] | None = None,
    ) -> State:
        keys = keys or tuple(statistics.values)
        m = len(self.alphas)
        experts = previous.meta.get("_portfolio_experts")
        if experts is None or len(experts) != m:
            experts = [
                {k: np.array(v, copy=True) for k, v in previous.values.items()}
                for _ in range(m)
            ]
        cumulative = np.asarray(
            previous.meta.get("_portfolio_loss", np.zeros(m)), dtype=np.float64
        )

        updated: list[dict[str, np.ndarray]] = []
        losses = np.zeros(m, dtype=np.float64)
        for i, rate in enumerate(self.alphas):
            values = {k: np.array(v, copy=True) for k, v in experts[i].items()}
            for key in keys:
                statistic = self._stat(statistics, key)
                if statistic is None:
                    continue
                previous_value = values.get(key)
                if previous_value is None or np.shape(previous_value) != np.shape(statistic):
                    values[key] = np.array(statistic, copy=True)
                    continue
                denom = float(np.sum(statistic * statistic)) + 1e-12
                losses[i] += float(np.sum((previous_value - statistic) ** 2)) / denom
                values[key] = (1.0 - rate) * previous_value + rate * statistic
            updated.append(values)

        losses = np.clip(losses, 0.0, self.clip)
        cumulative = cumulative + losses
        logits = -self.eta * cumulative
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()
        self._weights = weights

        values = dict(previous.values)
        for key in keys:
            parts = [u[key] for u in updated if key in u]
            if len(parts) != m:
                continue
            mixed = np.zeros_like(np.asarray(parts[0], dtype=np.float64))
            for weight, part in zip(weights, parts):
                mixed += weight * np.asarray(part, dtype=np.float64)
            values[key] = mixed

        meta = dict(previous.meta)
        meta["_portfolio_experts"] = updated
        meta["_portfolio_loss"] = cumulative
        meta["_portfolio_weights"] = weights
        return State(values, previous.step, meta)

    def config(self) -> dict:
        return {
            "name": self.name,
            "alphas": list(self.alphas),
            "eta": self.eta,
        }
