"""Replaceable state-update laws.

Every updater answers the same question: given the previous persistent state
and the statistic measured on the current batch, what is the new state?

| updater | rule | what it remembers |
|---|---|---|
| ``Replace`` | ``S <- s`` | nothing (one-pass rebuild; Lloyd/EM reference) |
| ``RunningMean`` | ``S <- (n S + s)/(n+1)`` | all batches, equally weighted |
| ``WeightedMean`` | ``S <- (W S + w s)/(W+w)`` | all batches, weighted by ``w`` |
| ``EMA`` | ``S <- (1-a) S + a s`` | recent batches, decayed at rate ``a`` |
| ``BiasCorrectedEMA`` | EMA warm-started at the first statistic | recent batches, no zero-init transient (alias ``WarmStartEMA``) |
| ``RobbinsMonro`` | EMA with ``a_t = a0/(t0+t)^gamma`` | all batches, vanishing step |
| ``AdaptiveEMA`` | ``a_t = b/(b+n_eff)`` | recent batches, rate set by effective sample count |
| ``MomentumEMA`` | ``S <- S + b V + a(s-S)`` | where the statistic has been moving (``b=0`` is EMA) |
| ``PlateauEMA`` | EMA, rate halved on an objective plateau | nothing extra; reads the batch NLL (H6 counter-example) |
| ``PortfolioEMA`` | EMA at several rates, exponential-weights mixture | which rate has been predicting the batch statistics |

The updater is independent of the algorithm: a GMM, a codebook and an NMF
basis all use the same objects, and only the keys they own change.
"""

from ..core import State, UpdateContext
from .base import StateUpdater
from .affine import AffineUpdater
from .sufficient import (
    CentroidStat,
    GramianStat,
    OptimizerStat,
    SufficientStat,
    merge_stats,
)
from .replace import Replace
from .running_mean import RunningMean
from .weighted_mean import WeightedMean
from .ema import EMA
from .bias_corrected import BiasCorrectedEMA

#: Honest name for the law: it warm-starts the recursion, it does not divide by
#: ``1 - prod(1 - a)``.  The historic name stays so records and imports resolve.
WarmStartEMA = BiasCorrectedEMA
from .stochastic import RobbinsMonro
from .adaptive import AdaptiveEMA
from .momentum import MomentumEMA
from .plateau import PlateauEMA
from .portfolio import PortfolioEMA
from .schedules import (
    ConstantSchedule,
    EMeansStepLaw,
    CosineSchedule,
    AnnealSchedule,
    JitterSchedule,
    LinearSchedule,
    InverseSqrtSchedule,
    InversePowerSchedule,
    EffectiveSampleSchedule,
    SCHEDULES,
    make_schedule,
)

UPDATERS = {
    "replace": Replace,
    "running_mean": RunningMean,
    "weighted_mean": WeightedMean,
    "ema": EMA,
    "bias_corrected": BiasCorrectedEMA,
    "robbins_monro": RobbinsMonro,
    "adaptive": AdaptiveEMA,
    "momentum": MomentumEMA,
    "plateau": PlateauEMA,
    "portfolio": PortfolioEMA,
}


def make_updater(name: str, **kwargs) -> StateUpdater:
    """Construct an updater by name (used by configs, sweeps and experiment records)."""
    try:
        cls = UPDATERS[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown updater {name!r}; available: {sorted(UPDATERS)}") from exc
    return cls(**kwargs)


__all__ = [
    "State",
    "UpdateContext",
    "StateUpdater",
    "AffineUpdater",
    "UPDATERS",
    "make_updater",
    "Replace",
    "RunningMean",
    "WeightedMean",
    "EMA",
    "BiasCorrectedEMA",
    "WarmStartEMA",
    "RobbinsMonro",
    "AdaptiveEMA",
    "MomentumEMA",
    "PlateauEMA",
    "PortfolioEMA",
    "SufficientStat",
    "CentroidStat",
    "GramianStat",
    "OptimizerStat",
    "merge_stats",
    "ConstantSchedule",
    "CosineSchedule",
    "AnnealSchedule",
    "JitterSchedule",
    "LinearSchedule",
    "InverseSqrtSchedule",
    "InversePowerSchedule",
    "EffectiveSampleSchedule",
    "EMeansStepLaw",
    "SCHEDULES",
    "make_schedule",
]
