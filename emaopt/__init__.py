"""emaopt -- EMA-accelerated statistical optimization.

One abstraction, many algorithms::

    measurement -> state update -> parameter decode -> objective -> evidence

Quick start (the P0 reproduction)::

    import numpy as np
    from emaopt import emeans
    x = np.random.default_rng(0).normal(size=(20_000, 32))
    result = emeans(x, k=64, nepochs=5)
    result.centroids, result.labels, result.objective

Swapping the update law is a keyword, not a rewrite::

    from emaopt import GaussianMixture
    from emaopt.state import EMA, Replace, AdaptiveEMA
    batch = GaussianMixture(8, covariance="diag", updater=Replace())
    streams = GaussianMixture(8, covariance="diag", updater=EMA(alpha=0.05))
"""

from .core import (
    DataSpec,
    FitConfig,
    FitResult,
    State,
    StatisticalOptimizer,
    UpdateContext,
    assign,
    evaluate,
    fit,
    logsumexp_rows,
    softmax_rows,
    squared_distances,
)
from .budget import ComputePlan, MemoryBudget, resolve_compute
from .calibration import (
    CalibratedProbabilities,
    PriorCorrector,
    TemperatureScaler,
    auto_calibrate,
    auroc_score,
    average_precision_score,
    base_rate_baseline,
    binary_reliability_curve,
    brier_score,
    calibration_report,
    classwise_expected_calibration_error,
    expected_calibration_error,
    macro_auroc_score,
    macro_average_precision_score,
    maximum_calibration_error,
    negative_log_likelihood,
    reliability_curve,
)
from .pipeline import PipelineConfig, PipelineResult, StreamingPipeline
from .stream import (
    StatFitResult,
    block_steps,
    iter_blocks,
    merge_stats,
    partitioned_fit,
    partitioned_reduce,
    partitioned_stat,
    stat_fit,
    stat_from_blocks,
)
from .threads import (
    ThreadPlan,
    cpu_count,
    physical_cores,
    resolve_threads,
)
from . import analysis, benchmarks, experiments, levels
from .algorithms import (
    ALSFactorization,
    DictionaryLearner,
    EMeans,
    EmeansResult,
    DictionaryStat,
    FastICA,
    GaussianMixture,
    HierarchicalCodebook,
    KMeans,
    LDA,
    LDAStat,
    NMFStat,
    LloydKMeans,
    NMF,
    OjaPCA,
    PCACovariance,
    pddp_seed,
    ProductQuantizer,
    ResidualQuantizer,
    SGDFactorization,
    ScalarQuantizer,
    SparseDiagTied,
    SpectralClustering,
    StatefulICA,
    StreamingRobustPCA,
    SubspaceTracker,
    VectorQuantizer,
    alpha0_from_law,
    batch_size_from_law,
    fuzzy_membership,
    find_neighbors,
    emeans,
    robust_pca,
    umap_embed,
    UMAPConfig,
    UMAPResult,
)
from .gpu_emeans import GpuEmeansConfig, GpuEmeansResult, emeans_gpu_fit
from .basemap import (
    BasemapResult,
    HierarchicalPlacement,
    PushApartResult,
    VoronoiResult,
    hierarchical_place,
    nearest_parent,
    place_around_parents,
    push_apart,
    run_basemap,
    voronoi_cut,
)
from .state import (
    CentroidStat,
    GramianStat,
    OptimizerStat,
    SufficientStat,
    UPDATERS,
    AdaptiveEMA,
    BiasCorrectedEMA,
    ConstantSchedule,
    CosineSchedule,
    EMA,
    EffectiveSampleSchedule,
    InversePowerSchedule,
    InverseSqrtSchedule,
    LinearSchedule,
    Replace,
    RobbinsMonro,
    RunningMean,
    WeightedMean,
    make_schedule,
    make_updater,
)

__version__ = "0.1.0"

__all__ = [
    "DataSpec",
    "FitConfig",
    "FitResult",
    "State",
    "StatisticalOptimizer",
    "UpdateContext",
    "assign",
    "evaluate",
    "fit",
    "logsumexp_rows",
    "softmax_rows",
    "squared_distances",
    "emeans",
    "EmeansResult",
    "EMeans",
    "LloydKMeans",
    "KMeans",
    "batch_size_from_law",
    "alpha0_from_law",
    "GaussianMixture",
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
    "ScalarQuantizer",
    "SparseDiagTied",
    "VectorQuantizer",
    "HierarchicalCodebook",
    "ProductQuantizer",
    "ResidualQuantizer",
    "NMF",
    "DictionaryLearner",
    "FastICA",
    "StatefulICA",
    "PCACovariance",
    "OjaPCA",
    "pddp_seed",
    "LDA",
    "ALSFactorization",
    "SGDFactorization",
    "SpectralClustering",
    "SubspaceTracker",
    "StreamingRobustPCA",
    "robust_pca",
    "MemoryBudget",
    "ComputePlan",
    "resolve_compute",
    "SufficientStat",
    "CentroidStat",
    "GramianStat",
    "OptimizerStat",
    "NMFStat",
    "LDAStat",
    "DictionaryStat",
    "iter_blocks",
    "block_steps",
    "stat_fit",
    "StatFitResult",
    "stat_from_blocks",
    "partitioned_reduce",
    "partitioned_stat",
    "partitioned_fit",
    "merge_stats",
    "ThreadPlan",
    "resolve_threads",
    "cpu_count",
    "physical_cores",
    "StreamingPipeline",
    "PipelineConfig",
    "PipelineResult",
    "Replace",
    "RunningMean",
    "WeightedMean",
    "EMA",
    "BiasCorrectedEMA",
    "RobbinsMonro",
    "AdaptiveEMA",
    "UPDATERS",
    "make_updater",
    "ConstantSchedule",
    "CosineSchedule",
    "LinearSchedule",
    "InverseSqrtSchedule",
    "InversePowerSchedule",
    "EffectiveSampleSchedule",
    "make_schedule",
    "benchmarks",
    "experiments",
    "analysis",
    "levels",
    "UMAPConfig",
    "UMAPResult",
    "find_neighbors",
    "fuzzy_membership",
    "umap_embed",
    "GpuEmeansConfig",
    "GpuEmeansResult",
    "emeans_gpu_fit",
    "hierarchical_place",
    "nearest_parent",
    "place_around_parents",
    "voronoi_cut",
    "push_apart",
    "run_basemap",
    "BasemapResult",
    "HierarchicalPlacement",
    "VoronoiResult",
    "PushApartResult",
    "__version__",
]


def __getattr__(name: str):
    # PEP 562 lazy export: emaopt.gpu imports torch, and the default host path
    # must never pay for it (or break on torch-free installs).
    if name == "egmm_gpu_measure":
        from .gpu import egmm_gpu_measure

        return egmm_gpu_measure
    if name == "egmm_gpu_measure_resident":
        from .gpu import egmm_gpu_measure_resident

        return egmm_gpu_measure_resident
    if name == "egmm_gpu_fit":
        from .gpu import egmm_gpu_fit

        return egmm_gpu_fit
    raise AttributeError(f"module 'emaopt' has no attribute {name!r}")
