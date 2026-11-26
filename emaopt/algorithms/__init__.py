"""Algorithms.

Every module exposes a reference (Replace) and a stateful (EMA/adaptive) variant
of the same estimator, so comparisons share sampler, initialization, decoder and
objective.

Tier 1 (research queue): kmeans/emeans, gmm, sparse_decomp, vq, pq.
Tier 2: nmf, dictionary, ica.
Tier 3 controls: pca, lda, matrix_factorization, spectral.
"""

from .kmeans import KMeans, LloydKMeans, EMeans, batch_size_from_law, alpha0_from_law
from .pddp import pddp_seed
from .smeans import SMeans, smeans, PowerCosineSchedule
from .emeans import emeans, EmeansResult
from .gmm import GaussianMixture
from .sparse_decomp import SparseDiagTied
from .vq import VectorQuantizer, HierarchicalCodebook
from .pq import ProductQuantizer, ResidualQuantizer
from .scalar_quant import ScalarQuantizer, hadamard_matrix
from .nmf import NMF, NMFStat
from .dictionary import DictionaryLearner, DictionaryStat
from .ica import FastICA, FastICAStat, StatefulICA, fastica, fastica_td
from .pca import PCACovariance, OjaPCA
from .lda import LDA, LDAStat
from .matrix_factorization import ALSFactorization, SGDFactorization
from .spectral import SpectralClustering
from .subspace import SubspaceTracker, orthonormalise, top_basis
from .robust_pca import StreamingRobustPCA, robust_pca
from .umap import (
    UMAPConfig,
    UMAPResult,
    find_neighbors,
    fuzzy_membership,
    umap_embed,
)

__all__ = [
    "KMeans",
    "LloydKMeans",
    "EMeans",
    "pddp_seed",
    "SMeans",
    "smeans",
    "PowerCosineSchedule",
    "emeans",
    "EmeansResult",
    "batch_size_from_law",
    "alpha0_from_law",
    "GaussianMixture",
    "SparseDiagTied",
    "VectorQuantizer",
    "HierarchicalCodebook",
    "ProductQuantizer",
    "ResidualQuantizer",
    "ScalarQuantizer",
    "hadamard_matrix",
    "NMF",
    "NMFStat",
    "DictionaryLearner",
    "DictionaryStat",
    "FastICA",
    "FastICAStat",
    "fastica",
    "fastica_td",
    "StatefulICA",
    "PCACovariance",
    "OjaPCA",
    "LDA",
    "LDAStat",
    "ALSFactorization",
    "SGDFactorization",
    "SpectralClustering",
    "SubspaceTracker",
    "StreamingRobustPCA",
    "robust_pca",
    "top_basis",
    "orthonormalise",
    "UMAPConfig",
    "UMAPResult",
    "find_neighbors",
    "fuzzy_membership",
    "umap_embed",
]

