"""FID / FDR6 / MIND across six representation spaces.

FDR6:  https://github.com/Jiawei-Yang/FD-loss
MIND:  https://github.com/toshas/torch-fidelity (https://arxiv.org/abs/2605.06797)
"""

from .api import compute_metrics, list_bundles, list_extractors
from .extractors import Extractor, ExtractorSpec, build_extractor
from .format import format_results
from .frechet import FrechetAccumulator, frechet_distance
from .mind import mind_from_features
from .stats import list_references, load_features, load_moments, resolve

__all__ = [
    "compute_metrics",
    "format_results",
    "list_extractors",
    "list_bundles",
    "list_references",
    "Extractor",
    "ExtractorSpec",
    "build_extractor",
    "FrechetAccumulator",
    "frechet_distance",
    "mind_from_features",
    "resolve",
    "load_moments",
    "load_features",
]
