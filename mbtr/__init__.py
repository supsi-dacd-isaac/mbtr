"""Multivariate boosted trees with structured leaf responses."""

from ._splash import splash
from ._version import __version__
from .estimator import MBTClassifier, MBTRegressor
from .losses import LOSS_MAP
from .mbtr import MBT, Tree

__all__ = [
    "MBTClassifier",
    "MBTRegressor",
    "MBT",
    "Tree",
    "LOSS_MAP",
    "splash",
    "__version__",
]
