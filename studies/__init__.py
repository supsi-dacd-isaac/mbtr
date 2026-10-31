"""Reproducible scientific studies for MBTR.

This package is intentionally kept outside :mod:`mbtr`: it is research
infrastructure, not part of the installed estimator API.
"""

from .adapters import IndependentLabelsClassifier, IndependentOutputsRegressor, make_estimator
from .catalog import PublishedScore, find_protocol, find_published_scores, load_published_scores, validate_catalog
from .metrics import (
    average_relative_rmse,
    binary_cross_entropy,
    classification_metrics,
    example_f1,
    example_jaccard,
    global_rmse,
    hamming_loss,
    mean_target_r2,
    regression_metrics,
    subset_accuracy,
    target_relative_rmse,
)

__all__ = [
    "IndependentLabelsClassifier",
    "IndependentOutputsRegressor",
    "PublishedScore",
    "average_relative_rmse",
    "binary_cross_entropy",
    "classification_metrics",
    "example_f1",
    "example_jaccard",
    "find_protocol",
    "find_published_scores",
    "global_rmse",
    "hamming_loss",
    "load_published_scores",
    "make_estimator",
    "mean_target_r2",
    "regression_metrics",
    "subset_accuracy",
    "target_relative_rmse",
    "validate_catalog",
]
