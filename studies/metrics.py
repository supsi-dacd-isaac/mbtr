"""Metric definitions used by the scientific study runner.

The implementations live here, rather than relying on library defaults, so a
result file identifies an exact and testable mathematical definition.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _as_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or not array.size:
        raise ValueError(f"{name} must be a non-empty one- or two-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _paired_matrices(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[FloatArray, FloatArray]:
    target = _as_matrix(y_true, name="y_true")
    prediction = _as_matrix(y_pred, name="y_pred")
    if target.shape != prediction.shape:
        raise ValueError(f"y_true and y_pred must have the same shape; got {target.shape} and {prediction.shape}.")
    return target, prediction


def _binary_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = _as_matrix(value, name=name)
    if np.any((array != 0.0) & (array != 1.0)):
        raise ValueError(f"{name} must contain only binary values 0 and 1.")
    return array


def global_rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return ``sqrt(mean((y_true - y_pred) ** 2))`` over all elements."""

    target, prediction = _paired_matrices(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(target - prediction))))


def mean_target_r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return uniform-average coefficient of determination across outputs.

    Constant targets follow scikit-learn's finite convention: a perfect
    prediction scores one and any imperfect prediction scores zero.
    """

    target, prediction = _paired_matrices(y_true, y_pred)
    residual = np.sum(np.square(target - prediction), axis=0)
    total = np.sum(np.square(target - np.mean(target, axis=0)), axis=0)
    score = np.divide(residual, total, out=np.zeros_like(residual), where=total != 0.0)
    score = 1.0 - score
    score = np.where(total == 0.0, np.asarray(residual == 0.0, dtype=np.float64), score)
    return float(np.mean(score))


def target_relative_rmse(y_train: ArrayLike, y_true: ArrayLike, y_pred: ArrayLike) -> FloatArray:
    """Return one train-mean-normalized relative RMSE per output.

    For target ``j`` this is ``sqrt(SSE_model / SSE_train_mean)`` on the test
    rows, where ``SSE_train_mean`` predicts every test row with the target's
    training-set mean. This is the Mulan multi-target-regression definition.
    """

    training = _as_matrix(y_train, name="y_train")
    target, prediction = _paired_matrices(y_true, y_pred)
    if training.shape[1] != target.shape[1]:
        raise ValueError("y_train and y_true must have the same number of outputs.")
    numerator = np.sum(np.square(target - prediction), axis=0)
    denominator = np.sum(np.square(target - np.mean(training, axis=0)), axis=0)
    if np.any(denominator <= 0.0):
        indices = np.flatnonzero(denominator <= 0.0).tolist()
        raise ValueError(f"Relative RMSE is undefined for constant test target(s): {indices}.")
    return np.sqrt(numerator / denominator)


def average_relative_rmse(y_train: ArrayLike, y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the unweighted mean of per-output relative RMSE values."""

    return float(np.mean(target_relative_rmse(y_train, y_true, y_pred)))


def mean_fold_average_relative_rmse(
    folds: Iterable[tuple[ArrayLike, ArrayLike, ArrayLike]],
) -> float:
    """Average fold-level aRRMSE without pooling fold predictions."""

    values = [average_relative_rmse(y_train, y_true, y_pred) for y_train, y_true, y_pred in folds]
    if not values:
        raise ValueError("At least one fold is required.")
    return float(np.mean(values))


def binary_cross_entropy(y_true: ArrayLike, y_probability: ArrayLike, *, clip: float = 1e-7) -> float:
    """Return mean elementwise binary cross-entropy.

    ``clip=1e-7`` reproduces the public SketchBoost evaluation script.
    """

    target = _binary_matrix(y_true, name="y_true")
    probability = _as_matrix(y_probability, name="y_probability")
    if target.shape != probability.shape:
        raise ValueError("y_true and y_probability must have the same shape.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("y_probability must contain values between zero and one.")
    if not 0.0 < clip < 0.5:
        raise ValueError("clip must be strictly between zero and 0.5.")
    bounded = np.clip(probability, clip, 1.0 - clip)
    return float(-np.mean(target * np.log(bounded) + (1.0 - target) * np.log1p(-bounded)))


def hamming_loss(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the fraction of incorrectly predicted sample-label pairs."""

    target = _binary_matrix(y_true, name="y_true")
    prediction = _binary_matrix(y_pred, name="y_pred")
    if target.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(target != prediction))


def subset_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return exact-match accuracy across all labels of each sample."""

    target = _binary_matrix(y_true, name="y_true")
    prediction = _binary_matrix(y_pred, name="y_pred")
    if target.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    return float(np.mean(np.all(target == prediction, axis=1)))


def example_jaccard(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean sample-wise label-set intersection over union."""

    target = _binary_matrix(y_true, name="y_true")
    prediction = _binary_matrix(y_pred, name="y_pred")
    if target.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    intersection = np.sum(target * prediction, axis=1)
    union = np.sum((target + prediction) > 0.0, axis=1)
    scores = np.divide(intersection, union, out=np.ones_like(intersection), where=union != 0.0)
    return float(np.mean(scores))


def example_f1(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean sample-wise F1 over label sets."""

    target = _binary_matrix(y_true, name="y_true")
    prediction = _binary_matrix(y_pred, name="y_pred")
    if target.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    intersection = np.sum(target * prediction, axis=1)
    denominator = np.sum(target, axis=1) + np.sum(prediction, axis=1)
    scores = np.divide(2.0 * intersection, denominator, out=np.ones_like(intersection), where=denominator != 0.0)
    return float(np.mean(scores))


def _f1_from_counts(true_positive: FloatArray, false_positive: FloatArray, false_negative: FloatArray) -> FloatArray:
    denominator = 2.0 * true_positive + false_positive + false_negative
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator != 0.0,
    )


def _binary_roc_auc(y_true: FloatArray, y_probability: FloatArray) -> float | None:
    """Return tie-aware binary ROC AUC, or ``None`` when one class is absent."""

    target = y_true[:, 0]
    probability = y_probability[:, 0]
    positives = float(np.sum(target == 1.0))
    negatives = float(np.sum(target == 0.0))
    if positives == 0.0 or negatives == 0.0:
        return None
    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order]
    concordant = 0.0
    preceding_negatives = 0.0
    start = 0
    while start < len(sorted_target):
        stop = start + 1
        while stop < len(sorted_target) and sorted_probability[stop] == sorted_probability[start]:
            stop += 1
        group = sorted_target[start:stop]
        group_positives = float(np.sum(group == 1.0))
        group_negatives = float(np.sum(group == 0.0))
        concordant += group_positives * (preceding_negatives + 0.5 * group_negatives)
        preceding_negatives += group_negatives
        start = stop
    return concordant / (positives * negatives)


def classification_metrics(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Return the common multilabel metrics used by the study harness."""

    target = _binary_matrix(y_true, name="y_true")
    probability = _as_matrix(y_probability, name="y_probability")
    if target.shape != probability.shape:
        raise ValueError("y_true and y_probability must have the same shape.")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("y_probability must contain values between zero and one.")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between zero and one.")
    prediction = np.asarray(probability >= threshold, dtype=np.float64)
    true_positive = np.sum(target * prediction, axis=0)
    false_positive = np.sum((1.0 - target) * prediction, axis=0)
    false_negative = np.sum(target * (1.0 - prediction), axis=0)
    per_label_f1 = _f1_from_counts(true_positive, false_positive, false_negative)
    micro_f1 = _f1_from_counts(
        np.asarray([np.sum(true_positive)]),
        np.asarray([np.sum(false_positive)]),
        np.asarray([np.sum(false_negative)]),
    )[0]
    metrics = {
        "binary_cross_entropy": binary_cross_entropy(target, probability),
        "hamming_loss": hamming_loss(target, prediction),
        "subset_accuracy": subset_accuracy(target, prediction),
        "example_jaccard": example_jaccard(target, prediction),
        "example_f1": example_f1(target, prediction),
        "micro_f1": float(micro_f1),
        "macro_f1": float(np.mean(per_label_f1)),
    }
    if target.shape[1] == 1:
        metrics["accuracy"] = float(np.mean(target == prediction))
        positives = np.sum(target[:, 0] == 1.0)
        negatives = np.sum(target[:, 0] == 0.0)
        if positives and negatives:
            true_positive_rate = np.sum((target[:, 0] == 1.0) & (prediction[:, 0] == 1.0)) / positives
            true_negative_rate = np.sum((target[:, 0] == 0.0) & (prediction[:, 0] == 0.0)) / negatives
            metrics["balanced_accuracy"] = float(0.5 * (true_positive_rate + true_negative_rate))
            roc_auc = _binary_roc_auc(target, probability)
            if roc_auc is not None:  # Both classes above make this defensive only.
                metrics["roc_auc"] = roc_auc
    return metrics


def regression_metrics(y_train: ArrayLike, y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return raw and train-mean-normalized multi-output regression metrics."""

    target, prediction = _paired_matrices(y_true, y_pred)
    metrics = {
        "global_rmse": global_rmse(target, prediction),
        "mae": float(np.mean(np.abs(target - prediction))),
        "mean_target_r2": mean_target_r2(target, prediction),
    }
    try:
        metrics["average_relative_rmse"] = average_relative_rmse(y_train, target, prediction)
    except ValueError as error:
        if "Relative RMSE is undefined" not in str(error):
            raise
    return metrics


__all__ = [
    "average_relative_rmse",
    "binary_cross_entropy",
    "classification_metrics",
    "example_f1",
    "example_jaccard",
    "global_rmse",
    "hamming_loss",
    "mean_target_r2",
    "mean_fold_average_relative_rmse",
    "regression_metrics",
    "subset_accuracy",
    "target_relative_rmse",
]
