"""Optional cached Numba kernels.

This module is imported only when the fitted estimator selects the Numba
engine, so importing :mod:`mbtr` never requires or compiles Numba.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, nogil=True)
def best_constant_hessian_split(
    counts: np.ndarray,
    gradient_hist: np.ndarray,
    n_bins_by_feature: np.ndarray,
    node_count: int,
    min_samples_leaf: int,
    lambda_weights: float,
    parent_gain: float,
    lambda_leaves: float,
) -> tuple[int, int, float, np.ndarray, float, np.ndarray, float]:
    """Find the best exact MSE split from packed feature histograms."""

    n_features = counts.shape[0]
    n_outputs = gradient_hist.shape[2]
    best_feature = -1
    best_bin = -1
    best_split_gain = -np.inf
    best_left_gain = 0.0
    best_right_gain = 0.0
    best_left_value = np.zeros(n_outputs, dtype=np.float64)
    best_right_value = np.zeros(n_outputs, dtype=np.float64)
    total_gradient = np.empty(n_outputs, dtype=np.float64)
    left_gradient = np.empty(n_outputs, dtype=np.float64)
    feature_best_left_gradient = np.empty(n_outputs, dtype=np.float64)
    epsilon = np.finfo(np.float64).eps

    for feature in range(n_features):
        n_bins = int(n_bins_by_feature[feature])
        if n_bins < 2:
            continue
        for output in range(n_outputs):
            total = 0.0
            for bin_index in range(n_bins):
                total += gradient_hist[feature, bin_index, output]
            total_gradient[output] = total
            left_gradient[output] = 0.0

        left_count = 0
        feature_has_candidate = False
        feature_best_bin = -1
        feature_best_left_gain = 0.0
        feature_best_right_gain = 0.0
        feature_best_split_gain = -np.inf
        for bin_index in range(n_bins - 1):
            left_count += int(counts[feature, bin_index])
            for output in range(n_outputs):
                left_gradient[output] += gradient_hist[feature, bin_index, output]
            right_count = node_count - left_count
            if left_count < min_samples_leaf or right_count < min_samples_leaf:
                continue

            left_denominator = max(float(left_count) + lambda_weights, epsilon)
            right_denominator = max(float(right_count) + lambda_weights, epsilon)
            left_dot = 0.0
            right_dot = 0.0
            for output in range(n_outputs):
                left_value = left_gradient[output] / left_denominator
                right_gradient = total_gradient[output] - left_gradient[output]
                right_value = right_gradient / right_denominator
                left_dot += left_gradient[output] * left_value
                right_dot += right_gradient * right_value
            left_gain = 0.5 * left_dot
            right_gain = 0.5 * right_dot
            split_gain = left_gain + right_gain - parent_gain - lambda_leaves
            if not feature_has_candidate or split_gain > feature_best_split_gain:
                feature_has_candidate = True
                feature_best_bin = bin_index
                feature_best_left_gain = left_gain
                feature_best_right_gain = right_gain
                feature_best_split_gain = split_gain
                for output in range(n_outputs):
                    feature_best_left_gradient[output] = left_gradient[output]

        if feature_has_candidate and (best_feature < 0 or feature_best_split_gain > best_split_gain + 1e-15):
            best_feature = feature
            best_bin = feature_best_bin
            best_split_gain = feature_best_split_gain
            best_left_gain = feature_best_left_gain
            best_right_gain = feature_best_right_gain
            left_count = 0
            for bin_index in range(feature_best_bin + 1):
                left_count += int(counts[feature, bin_index])
            right_count = node_count - left_count
            left_denominator = max(float(left_count) + lambda_weights, epsilon)
            right_denominator = max(float(right_count) + lambda_weights, epsilon)
            for output in range(n_outputs):
                best_left_value[output] = feature_best_left_gradient[output] / left_denominator
                best_right_value[output] = (
                    total_gradient[output] - feature_best_left_gradient[output]
                ) / right_denominator

    return (
        best_feature,
        best_bin,
        best_split_gain,
        best_left_value,
        best_left_gain,
        best_right_value,
        best_right_gain,
    )


@njit(cache=True, nogil=True)
def histogram_statistics(
    bins: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_bins, dtype=np.int64)
    gradient_hist = np.zeros((n_bins, gradient.shape[1]), dtype=np.float64)
    hessian_hist = np.zeros((n_bins, hessian.shape[1]), dtype=np.float64)
    for row in range(bins.shape[0]):
        bin_index = int(bins[row])
        counts[bin_index] += 1
        for output in range(gradient.shape[1]):
            gradient_hist[bin_index, output] += gradient[row, output]
            hessian_hist[bin_index, output] += hessian[row, output]
    return counts, gradient_hist, hessian_hist


@njit(cache=True, nogil=True)
def histogram_gradient_statistics(
    bins: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate counts and gradients for a unit-Hessian objective."""

    counts = np.zeros(n_bins, dtype=np.int64)
    gradient_hist = np.zeros((n_bins, gradient.shape[1]), dtype=np.float64)
    for row in range(bins.shape[0]):
        bin_index = int(bins[row])
        counts[bin_index] += 1
        for output in range(gradient.shape[1]):
            gradient_hist[bin_index, output] += gradient[row, output]
    return counts, gradient_hist


@njit(cache=True, nogil=True)
def histogram_gradient_statistics_all_features(
    binned_x: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate unit-Hessian statistics for every feature in one call."""

    counts = np.zeros((binned_x.shape[1], n_bins), dtype=np.int64)
    gradient_hist = np.zeros(
        (binned_x.shape[1], n_bins, gradient.shape[1]),
        dtype=np.float64,
    )
    for row in range(binned_x.shape[0]):
        for feature in range(binned_x.shape[1]):
            bin_index = int(binned_x[row, feature])
            counts[feature, bin_index] += 1
            for output in range(gradient.shape[1]):
                gradient_hist[feature, bin_index, output] += gradient[row, output]
    return counts, gradient_hist


@njit(cache=True, nogil=True)
def histogram_gradient_statistics_all_features_indexed(
    binned_x: np.ndarray,
    gradient: np.ndarray,
    rows: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate unit-Hessian statistics without materializing node rows."""

    counts = np.zeros((binned_x.shape[1], n_bins), dtype=np.int64)
    gradient_hist = np.zeros(
        (binned_x.shape[1], n_bins, gradient.shape[1]),
        dtype=np.float64,
    )
    for position in range(rows.shape[0]):
        row = int(rows[position])
        for feature in range(binned_x.shape[1]):
            bin_index = int(binned_x[row, feature])
            counts[feature, bin_index] += 1
            for output in range(gradient.shape[1]):
                gradient_hist[feature, bin_index, output] += gradient[row, output]
    return counts, gradient_hist


@njit(cache=True, nogil=True, parallel=True)
def histogram_gradient_statistics_all_features_parallel(
    binned_x_by_feature: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Feature-parallel unit-Hessian histogram aggregation."""

    counts = np.zeros((binned_x_by_feature.shape[0], n_bins), dtype=np.int64)
    gradient_hist = np.zeros(
        (binned_x_by_feature.shape[0], n_bins, gradient.shape[1]),
        dtype=np.float64,
    )
    for feature in prange(binned_x_by_feature.shape[0]):
        for row in range(binned_x_by_feature.shape[1]):
            bin_index = int(binned_x_by_feature[feature, row])
            counts[feature, bin_index] += 1
            for output in range(gradient.shape[1]):
                gradient_hist[feature, bin_index, output] += gradient[row, output]
    return counts, gradient_hist


@njit(cache=True, nogil=True)
def histogram_linear_statistics(
    bins: np.ndarray,
    design: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(n_bins, dtype=np.int64)
    gram_hist = np.zeros((n_bins, design.shape[1], design.shape[1]), dtype=np.float64)
    gradient_hist = np.zeros((n_bins, design.shape[1], gradient.shape[1]), dtype=np.float64)
    for row in range(bins.shape[0]):
        bin_index = int(bins[row])
        counts[bin_index] += 1
        for first in range(design.shape[1]):
            for second in range(first, design.shape[1]):
                gram_hist[bin_index, first, second] += design[row, first] * design[row, second]
            for output in range(gradient.shape[1]):
                gradient_hist[bin_index, first, output] += design[row, first] * gradient[row, output]
    for bin_index in range(n_bins):
        for first in range(design.shape[1]):
            for second in range(first + 1, design.shape[1]):
                gram_hist[bin_index, second, first] = gram_hist[bin_index, first, second]
    return counts, gram_hist, gradient_hist


@njit(cache=True, nogil=True, parallel=True)
def histogram_linear_statistics_all_features_parallel(
    binned_x_by_feature: np.ndarray,
    design: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feature-parallel shared-Gram histogram aggregation."""

    counts = np.zeros((binned_x_by_feature.shape[0], n_bins), dtype=np.int64)
    gram_hist = np.zeros(
        (binned_x_by_feature.shape[0], n_bins, design.shape[1], design.shape[1]),
        dtype=np.float64,
    )
    gradient_hist = np.zeros(
        (binned_x_by_feature.shape[0], n_bins, design.shape[1], gradient.shape[1]),
        dtype=np.float64,
    )
    for feature in prange(binned_x_by_feature.shape[0]):
        for row in range(binned_x_by_feature.shape[1]):
            bin_index = int(binned_x_by_feature[feature, row])
            counts[feature, bin_index] += 1
            for first in range(design.shape[1]):
                for second in range(first, design.shape[1]):
                    gram_hist[feature, bin_index, first, second] += design[row, first] * design[row, second]
                for output in range(gradient.shape[1]):
                    gradient_hist[feature, bin_index, first, output] += design[row, first] * gradient[row, output]
        for bin_index in range(n_bins):
            for first in range(design.shape[1]):
                for second in range(first + 1, design.shape[1]):
                    gram_hist[feature, bin_index, second, first] = gram_hist[
                        feature,
                        bin_index,
                        first,
                        second,
                    ]
    return counts, gram_hist, gradient_hist


@njit(cache=True, nogil=True)
def histogram_linear_logistic_statistics(
    bins: np.ndarray,
    design: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate exact per-output linear-logistic Newton statistics."""

    counts = np.zeros(n_bins, dtype=np.int64)
    gram_hist = np.zeros(
        (n_bins, gradient.shape[1], design.shape[1], design.shape[1]),
        dtype=np.float64,
    )
    gradient_hist = np.zeros((n_bins, design.shape[1], gradient.shape[1]), dtype=np.float64)
    for row in range(bins.shape[0]):
        bin_index = int(bins[row])
        counts[bin_index] += 1
        for output in range(gradient.shape[1]):
            curvature = hessian[row, output]
            for first in range(design.shape[1]):
                gradient_hist[bin_index, first, output] += design[row, first] * gradient[row, output]
                for second in range(first, design.shape[1]):
                    gram_hist[bin_index, output, first, second] += curvature * design[row, first] * design[row, second]
    for bin_index in range(n_bins):
        for output in range(gradient.shape[1]):
            for first in range(design.shape[1]):
                for second in range(first + 1, design.shape[1]):
                    gram_hist[bin_index, output, second, first] = gram_hist[bin_index, output, first, second]
    return counts, gram_hist, gradient_hist


__all__ = [
    "best_constant_hessian_split",
    "histogram_gradient_statistics",
    "histogram_gradient_statistics_all_features",
    "histogram_gradient_statistics_all_features_indexed",
    "histogram_gradient_statistics_all_features_parallel",
    "histogram_linear_logistic_statistics",
    "histogram_linear_statistics",
    "histogram_linear_statistics_all_features_parallel",
    "histogram_statistics",
]
