"""Compact histogram regression tree used by the boosted estimator."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .losses import LeafSolution, Loss

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.integer[Any]]
CountArray = NDArray[np.int64]


@dataclass
class TreeNode:
    """One node in a compact array-addressed binary tree."""

    depth: int
    sample_count: int
    value: FloatArray
    feature: int = -1
    threshold: float = np.nan
    left: int = -1
    right: int = -1
    split_gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.left < 0


@dataclass
class _FeatureHistogram:
    counts: CountArray
    gradient: FloatArray | None = None
    gram: FloatArray | None = None
    linear_gradient: FloatArray | None = None


@dataclass
class _NodeHistograms:
    features: list[_FeatureHistogram | None]
    nbytes: int
    packed_counts: CountArray | None = None
    packed_gradient: FloatArray | None = None
    n_bins_by_feature: CountArray | None = None


class RegressionTree:
    """Histogram tree with constant or ridge-linear leaf responses."""

    _HISTOGRAM_CACHE_MAX_BYTES = 256 * 1024**2

    def __init__(
        self,
        *,
        objective: Loss,
        min_samples_leaf: int,
        max_depth: int | None,
        max_leaves: int | None,
        min_split_gain: float,
        learning_rate: float,
        engine: str = "numpy",
    ) -> None:
        self.objective = objective
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_split_gain = float(min_split_gain)
        self.learning_rate = float(learning_rate)
        self.engine = engine
        self.nodes: list[TreeNode] = []
        self.n_features_in_: int | None = None
        self.n_outputs_: int | None = None
        self.n_response_features_: int | None = None
        self.feature_importances_: FloatArray | None = None
        self.split_output_indices_: NDArray[np.int64] | None = None

    @staticmethod
    def _augment_response_features(features: FloatArray) -> FloatArray:
        return np.column_stack((features, np.ones(features.shape[0], dtype=np.float64)))

    def _solve_rows(
        self,
        rows: IntArray,
        gradient: FloatArray,
        hessian: FloatArray,
        response_features: FloatArray | None,
    ) -> LeafSolution:
        gradient_sum = np.sum(gradient[rows], axis=0)
        if self.objective.constant_hessian:
            hessian_sum = np.full(gradient.shape[1], len(rows), dtype=np.float64)
        else:
            hessian_sum = np.sum(hessian[rows], axis=0)
        linear_mode = self.objective.linear_response_mode
        if linear_mode is not None:
            if response_features is None:
                raise ValueError("Linear leaves require response_features.")
            design = response_features[rows]
            if linear_mode == "shared":
                # NumPy's macOS Accelerate-backed ``matmul`` can set spurious
                # floating-point flags for these finite reductions (notably in
                # NumPy 2.2). ``dot`` uses the same BLAS-sized operation without
                # inheriting those false RuntimeWarnings.
                linear_gram = self.objective.linear_gram_scale * np.dot(design.T, design)
            elif linear_mode == "per_output":
                linear_gram = np.einsum(
                    "ni,no,nj->oij",
                    design,
                    hessian[rows],
                    design,
                    optimize=True,
                )
            else:  # pragma: no cover - protected by built-in objective definitions.
                raise RuntimeError(f"Unknown linear response mode {linear_mode!r}.")
            return self.objective.solve_leaf(
                gradient_sum,
                hessian_sum,
                len(rows),
                linear_gram=linear_gram,
                linear_gradient=np.dot(design.T, gradient[rows]),
            )
        return self.objective.solve_leaf(gradient_sum, hessian_sum, len(rows))

    def _solve_shared_linear_candidates(
        self,
        gram: FloatArray,
        projected_gradient: FloatArray,
    ) -> tuple[FloatArray, FloatArray] | None:
        """Solve a batch of built-in shared-Gram leaf systems.

        NumPy dispatches the stacked factorizations in compiled code, avoiding
        two Python-level small-system solves for every candidate threshold.
        A singular member invalidates the batch and lets the caller use the
        authoritative scalar fallback.
        """

        if not hasattr(self.objective, "regularize_intercept"):
            return None
        systems = np.array(gram, dtype=np.float64, copy=True, order="C")
        right_hand_side = np.ascontiguousarray(projected_gradient, dtype=np.float64)
        if systems.ndim != 3 or right_hand_side.ndim != 3:
            return None
        width = systems.shape[1]
        if systems.shape[2] != width or right_hand_side.shape[:2] != systems.shape[:2]:
            return None
        penalty = np.full(width, self.objective.lambda_weights, dtype=np.float64)
        if not bool(self.objective.regularize_intercept):
            penalty[-1] = 0.0
        diagonal = np.arange(width)
        systems[:, diagonal, diagonal] += penalty
        try:
            factor = np.linalg.cholesky(systems)
            intermediate = np.linalg.solve(factor, right_hand_side)
            coefficients = np.linalg.solve(np.swapaxes(factor, 1, 2), intermediate)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(coefficients)):
            return None
        gains = 0.5 * np.einsum(
            "bqo,bqo->b",
            right_hand_side,
            coefficients,
            optimize=False,
        )
        return np.asarray(coefficients), np.asarray(gains)

    def _solve_diagonal_candidates(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
    ) -> tuple[FloatArray, FloatArray] | None:
        """Vectorize candidates using the base diagonal leaf solution."""

        if type(self.objective).solve_leaf is not Loss.solve_leaf:
            return None
        denominator = np.maximum(
            np.asarray(hessian_sum, dtype=np.float64) + self.objective.lambda_weights,
            np.finfo(np.float64).eps,
        )
        response = np.asarray(gradient_sum, dtype=np.float64) / denominator
        gains = 0.5 * np.einsum(
            "bo,bo->b",
            np.asarray(gradient_sum, dtype=np.float64),
            response,
            optimize=False,
        )
        return response, np.asarray(gains)

    def _histogram_cache_supported(
        self,
        bin_thresholds: Sequence[FloatArray],
        *,
        n_outputs: int,
        design_width: int | None,
    ) -> bool:
        """Return whether one raw all-feature histogram package is affordable."""

        total_bins = sum(len(values) + 1 for values in bin_thresholds if len(values))
        linear_mode = self.objective.linear_response_mode
        if linear_mode == "shared":
            assert design_width is not None
            bytes_per_bin = 8 + 8 * design_width * design_width + 8 * design_width * n_outputs
        elif linear_mode is None and self.objective.constant_hessian:
            bytes_per_bin = 8 + 8 * n_outputs
        else:
            return False
        return total_bins * bytes_per_bin <= self._HISTOGRAM_CACHE_MAX_BYTES

    def _build_cacheable_histograms(
        self,
        rows: IntArray,
        binned_x: IntArray,
        bin_thresholds: Sequence[FloatArray],
        gradient: FloatArray,
        response_features: FloatArray | None,
    ) -> _NodeHistograms:
        """Build raw histograms for cache-supported objective layouts."""

        linear_mode = self.objective.linear_response_mode
        features: list[_FeatureHistogram | None] = []
        parallel_histograms = int(os.environ.get("NUMBA_NUM_THREADS", "1")) > 1
        if self.engine == "numba" and linear_mode is None and not parallel_histograms:
            max_bins = max((len(thresholds) + 1 for thresholds in bin_thresholds), default=1)
            from ._numba import histogram_gradient_statistics_all_features_indexed

            all_counts, all_gradient = histogram_gradient_statistics_all_features_indexed(
                binned_x,
                gradient,
                np.ascontiguousarray(rows),
                max_bins,
            )
            for feature, thresholds in enumerate(bin_thresholds):
                if len(thresholds) == 0:
                    features.append(None)
                    continue
                n_bins = len(thresholds) + 1
                features.append(
                    _FeatureHistogram(
                        counts=all_counts[feature, :n_bins],
                        gradient=all_gradient[feature, :n_bins],
                    )
                )
            return _NodeHistograms(
                features=features,
                nbytes=all_counts.nbytes + all_gradient.nbytes,
                packed_counts=all_counts,
                packed_gradient=all_gradient,
                n_bins_by_feature=np.asarray(
                    [len(thresholds) + 1 if len(thresholds) else 0 for thresholds in bin_thresholds],
                    dtype=np.int64,
                ),
            )

        node_gradient = np.ascontiguousarray(gradient[rows])
        node_design = None if response_features is None else np.ascontiguousarray(response_features[rows])
        if self.engine == "numba" and parallel_histograms:
            node_bins_by_feature = np.ascontiguousarray(binned_x[rows].T)
            max_bins = max((len(thresholds) + 1 for thresholds in bin_thresholds), default=1)
            if linear_mode == "shared":
                assert node_design is not None
                from ._numba import histogram_linear_statistics_all_features_parallel

                all_counts, all_gram, all_linear_gradient = histogram_linear_statistics_all_features_parallel(
                    node_bins_by_feature,
                    node_design,
                    node_gradient,
                    max_bins,
                )
                all_gram *= self.objective.linear_gram_scale
            else:
                from ._numba import histogram_gradient_statistics_all_features_parallel

                all_counts, all_gradient = histogram_gradient_statistics_all_features_parallel(
                    node_bins_by_feature,
                    node_gradient,
                    max_bins,
                )
            for feature, thresholds in enumerate(bin_thresholds):
                if len(thresholds) == 0:
                    features.append(None)
                    continue
                n_bins = len(thresholds) + 1
                if linear_mode == "shared":
                    features.append(
                        _FeatureHistogram(
                            counts=all_counts[feature, :n_bins],
                            gram=all_gram[feature, :n_bins],
                            linear_gradient=all_linear_gradient[feature, :n_bins],
                        )
                    )
                else:
                    features.append(
                        _FeatureHistogram(
                            counts=all_counts[feature, :n_bins],
                            gradient=all_gradient[feature, :n_bins],
                        )
                    )
            if linear_mode is None:
                return _NodeHistograms(
                    features=features,
                    nbytes=all_counts.nbytes + all_gradient.nbytes,
                    packed_counts=all_counts,
                    packed_gradient=all_gradient,
                    n_bins_by_feature=np.asarray(
                        [len(thresholds) + 1 if len(thresholds) else 0 for thresholds in bin_thresholds],
                        dtype=np.int64,
                    ),
                )
            nbytes = sum(
                array.nbytes
                for histogram in features
                if histogram is not None
                for array in (
                    histogram.counts,
                    histogram.gradient,
                    histogram.gram,
                    histogram.linear_gradient,
                )
                if array is not None
            )
            return _NodeHistograms(features=features, nbytes=nbytes)

        for feature, thresholds in enumerate(bin_thresholds):
            if len(thresholds) == 0:
                features.append(None)
                continue
            local_bins = np.asarray(binned_x[rows, feature], dtype=np.int64)
            n_bins = len(thresholds) + 1
            if linear_mode == "shared":
                assert node_design is not None
                if self.engine == "numba":
                    from ._numba import histogram_linear_statistics

                    counts, gram, linear_gradient = histogram_linear_statistics(
                        np.ascontiguousarray(local_bins),
                        node_design,
                        node_gradient,
                        n_bins,
                    )
                    gram *= self.objective.linear_gram_scale
                else:
                    counts = np.bincount(local_bins, minlength=n_bins).astype(np.int64)
                    width = node_design.shape[1]
                    linear_gradient = np.zeros((n_bins, width, gradient.shape[1]))
                    for first in range(width):
                        for output in range(gradient.shape[1]):
                            linear_gradient[:, first, output] = np.bincount(
                                local_bins,
                                weights=node_design[:, first] * node_gradient[:, output],
                                minlength=n_bins,
                            )
                    gram = np.zeros((n_bins, width, width))
                    for first in range(width):
                        for second in range(first, width):
                            values = node_design[:, first] * node_design[:, second]
                            sums = np.asarray(
                                np.bincount(local_bins, weights=values, minlength=n_bins),
                                dtype=np.float64,
                            )
                            sums *= self.objective.linear_gram_scale
                            gram[:, first, second] = sums
                            gram[:, second, first] = sums
                features.append(
                    _FeatureHistogram(
                        counts=counts,
                        gram=gram,
                        linear_gradient=linear_gradient,
                    )
                )
            else:
                counts = np.bincount(local_bins, minlength=n_bins).astype(np.int64)
                gradient_hist = np.empty((n_bins, gradient.shape[1]))
                for output in range(gradient.shape[1]):
                    gradient_hist[:, output] = np.bincount(
                        local_bins,
                        weights=node_gradient[:, output],
                        minlength=n_bins,
                    )
                features.append(_FeatureHistogram(counts=counts, gradient=gradient_hist))
        nbytes = sum(
            array.nbytes
            for histogram in features
            if histogram is not None
            for array in (
                histogram.counts,
                histogram.gradient,
                histogram.gram,
                histogram.linear_gradient,
            )
            if array is not None
        )
        return _NodeHistograms(features=features, nbytes=nbytes)

    @staticmethod
    def _subtract_histograms(
        parent: _NodeHistograms,
        smaller_child: _NodeHistograms,
    ) -> _NodeHistograms:
        """Mutate an exhausted parent package into its larger child package."""

        for parent_feature, child_feature in zip(
            parent.features,
            smaller_child.features,
            strict=True,
        ):
            if parent_feature is None:
                if child_feature is not None:  # pragma: no cover - internal invariant.
                    raise RuntimeError("Histogram feature-layout invariant was violated.")
                continue
            if child_feature is None:  # pragma: no cover - internal invariant.
                raise RuntimeError("Histogram feature-layout invariant was violated.")
            parent_feature.counts -= child_feature.counts
            for name in ("gradient", "gram", "linear_gradient"):
                parent_value = getattr(parent_feature, name)
                child_value = getattr(child_feature, name)
                if parent_value is None:
                    if child_value is not None:  # pragma: no cover - internal invariant.
                        raise RuntimeError("Histogram statistic-layout invariant was violated.")
                else:
                    if child_value is None:  # pragma: no cover - internal invariant.
                        raise RuntimeError("Histogram statistic-layout invariant was violated.")
                    parent_value -= child_value
        return parent

    def fit(
        self,
        binned_x: IntArray,
        bin_thresholds: Sequence[FloatArray],
        gradient: FloatArray,
        hessian: FloatArray,
        *,
        response_features: FloatArray | None = None,
        split_gradient: FloatArray | None = None,
        split_hessian: FloatArray | None = None,
        split_output_indices: NDArray[np.int64] | None = None,
    ) -> RegressionTree:
        n_samples, n_features = binned_x.shape
        if gradient.shape != hessian.shape or gradient.shape[0] != n_samples:
            raise ValueError("gradient and hessian must have matching sample/output shapes.")
        value_gradient = gradient
        value_hessian = hessian
        if (split_gradient is None) != (split_hessian is None):
            raise ValueError("split_gradient and split_hessian must be provided together.")
        if split_gradient is not None and split_hessian is not None:
            if split_gradient.shape != split_hessian.shape or split_gradient.shape[0] != n_samples:
                raise ValueError("Split derivatives must have matching sample/output shapes.")
            gradient = np.ascontiguousarray(split_gradient)
            hessian = np.ascontiguousarray(split_hessian)
            if split_output_indices is None or len(split_output_indices) != gradient.shape[1]:
                raise ValueError("split_output_indices must identify every sketched output.")
            self.split_output_indices_ = np.asarray(split_output_indices, dtype=np.int64)
        else:
            self.split_output_indices_ = None
        uses_output_sketch = gradient is not value_gradient
        if response_features is not None:
            if response_features.ndim != 2 or response_features.shape[0] != n_samples:
                raise ValueError("response_features must match the number of samples.")
            response_features = self._augment_response_features(response_features)
            self.n_response_features_ = response_features.shape[1] - 1
        elif self.objective.requires_response_features:
            raise ValueError("This objective requires response_features.")

        self.n_features_in_ = n_features
        self.n_outputs_ = value_gradient.shape[1]
        self.nodes = []
        self.feature_importances_ = np.zeros(n_features, dtype=np.float64)
        training_prediction = np.empty_like(value_gradient)
        histogram_cache_enabled = self._histogram_cache_supported(
            bin_thresholds,
            n_outputs=gradient.shape[1],
            design_width=None if response_features is None else response_features.shape[1],
        )
        self.histogram_subtraction_ = histogram_cache_enabled

        def assign_leaf_prediction(rows: IntArray, value: FloatArray) -> None:
            if self.objective.linear_response_mode is None:
                training_prediction[rows] = value
            else:
                assert response_features is not None
                training_prediction[rows] = np.dot(response_features[rows], value)

        root_rows = np.arange(n_samples, dtype=np.int64)
        root_solution = self._solve_rows(root_rows, gradient, hessian, response_features)
        root_value_solution = (
            self._solve_rows(root_rows, value_gradient, value_hessian, response_features)
            if uses_output_sketch
            else root_solution
        )
        self.nodes.append(
            TreeNode(
                depth=0,
                sample_count=n_samples,
                value=np.asarray(root_value_solution.value * self.learning_rate),
            )
        )
        stack: list[tuple[int, IntArray, LeafSolution, _NodeHistograms | None]] = [(0, root_rows, root_solution, None)]
        leaf_count = 1

        while stack:
            node_index, rows, parent_solution, cached_histograms = stack.pop()
            node = self.nodes[node_index]
            if len(rows) < 2 * self.min_samples_leaf:
                assign_leaf_prediction(rows, node.value)
                continue
            if self.max_depth is not None and node.depth >= self.max_depth:
                assign_leaf_prediction(rows, node.value)
                continue
            if self.max_leaves is not None and leaf_count >= self.max_leaves:
                assign_leaf_prediction(rows, node.value)
                continue

            linear_mode = self.objective.linear_response_mode
            constant_hessian = bool(self.objective.constant_hessian) and linear_mode is None
            if histogram_cache_enabled:
                node_histograms = cached_histograms or self._build_cacheable_histograms(
                    rows,
                    binned_x,
                    bin_thresholds,
                    gradient,
                    response_features,
                )
                node_gradient = None
                node_hessian = None
                node_design = None
            else:
                node_histograms = None
                # Advanced indexing materializes these arrays.  They are
                # identical for every feature considered at this node, so
                # gather them once rather than once per feature.
                node_gradient = np.ascontiguousarray(gradient[rows])
                node_hessian = (
                    None if linear_mode == "shared" or constant_hessian else np.ascontiguousarray(hessian[rows])
                )
                node_design = None if response_features is None else np.ascontiguousarray(response_features[rows])

            best: tuple[float, int, int, LeafSolution, LeafSolution] | None = None
            features_to_scan: Sequence[int] = range(n_features)
            if (
                node_histograms is not None
                and node_histograms.packed_counts is not None
                and node_histograms.packed_gradient is not None
                and node_histograms.n_bins_by_feature is not None
                and linear_mode is None
                and constant_hessian
                and type(self.objective).solve_leaf is Loss.solve_leaf
            ):
                from ._numba import best_constant_hessian_split

                (
                    best_feature,
                    best_split_bin,
                    split_gain,
                    left_response,
                    left_gain,
                    right_response,
                    right_gain,
                ) = best_constant_hessian_split(
                    node_histograms.packed_counts,
                    node_histograms.packed_gradient,
                    node_histograms.n_bins_by_feature,
                    len(rows),
                    self.min_samples_leaf,
                    self.objective.lambda_weights,
                    parent_solution.gain,
                    self.objective.lambda_leaves,
                )
                if best_feature >= 0:
                    best = (
                        float(split_gain),
                        int(best_feature),
                        int(best_split_bin),
                        LeafSolution(np.asarray(left_response), float(left_gain)),
                        LeafSolution(np.asarray(right_response), float(right_gain)),
                    )
                features_to_scan = ()

            for feature in features_to_scan:
                thresholds = np.asarray(bin_thresholds[feature])
                if thresholds.size == 0:
                    continue
                n_bins = thresholds.size + 1
                cached_feature = None if node_histograms is None else node_histograms.features[feature]
                if cached_feature is not None:
                    counts = cached_feature.counts
                    if linear_mode is not None:
                        assert cached_feature.gram is not None
                        assert cached_feature.linear_gradient is not None
                        gram_hist = cached_feature.gram
                        linear_gradient_hist = cached_feature.linear_gradient
                    else:
                        assert cached_feature.gradient is not None
                        gradient_hist = cached_feature.gradient
                elif self.engine == "numba":
                    assert node_gradient is not None
                    local_bins = np.asarray(binned_x[rows, feature], dtype=np.int64)
                    from ._numba import (
                        histogram_gradient_statistics,
                        histogram_linear_logistic_statistics,
                        histogram_linear_statistics,
                        histogram_statistics,
                    )

                    if linear_mode is not None:
                        assert node_design is not None
                        design = node_design
                        if linear_mode == "shared":
                            counts, gram_hist, linear_gradient_hist = histogram_linear_statistics(
                                np.ascontiguousarray(local_bins),
                                design,
                                node_gradient,
                                n_bins,
                            )
                            gram_hist *= self.objective.linear_gram_scale
                        else:
                            assert node_hessian is not None
                            counts, gram_hist, linear_gradient_hist = histogram_linear_logistic_statistics(
                                np.ascontiguousarray(local_bins),
                                design,
                                node_gradient,
                                node_hessian,
                                n_bins,
                            )
                    else:
                        if constant_hessian:
                            counts, gradient_hist = histogram_gradient_statistics(
                                np.ascontiguousarray(local_bins),
                                node_gradient,
                                n_bins,
                            )
                        else:
                            assert node_hessian is not None
                            counts, gradient_hist, hessian_hist = histogram_statistics(
                                np.ascontiguousarray(local_bins),
                                node_gradient,
                                node_hessian,
                                n_bins,
                            )
                else:
                    local_bins = np.asarray(binned_x[rows, feature], dtype=np.int64)
                    counts = np.bincount(local_bins, minlength=n_bins).astype(np.int64)
                left_counts = np.cumsum(counts)[:-1]
                valid = (left_counts >= self.min_samples_leaf) & (len(rows) - left_counts >= self.min_samples_leaf)
                if not np.any(valid):
                    continue

                if linear_mode is not None:
                    if cached_feature is None and self.engine != "numba":
                        assert node_design is not None
                        assert node_gradient is not None
                        design = node_design
                        linear_gradient_hist = np.zeros((n_bins, design.shape[1], gradient.shape[1]))
                        for first in range(design.shape[1]):
                            for output in range(gradient.shape[1]):
                                linear_gradient_hist[:, first, output] = np.bincount(
                                    local_bins,
                                    weights=design[:, first] * node_gradient[:, output],
                                    minlength=n_bins,
                                )
                        if linear_mode == "shared":
                            gram_hist = np.zeros((n_bins, design.shape[1], design.shape[1]))
                            for first in range(design.shape[1]):
                                for second in range(first, design.shape[1]):
                                    values = design[:, first] * design[:, second]
                                    sums = np.asarray(
                                        np.bincount(local_bins, weights=values, minlength=n_bins),
                                        dtype=np.float64,
                                    )
                                    sums *= self.objective.linear_gram_scale
                                    gram_hist[:, first, second] = sums
                                    gram_hist[:, second, first] = sums
                        else:
                            assert node_hessian is not None
                            gram_hist = np.zeros(
                                (
                                    n_bins,
                                    gradient.shape[1],
                                    design.shape[1],
                                    design.shape[1],
                                )
                            )
                            for output in range(gradient.shape[1]):
                                for first in range(design.shape[1]):
                                    for second in range(first, design.shape[1]):
                                        values = node_hessian[:, output] * design[:, first] * design[:, second]
                                        sums = np.asarray(
                                            np.bincount(
                                                local_bins,
                                                weights=values,
                                                minlength=n_bins,
                                            ),
                                            dtype=np.float64,
                                        )
                                        gram_hist[:, output, first, second] = sums
                                        gram_hist[:, output, second, first] = sums
                    left_gram = np.cumsum(gram_hist, axis=0)[:-1]
                    left_linear_gradient = np.cumsum(linear_gradient_hist, axis=0)[:-1]
                    total_gram = np.sum(gram_hist, axis=0)
                    total_linear_gradient = np.sum(linear_gradient_hist, axis=0)
                else:
                    if cached_feature is None and self.engine != "numba":
                        assert node_gradient is not None
                        gradient_hist = np.empty((n_bins, gradient.shape[1]))
                        if not constant_hessian:
                            assert node_hessian is not None
                            hessian_hist = np.empty_like(gradient_hist)
                        for output in range(gradient.shape[1]):
                            gradient_hist[:, output] = np.bincount(
                                local_bins,
                                weights=node_gradient[:, output],
                                minlength=n_bins,
                            )
                            if not constant_hessian:
                                assert node_hessian is not None
                                hessian_hist[:, output] = np.bincount(
                                    local_bins,
                                    weights=node_hessian[:, output],
                                    minlength=n_bins,
                                )
                    left_gradient = np.cumsum(gradient_hist, axis=0)[:-1]
                    total_gradient = np.sum(gradient_hist, axis=0)
                    if constant_hessian:
                        left_hessian = np.broadcast_to(
                            left_counts[:, None],
                            (len(left_counts), gradient.shape[1]),
                        )
                        total_hessian = np.full(gradient.shape[1], len(rows), dtype=np.float64)
                    else:
                        left_hessian = np.cumsum(hessian_hist, axis=0)[:-1]
                        total_hessian = np.sum(hessian_hist, axis=0)

                candidate_bins = np.flatnonzero(valid)
                if linear_mode == "shared":
                    left_candidate_gram = left_gram[candidate_bins]
                    left_candidate_gradient = left_linear_gradient[candidate_bins]
                    right_candidate_gram = total_gram - left_candidate_gram
                    right_candidate_gradient = total_linear_gradient - left_candidate_gradient
                    candidate_count = len(candidate_bins)
                    batched = self._solve_shared_linear_candidates(
                        np.concatenate((left_candidate_gram, right_candidate_gram), axis=0),
                        np.concatenate((left_candidate_gradient, right_candidate_gradient), axis=0),
                    )
                    if batched is not None:
                        coefficients, gains = batched
                        split_gains = (
                            gains[:candidate_count]
                            + gains[candidate_count:]
                            - parent_solution.gain
                            - self.objective.lambda_leaves
                        )
                        candidate_position = int(np.argmax(split_gains))
                        split_gain = float(split_gains[candidate_position])
                        if best is None or split_gain > best[0] + 1e-15:
                            best = (
                                split_gain,
                                feature,
                                int(candidate_bins[candidate_position]),
                                LeafSolution(
                                    coefficients[candidate_position],
                                    float(gains[candidate_position]),
                                ),
                                LeafSolution(
                                    coefficients[candidate_count + candidate_position],
                                    float(gains[candidate_count + candidate_position]),
                                ),
                            )
                        continue

                if linear_mode is None:
                    left_candidate_gradient = left_gradient[candidate_bins]
                    left_candidate_hessian = left_hessian[candidate_bins]
                    right_candidate_gradient = total_gradient - left_candidate_gradient
                    right_candidate_hessian = total_hessian - left_candidate_hessian
                    candidate_count = len(candidate_bins)
                    batched = self._solve_diagonal_candidates(
                        np.concatenate((left_candidate_gradient, right_candidate_gradient), axis=0),
                        np.concatenate((left_candidate_hessian, right_candidate_hessian), axis=0),
                    )
                    if batched is not None:
                        responses, gains = batched
                        split_gains = (
                            gains[:candidate_count]
                            + gains[candidate_count:]
                            - parent_solution.gain
                            - self.objective.lambda_leaves
                        )
                        candidate_position = int(np.argmax(split_gains))
                        split_gain = float(split_gains[candidate_position])
                        if best is None or split_gain > best[0] + 1e-15:
                            best = (
                                split_gain,
                                feature,
                                int(candidate_bins[candidate_position]),
                                LeafSolution(
                                    responses[candidate_position],
                                    float(gains[candidate_position]),
                                ),
                                LeafSolution(
                                    responses[candidate_count + candidate_position],
                                    float(gains[candidate_count + candidate_position]),
                                ),
                            )
                        continue

                for candidate_bin in candidate_bins:
                    left_count = int(left_counts[candidate_bin])
                    right_count = len(rows) - left_count
                    if linear_mode is not None:
                        zeros = np.zeros(gradient.shape[1], dtype=np.float64)
                        left_solution = self.objective.solve_leaf(
                            zeros,
                            zeros,
                            left_count,
                            linear_gram=left_gram[candidate_bin],
                            linear_gradient=left_linear_gradient[candidate_bin],
                        )
                        right_solution = self.objective.solve_leaf(
                            zeros,
                            zeros,
                            right_count,
                            linear_gram=total_gram - left_gram[candidate_bin],
                            linear_gradient=total_linear_gradient - left_linear_gradient[candidate_bin],
                        )
                    else:
                        left_solution = self.objective.solve_leaf(
                            left_gradient[candidate_bin],
                            left_hessian[candidate_bin],
                            left_count,
                        )
                        right_solution = self.objective.solve_leaf(
                            total_gradient - left_gradient[candidate_bin],
                            total_hessian - left_hessian[candidate_bin],
                            right_count,
                        )
                    split_gain = (
                        left_solution.gain + right_solution.gain - parent_solution.gain - self.objective.lambda_leaves
                    )
                    if best is None or split_gain > best[0] + 1e-15:
                        best = (
                            float(split_gain),
                            feature,
                            int(candidate_bin),
                            left_solution,
                            right_solution,
                        )

            if best is None or best[0] <= self.min_split_gain:
                assign_leaf_prediction(rows, node.value)
                continue

            split_gain, feature, best_split_bin, left_solution, right_solution = best
            local_bins = np.asarray(binned_x[rows, feature], dtype=np.int64)
            left_mask = local_bins <= best_split_bin
            left_rows = rows[left_mask]
            right_rows = rows[~left_mask]
            if len(left_rows) < self.min_samples_leaf or len(right_rows) < self.min_samples_leaf:
                raise RuntimeError("Internal split-size invariant was violated.")

            left_index = len(self.nodes)
            right_index = left_index + 1
            if uses_output_sketch:
                left_value_solution = self._solve_rows(
                    left_rows,
                    value_gradient,
                    value_hessian,
                    response_features,
                )
                right_value_solution = self._solve_rows(
                    right_rows,
                    value_gradient,
                    value_hessian,
                    response_features,
                )
            else:
                left_value_solution = left_solution
                right_value_solution = right_solution
            node.feature = feature
            node.threshold = float(bin_thresholds[feature][best_split_bin])
            node.left = left_index
            node.right = right_index
            node.split_gain = split_gain
            assert self.feature_importances_ is not None
            self.feature_importances_[feature] += split_gain
            self.nodes.extend(
                (
                    TreeNode(
                        depth=node.depth + 1,
                        sample_count=len(left_rows),
                        value=np.asarray(left_value_solution.value * self.learning_rate),
                    ),
                    TreeNode(
                        depth=node.depth + 1,
                        sample_count=len(right_rows),
                        value=np.asarray(right_value_solution.value * self.learning_rate),
                    ),
                )
            )
            leaf_count += 1
            left_histograms: _NodeHistograms | None = None
            right_histograms: _NodeHistograms | None = None
            if histogram_cache_enabled:
                assert node_histograms is not None
                child_depth = node.depth + 1
                depth_allows_split = self.max_depth is None or child_depth < self.max_depth
                leaves_allow_split = self.max_leaves is None or leaf_count < self.max_leaves
                left_can_split = (
                    depth_allows_split and leaves_allow_split and len(left_rows) >= 2 * self.min_samples_leaf
                )
                right_can_split = (
                    depth_allows_split and leaves_allow_split and len(right_rows) >= 2 * self.min_samples_leaf
                )
                if left_can_split or right_can_split:
                    left_is_smaller = len(left_rows) <= len(right_rows)
                    smaller_rows = left_rows if left_is_smaller else right_rows
                    smaller_histograms = self._build_cacheable_histograms(
                        smaller_rows,
                        binned_x,
                        bin_thresholds,
                        gradient,
                        response_features,
                    )
                    larger_histograms = self._subtract_histograms(
                        node_histograms,
                        smaller_histograms,
                    )
                    if left_is_smaller:
                        left_histograms = smaller_histograms if left_can_split else None
                        right_histograms = larger_histograms if right_can_split else None
                    else:
                        right_histograms = smaller_histograms if right_can_split else None
                        left_histograms = larger_histograms if left_can_split else None
            stack.append((right_index, right_rows, right_solution, right_histograms))
            stack.append((left_index, left_rows, left_solution, left_histograms))

        self._training_prediction = training_prediction
        return self

    def take_training_prediction(self) -> FloatArray:
        """Return and release the update cached while fitting this tree."""

        if not hasattr(self, "_training_prediction"):
            raise RuntimeError("No fitted-training prediction is available.")
        prediction = self._training_prediction
        del self._training_prediction
        return prediction

    def _validate_prediction_data(
        self, x: FloatArray, response_features: FloatArray | None
    ) -> tuple[FloatArray, FloatArray | None]:
        if not self.nodes or self.n_features_in_ is None:
            raise RuntimeError("The tree is not fitted.")
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_features_in_:
            raise ValueError(f"x must have shape (n_samples, {self.n_features_in_}); got {x.shape}.")
        if not np.all(np.isfinite(x)):
            raise ValueError("x must contain only finite values.")
        if self.objective.requires_response_features:
            if response_features is None:
                raise ValueError("This tree requires response_features.")
            response_features = np.asarray(response_features, dtype=np.float64)
            expected = self.n_response_features_
            if response_features.ndim != 2 or response_features.shape != (
                x.shape[0],
                expected,
            ):
                raise ValueError(f"response_features must have the same rows as x and {expected} columns.")
            if not np.all(np.isfinite(response_features)):
                raise ValueError("response_features must contain only finite values.")
            response_features = self._augment_response_features(response_features)
        return x, response_features

    def apply(self, x: FloatArray) -> NDArray[np.int64]:
        x, _ = self._validate_prediction_data(
            x,
            np.zeros((len(x), self.n_response_features_ or 0)) if self.objective.requires_response_features else None,
        )
        leaves = np.empty(x.shape[0], dtype=np.int64)
        stack: list[tuple[int, NDArray[np.int64]]] = [(0, np.arange(x.shape[0], dtype=np.int64))]
        while stack:
            node_index, rows = stack.pop()
            if rows.size == 0:
                continue
            node = self.nodes[node_index]
            if node.is_leaf:
                leaves[rows] = node_index
                continue
            mask = x[rows, node.feature] < node.threshold
            stack.append((node.right, rows[~mask]))
            stack.append((node.left, rows[mask]))
        return leaves

    def predict(self, x: FloatArray, *, response_features: FloatArray | None = None) -> FloatArray:
        x, design = self._validate_prediction_data(x, response_features)
        assert self.n_outputs_ is not None
        prediction = np.empty((x.shape[0], self.n_outputs_), dtype=np.float64)
        stack: list[tuple[int, NDArray[np.int64]]] = [(0, np.arange(x.shape[0], dtype=np.int64))]
        while stack:
            node_index, rows = stack.pop()
            if rows.size == 0:
                continue
            node = self.nodes[node_index]
            if node.is_leaf:
                if self.objective.linear_response_mode is not None:
                    assert design is not None
                    prediction[rows] = np.dot(design[rows], node.value)
                else:
                    prediction[rows] = node.value
                continue
            mask = x[rows, node.feature] < node.threshold
            stack.append((node.right, rows[~mask]))
            stack.append((node.left, rows[mask]))
        return prediction

    def refit_leaves(self, x: FloatArray, residual: FloatArray) -> None:
        if not self.objective.supports_leaf_refit:
            return
        leaves = self.apply(x)
        for node_index in np.unique(leaves):
            rows = leaves == node_index
            self.nodes[int(node_index)].value = np.asarray(
                self.objective.refit_leaf(residual[rows]) * self.learning_rate,
                dtype=np.float64,
            )

    @property
    def leaf_count(self) -> int:
        return sum(node.is_leaf for node in self.nodes)

    def to_networkx(self) -> Any:
        """Return an optional NetworkX representation for visualization."""

        try:
            import networkx as nx
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("Install mbtr[plot] to export a NetworkX tree.") from exc
        graph = nx.DiGraph()
        for index, node in enumerate(self.nodes):
            graph.add_node(
                index,
                leaf=node.is_leaf,
                response=node.value if node.is_leaf else None,
                feature=node.feature,
                threshold=node.threshold,
                sample_count=node.sample_count,
                split_gain=node.split_gain,
            )
            if not node.is_leaf:
                graph.add_edge(index, node.left, side="l")
                graph.add_edge(index, node.right, side="r")
        return graph


__all__ = ["RegressionTree", "TreeNode"]
