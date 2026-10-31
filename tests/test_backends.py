from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from mbtr import MBTClassifier, MBTRegressor

os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "mbtr-numba-test-cache"))
NUMBA_AVAILABLE = importlib.util.find_spec("numba") is not None
requires_numba = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba is an optional mbtr[fast] dependency")


def _ordinary_numpy_histograms(
    bins: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    gradient_hist = np.column_stack(
        [np.bincount(bins, weights=gradient[:, output], minlength=n_bins) for output in range(gradient.shape[1])]
    )
    hessian_hist = np.column_stack(
        [np.bincount(bins, weights=hessian[:, output], minlength=n_bins) for output in range(hessian.shape[1])]
    )
    return counts, gradient_hist, hessian_hist


def _linear_numpy_histograms(
    bins: np.ndarray,
    design: np.ndarray,
    gradient: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    gram = np.zeros((n_bins, design.shape[1], design.shape[1]), dtype=np.float64)
    linear_gradient = np.zeros((n_bins, design.shape[1], gradient.shape[1]), dtype=np.float64)
    for bin_index in range(n_bins):
        selected = bins == bin_index
        local_design = design[selected]
        gram[bin_index] = local_design.T @ local_design
        linear_gradient[bin_index] = local_design.T @ gradient[selected]
    return counts, gram, linear_gradient


def _linear_logistic_numpy_histograms(
    bins: np.ndarray,
    design: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.bincount(bins, minlength=n_bins).astype(np.int64)
    gram = np.zeros(
        (n_bins, gradient.shape[1], design.shape[1], design.shape[1]),
        dtype=np.float64,
    )
    linear_gradient = np.zeros(
        (n_bins, design.shape[1], gradient.shape[1]),
        dtype=np.float64,
    )
    for bin_index in range(n_bins):
        selected = bins == bin_index
        local_design = np.asarray(design[selected], dtype=np.float64)
        local_gradient = np.asarray(gradient[selected], dtype=np.float64)
        local_hessian = np.asarray(hessian[selected], dtype=np.float64)
        linear_gradient[bin_index] = local_design.T @ local_gradient
        for output in range(gradient.shape[1]):
            gram[bin_index, output] = local_design.T @ (local_hessian[:, output, None] * local_design)
    return counts, gram, linear_gradient


def _reference_best_constant_hessian_split(
    counts: np.ndarray,
    gradient_hist: np.ndarray,
    n_bins_by_feature: np.ndarray,
    *,
    node_count: int,
    min_samples_leaf: int,
    lambda_weights: float,
    parent_gain: float,
    lambda_leaves: float,
) -> tuple[int, int, float, np.ndarray, float, np.ndarray, float]:
    """Reference the feature-local argmax and cross-feature tie rule."""

    best_feature = -1
    best_bin = -1
    best_split_gain = -np.inf
    best_left_gain = 0.0
    best_right_gain = 0.0
    best_left_value = np.zeros(gradient_hist.shape[2], dtype=np.float64)
    best_right_value = np.zeros(gradient_hist.shape[2], dtype=np.float64)
    epsilon = np.finfo(np.float64).eps

    for feature, n_bins_value in enumerate(n_bins_by_feature):
        n_bins = int(n_bins_value)
        if n_bins < 2:
            continue
        local_counts = counts[feature, :n_bins]
        local_gradient = gradient_hist[feature, :n_bins]
        left_counts = np.cumsum(local_counts[:-1])
        right_counts = node_count - left_counts
        valid = (left_counts >= min_samples_leaf) & (right_counts >= min_samples_leaf)
        if not np.any(valid):
            continue

        left_gradient = np.cumsum(local_gradient, axis=0)[:-1]
        total_gradient = np.sum(local_gradient, axis=0)
        right_gradient = total_gradient - left_gradient
        left_denominator = np.maximum(left_counts + lambda_weights, epsilon)
        right_denominator = np.maximum(right_counts + lambda_weights, epsilon)
        left_values = left_gradient / left_denominator[:, None]
        right_values = right_gradient / right_denominator[:, None]
        left_gains = 0.5 * np.sum(left_gradient * left_values, axis=1)
        right_gains = 0.5 * np.sum(right_gradient * right_values, axis=1)
        split_gains = left_gains + right_gains - parent_gain - lambda_leaves
        split_gains = np.where(valid, split_gains, -np.inf)
        local_bin = int(np.argmax(split_gains))
        local_gain = float(split_gains[local_bin])
        if best_feature < 0 or local_gain > best_split_gain + 1e-15:
            best_feature = feature
            best_bin = local_bin
            best_split_gain = local_gain
            best_left_gain = float(left_gains[local_bin])
            best_right_gain = float(right_gains[local_bin])
            best_left_value = left_values[local_bin].copy()
            best_right_value = right_values[local_bin].copy()

    return (
        best_feature,
        best_bin,
        best_split_gain,
        best_left_value,
        best_left_gain,
        best_right_value,
        best_right_gain,
    )


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_ordinary_histogram_matches_numpy(dtype: type[np.floating]) -> None:
    from mbtr._numba import histogram_statistics

    rng = np.random.default_rng(42)
    bins = rng.integers(0, 7, size=83, dtype=np.int64)
    gradient = rng.normal(size=(83, 4)).astype(dtype)
    hessian = rng.uniform(0.01, 2.0, size=(83, 4)).astype(dtype)
    expected = _ordinary_numpy_histograms(bins, gradient, hessian, n_bins=7)

    actual = histogram_statistics(bins, gradient, hessian, 7)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual[2], expected[2], rtol=1e-13, atol=1e-13)


@requires_numba
def test_numba_all_feature_unit_hessian_histogram_matches_feature_kernels() -> None:
    from mbtr._numba import (
        histogram_gradient_statistics,
        histogram_gradient_statistics_all_features,
    )

    rng = np.random.default_rng(43)
    binned_x = rng.integers(0, 7, size=(83, 6), dtype=np.uint8)
    gradient = rng.normal(size=(83, 4))

    counts, gradients = histogram_gradient_statistics_all_features(
        binned_x,
        gradient,
        7,
    )

    for feature in range(binned_x.shape[1]):
        expected_counts, expected_gradient = histogram_gradient_statistics(
            binned_x[:, feature],
            gradient,
            7,
        )
        np.testing.assert_array_equal(counts[feature], expected_counts)
        np.testing.assert_array_equal(gradients[feature], expected_gradient)


@requires_numba
def test_numba_indexed_all_feature_histogram_matches_materialized_rows() -> None:
    from mbtr._numba import (
        histogram_gradient_statistics_all_features,
        histogram_gradient_statistics_all_features_indexed,
    )

    rng = np.random.default_rng(44)
    binned_x = rng.integers(0, 9, size=(97, 7), dtype=np.uint8)
    gradient = rng.normal(size=(97, 5))
    rows = rng.choice(97, size=61, replace=False).astype(np.int64)

    expected = histogram_gradient_statistics_all_features(
        binned_x[rows],
        gradient[rows],
        9,
    )
    actual = histogram_gradient_statistics_all_features_indexed(
        binned_x,
        gradient,
        rows,
        9,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


@requires_numba
def test_numba_fused_constant_split_matches_reference() -> None:
    from mbtr._numba import best_constant_hessian_split

    rng = np.random.default_rng(45)
    n_samples = 73
    n_outputs = 4
    n_bins_by_feature = np.asarray([2, 6, 4, 5], dtype=np.int64)
    counts = np.zeros((4, 7), dtype=np.int64)
    gradient_hist = np.zeros((4, 7, n_outputs), dtype=np.float64)
    gradient = rng.normal(size=(n_samples, n_outputs))
    for feature, n_bins in enumerate(n_bins_by_feature):
        bins = rng.integers(0, int(n_bins), size=n_samples)
        counts[feature, :n_bins] = np.bincount(bins, minlength=int(n_bins))
        for output in range(n_outputs):
            gradient_hist[feature, :n_bins, output] = np.bincount(
                bins,
                weights=gradient[:, output],
                minlength=int(n_bins),
            )

    parameters = dict(
        node_count=n_samples,
        min_samples_leaf=8,
        lambda_weights=0.3,
        parent_gain=0.17,
        lambda_leaves=0.02,
    )
    expected = _reference_best_constant_hessian_split(
        counts,
        gradient_hist,
        n_bins_by_feature,
        **parameters,
    )
    actual = best_constant_hessian_split(
        counts,
        gradient_hist,
        n_bins_by_feature,
        parameters["node_count"],
        parameters["min_samples_leaf"],
        parameters["lambda_weights"],
        parameters["parent_gain"],
        parameters["lambda_leaves"],
    )

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert actual[2] == pytest.approx(expected[2], rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(actual[3], expected[3], rtol=2e-13, atol=2e-13)
    assert actual[4] == pytest.approx(expected[4], rel=2e-13, abs=2e-13)
    np.testing.assert_allclose(actual[5], expected[5], rtol=2e-13, atol=2e-13)
    assert actual[6] == pytest.approx(expected[6], rel=2e-13, abs=2e-13)


@requires_numba
def test_numba_fused_constant_split_preserves_first_ties_and_ignores_padding() -> None:
    from mbtr._numba import best_constant_hessian_split

    counts = np.asarray(
        [
            [1, 1, 1, 10_000],
            [1, 1, 1, 10_000],
        ],
        dtype=np.int64,
    )
    gradient_hist = np.asarray(
        [
            [[-1.0], [0.0], [1.0], [1.0e12]],
            [[-1.0], [0.0], [1.0], [-1.0e12]],
        ]
    )

    result = best_constant_hessian_split(
        counts,
        gradient_hist,
        np.asarray([3, 3], dtype=np.int64),
        3,
        1,
        0.0,
        0.0,
        0.0,
    )

    # Both thresholds within each feature and both features have equal gain.
    assert result[0] == 0
    assert result[1] == 0
    assert result[2] == pytest.approx(0.75)
    np.testing.assert_allclose(result[3], [-1.0])
    np.testing.assert_allclose(result[5], [0.5])


@requires_numba
def test_numba_fused_constant_split_reports_no_valid_candidate() -> None:
    from mbtr._numba import best_constant_hessian_split

    result = best_constant_hessian_split(
        np.asarray([[1, 3]], dtype=np.int64),
        np.asarray([[[-1.0], [1.0]]]),
        np.asarray([2], dtype=np.int64),
        4,
        2,
        0.0,
        0.0,
        0.0,
    )

    assert result[0] == -1
    assert result[1] == -1
    assert result[2] == -np.inf
    np.testing.assert_array_equal(result[3], np.zeros(1))
    np.testing.assert_array_equal(result[5], np.zeros(1))


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_linear_histogram_matches_numpy(dtype: type[np.floating]) -> None:
    from mbtr._numba import histogram_linear_statistics

    rng = np.random.default_rng(7)
    bins = rng.integers(0, 5, size=71, dtype=np.int64)
    design = rng.normal(size=(71, 4)).astype(dtype)
    design[:, -1] = 1.0
    gradient = rng.normal(size=(71, 3)).astype(dtype)
    expected = _linear_numpy_histograms(bins, design, gradient, n_bins=5)

    actual = histogram_linear_statistics(bins, design, gradient, 5)

    np.testing.assert_array_equal(actual[0], expected[0])
    tolerance = 3e-6 if dtype is np.float32 else 2e-13
    np.testing.assert_allclose(actual[1], expected[1], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual[2], expected[2], rtol=tolerance, atol=tolerance)


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numba_exact_linear_logistic_histogram_matches_numpy(
    dtype: type[np.floating],
) -> None:
    from mbtr._numba import histogram_linear_logistic_statistics

    rng = np.random.default_rng(71)
    bins = rng.integers(0, 6, size=79, dtype=np.int64)
    design = rng.normal(size=(79, 4)).astype(dtype)
    design[:, -1] = 1.0
    gradient = rng.normal(size=(79, 3)).astype(dtype)
    hessian = rng.uniform(1e-4, 0.25, size=(79, 3)).astype(dtype)
    expected = _linear_logistic_numpy_histograms(
        bins,
        design,
        gradient,
        hessian,
        n_bins=6,
    )

    actual = histogram_linear_logistic_statistics(
        bins,
        design,
        gradient,
        hessian,
        6,
    )

    np.testing.assert_array_equal(actual[0], expected[0])
    tolerance = 4e-6 if dtype is np.float32 else 3e-13
    np.testing.assert_allclose(actual[1], expected[1], rtol=tolerance, atol=tolerance)
    np.testing.assert_allclose(actual[2], expected[2], rtol=tolerance, atol=tolerance)


def _dataset(dtype: type[np.floating]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(96, 4)).astype(dtype)
    response_features = np.column_stack((x[:, 0], x[:, 1], np.sin(x[:, 2]))).astype(dtype)
    y = np.column_stack(
        (
            np.where(x[:, 0] < 0.0, -2.0, 2.0) + 0.4 * x[:, 1],
            np.where(x[:, 2] < 0.2, 1.5, -1.0) - 0.3 * x[:, 3],
        )
    ).astype(dtype)
    return x, y, response_features


def _assert_same_tree_structure_and_values(numpy_model: MBTRegressor, numba_model: MBTRegressor) -> None:
    assert len(numpy_model.trees_) == len(numba_model.trees_)
    for numpy_tree, numba_tree in zip(numpy_model.trees_, numba_model.trees_, strict=True):
        assert len(numpy_tree.nodes) == len(numba_tree.nodes)
        for numpy_node, numba_node in zip(numpy_tree.nodes, numba_tree.nodes, strict=True):
            assert numpy_node.feature == numba_node.feature
            if np.isnan(numpy_node.threshold):
                assert np.isnan(numba_node.threshold)
            else:
                assert numpy_node.threshold == pytest.approx(numba_node.threshold, abs=0.0)
            assert numpy_node.left == numba_node.left
            assert numpy_node.right == numba_node.right
            assert numpy_node.sample_count == numba_node.sample_count
            assert numpy_node.split_gain == pytest.approx(numba_node.split_gain, rel=2e-12, abs=2e-12)
            np.testing.assert_allclose(numpy_node.value, numba_node.value, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("linear", [False, True])
def test_histogram_cache_budget_fallback_preserves_predictions(
    monkeypatch: pytest.MonkeyPatch,
    linear: bool,
) -> None:
    from mbtr._tree import RegressionTree

    x, y, response_features = _dataset(np.float64)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.15,
        max_bins=10,
        min_samples_leaf=8,
        max_depth=3,
        l2_regularization=0.2,
        early_stopping_rounds=None,
        engine="numpy",
    )
    if linear:
        cached = MBTRegressor(objective="linear_regression", **parameters).fit(
            x,
            y,
            response_features=response_features,
        )
        cached_prediction = cached.predict(x, response_features=response_features)
    else:
        cached = MBTRegressor(**parameters).fit(x, y)
        cached_prediction = cached.predict(x)
    assert all(tree.histogram_subtraction_ for tree in cached.trees_)

    monkeypatch.setattr(RegressionTree, "_HISTOGRAM_CACHE_MAX_BYTES", 0)
    if linear:
        uncached = MBTRegressor(objective="linear_regression", **parameters).fit(
            x,
            y,
            response_features=response_features,
        )
        uncached_prediction = uncached.predict(x, response_features=response_features)
    else:
        uncached = MBTRegressor(**parameters).fit(x, y)
        uncached_prediction = uncached.predict(x)
    assert all(not tree.histogram_subtraction_ for tree in uncached.trees_)
    np.testing.assert_allclose(cached_prediction, uncached_prediction, rtol=2e-11, atol=2e-11)


def test_exact_linear_logistic_layout_uses_histogram_cache_fallback() -> None:
    x, regression_target, response_features = _dataset(np.float64)
    labels = (regression_target > np.median(regression_target, axis=0)).astype(np.float64)
    model = MBTClassifier(
        n_estimators=2,
        learning_rate=0.15,
        max_bins=10,
        min_samples_leaf=8,
        max_depth=2,
        l2_regularization=0.2,
        early_stopping_rounds=None,
        engine="numpy",
        linear_response_curvature="exact",
    ).fit(x, labels, response_features=response_features)

    assert all(not tree.histogram_subtraction_ for tree in model.trees_)


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_ordinary_tree_and_predictions_match_between_engines(
    dtype: type[np.floating],
) -> None:
    x, y, _ = _dataset(dtype)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.2,
        max_bins=12,
        min_samples_leaf=7,
        max_depth=3,
        l2_regularization=0.2,
        early_stopping_rounds=None,
        random_state=0,
    )

    numpy_model = MBTRegressor(engine="numpy", **parameters).fit(x, y)
    numba_model = MBTRegressor(engine="numba", **parameters).fit(x, y)

    assert numpy_model.engine_ == "numpy"
    assert numba_model.engine_ == "numba"
    _assert_same_tree_structure_and_values(numpy_model, numba_model)
    np.testing.assert_allclose(numpy_model.predict(x), numba_model.predict(x), rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(numpy_model.apply(x), numba_model.apply(x))


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_linear_tree_and_predictions_match_between_engines(
    dtype: type[np.floating],
) -> None:
    x, y, response_features = _dataset(dtype)
    parameters = dict(
        objective="linear_regression",
        n_estimators=2,
        learning_rate=0.15,
        max_bins=10,
        min_samples_leaf=9,
        max_depth=2,
        l2_regularization=0.5,
        early_stopping_rounds=None,
        random_state=0,
    )

    numpy_model = MBTRegressor(engine="numpy", **parameters).fit(x, y, response_features=response_features)
    numba_model = MBTRegressor(engine="numba", **parameters).fit(x, y, response_features=response_features)

    _assert_same_tree_structure_and_values(numpy_model, numba_model)
    np.testing.assert_allclose(
        numpy_model.predict(x, response_features=response_features),
        numba_model.predict(x, response_features=response_features),
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_array_equal(numpy_model.apply(x), numba_model.apply(x))


@requires_numba
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fitted_multilabel_classifier_matches_between_engines(
    dtype: type[np.floating],
) -> None:
    x, y_continuous, _ = _dataset(dtype)
    y = np.column_stack((y_continuous[:, 0] > 0.0, y_continuous[:, 1] > 0.0)).astype(np.int64)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.2,
        max_bins=12,
        min_samples_leaf=7,
        max_depth=3,
        l2_regularization=0.2,
        early_stopping_rounds=None,
        random_state=0,
    )

    numpy_model = MBTClassifier(engine="numpy", **parameters).fit(x, y)
    numba_model = MBTClassifier(engine="numba", **parameters).fit(x, y)

    _assert_same_tree_structure_and_values(numpy_model, numba_model)
    np.testing.assert_allclose(
        numpy_model.decision_function(x),
        numba_model.decision_function(x),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        numpy_model.predict_proba(x),
        numba_model.predict_proba(x),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_array_equal(numpy_model.predict(x), numba_model.predict(x))


@requires_numba
@pytest.mark.parametrize("curvature", ["upper_bound", "exact"])
def test_fitted_linear_response_classifier_matches_between_engines(
    curvature: str,
) -> None:
    x, y_continuous, response_features = _dataset(np.float64)
    y = np.column_stack(
        (
            1.3 * response_features[:, 0] - 0.4 * response_features[:, 1] > 0.0,
            -0.8 * response_features[:, 0] + response_features[:, 2] > 0.0,
        )
    ).astype(np.int64)
    parameters = dict(
        n_estimators=3,
        learning_rate=0.2,
        max_bins=12,
        min_samples_leaf=7,
        max_depth=3,
        l2_regularization=0.2,
        early_stopping_rounds=None,
        random_state=0,
        linear_response_curvature=curvature,
    )

    numpy_model = MBTClassifier(engine="numpy", **parameters).fit(
        x,
        y,
        response_features=response_features,
    )
    numba_model = MBTClassifier(engine="numba", **parameters).fit(
        x,
        y,
        response_features=response_features,
    )

    _assert_same_tree_structure_and_values(numpy_model, numba_model)
    np.testing.assert_allclose(
        numpy_model.decision_function(x, response_features=response_features),
        numba_model.decision_function(x, response_features=response_features),
        rtol=5e-11,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        numpy_model.predict_proba(x, response_features=response_features),
        numba_model.predict_proba(x, response_features=response_features),
        rtol=5e-11,
        atol=5e-11,
    )
    np.testing.assert_array_equal(
        numpy_model.predict(x, response_features=response_features),
        numba_model.predict(x, response_features=response_features),
    )
    np.testing.assert_array_equal(numpy_model.apply(x), numba_model.apply(x))


def test_auto_engine_keeps_numpy_for_small_constant_response_workloads() -> None:
    x, y, _ = _dataset(np.float64)
    model = MBTRegressor(
        engine="auto",
        n_estimators=1,
        max_bins=6,
        min_samples_leaf=12,
        max_depth=1,
        early_stopping_rounds=None,
    ).fit(x, y)

    assert model.engine_ == "numpy"


def test_auto_engine_keeps_numpy_for_small_linear_response_workloads() -> None:
    x, y, response_features = _dataset(np.float64)
    model = MBTRegressor(
        objective="linear_regression",
        engine="auto",
        n_estimators=1,
        max_bins=6,
        min_samples_leaf=12,
        max_depth=1,
        early_stopping_rounds=None,
    ).fit(x, y, response_features=response_features)

    assert model.engine_ == "numpy"


@requires_numba
def test_auto_engine_amortizes_numba_over_repeated_constant_trees() -> None:
    rng = np.random.default_rng(987)
    x = rng.normal(size=(1_000, 16))
    y = x[:, :4] + 0.05 * rng.normal(size=(len(x), 4))
    model = MBTRegressor(
        engine="auto",
        n_estimators=200,
        max_bins=8,
        min_samples_leaf=20,
        max_depth=1,
        early_stopping_rounds=None,
    ).fit(x, y)

    assert model.engine_ == "numba"


@requires_numba
def test_auto_engine_selects_numba_for_large_linear_response_workloads() -> None:
    rng = np.random.default_rng(321)
    x = rng.normal(size=(3_000, 32))
    response_features = x[:, :8]
    y = x[:, :4] + 0.05 * rng.normal(size=(len(x), 4))
    model = MBTRegressor(
        objective="linear_regression",
        engine="auto",
        n_estimators=1,
        max_bins=8,
        min_samples_leaf=20,
        max_depth=1,
        early_stopping_rounds=None,
    ).fit(x, y, response_features=response_features)

    assert model.engine_ == "numba"
