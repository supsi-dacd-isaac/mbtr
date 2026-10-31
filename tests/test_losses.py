from __future__ import annotations

import numpy as np
import pytest

from mbtr.losses import (
    LOSS_MAP,
    MSE,
    FourierLoss,
    LatentVariable,
    LinearBinaryLogLoss,
    LinRegLoss,
    PseudoHuberQuantileLoss,
    QuadraticQuantileLoss,
    QuantileLoss,
    TimeSmoother,
)


def test_mse_leaf_matches_regularized_analytic_solution() -> None:
    gradient = np.array([6.0, -3.0])
    hessian = np.array([4.0, 2.0])
    objective = MSE(lambda_weights=0.5).configure(2)

    solution = objective.solve_leaf(gradient, hessian, count=4)
    expected = gradient / (hessian + 0.5)

    np.testing.assert_allclose(solution.value, expected)
    assert solution.gain == pytest.approx(0.5 * gradient @ expected)


def test_smoothing_zero_penalty_matches_mse() -> None:
    gradient = np.array([4.0, -2.0, 8.0, 1.0])
    hessian = np.full(4, 5.0)
    mse = MSE(lambda_weights=0.25).configure(4)
    smoother = TimeSmoother(lambda_smooth=0.0, lambda_weights=0.25).configure(4)

    expected = mse.solve_leaf(gradient, hessian, count=5)
    actual = smoother.solve_leaf(gradient, hessian, count=5)

    np.testing.assert_allclose(actual.value, expected.value)
    assert actual.gain == pytest.approx(expected.gain)


def test_smoothing_preserves_constant_mode_and_reduces_curvature() -> None:
    constant_gradient = np.full(8, 12.0)
    hessian = np.full(8, 6.0)
    weak = TimeSmoother(lambda_smooth=0.1, lambda_weights=0.2).configure(8)
    strong = TimeSmoother(lambda_smooth=100.0, lambda_weights=0.2).configure(8)

    weak_constant = weak.solve_leaf(constant_gradient, hessian, count=6).value
    strong_constant = strong.solve_leaf(constant_gradient, hessian, count=6).value
    np.testing.assert_allclose(strong_constant, weak_constant, atol=1e-12)

    irregular_gradient = np.array([0.0, 8.0, -3.0, 10.0, -5.0, 1.0, 7.0, -2.0])
    weak_value = weak.solve_leaf(irregular_gradient, hessian, count=6).value
    strong_value = strong.solve_leaf(irregular_gradient, hessian, count=6).value
    assert weak.difference_matrix_ is not None
    assert strong.difference_matrix_ is not None
    assert np.linalg.norm(strong.difference_matrix_ @ strong_value) < np.linalg.norm(
        weak.difference_matrix_ @ weak_value
    )


@pytest.mark.parametrize(
    "signal",
    [
        np.ones(12),
        np.cos(2.0 * np.pi * np.arange(12) / 12),
        np.sin(4.0 * np.pi * np.arange(12) / 12),
        1.5 + 0.8 * np.cos(2.0 * np.pi * np.arange(12) / 12) - 0.3 * np.sin(4.0 * np.pi * np.arange(12) / 12),
    ],
)
def test_fourier_leaf_exactly_recovers_in_span_signal(signal: np.ndarray) -> None:
    count = 7
    objective = FourierLoss(n_harmonics=2, lambda_weights=0.0).configure(12)
    solution = objective.solve_leaf(count * signal, np.full(12, count, dtype=float), count=count)
    np.testing.assert_allclose(solution.value, signal, atol=1e-11)


def test_fourier_orthogonal_residual_has_zero_gain() -> None:
    rng = np.random.default_rng(17)
    objective = FourierLoss(n_harmonics=1, lambda_weights=0.0).configure(10)
    assert objective.basis_ is not None
    residual = rng.normal(size=10)
    residual -= objective.basis_ @ (objective.basis_.T @ residual)

    solution = objective.solve_leaf(residual, np.ones(10), count=1)

    np.testing.assert_allclose(solution.value, 0.0, atol=1e-12)
    assert solution.gain == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.matrix_rank(objective.basis_) == objective.basis_.shape[1]


def test_latent_identity_matches_mse() -> None:
    gradient = np.array([2.0, -5.0, 1.5])
    count = 4
    regularization = 0.3
    latent = LatentVariable(S=np.eye(3), precision=np.eye(3), lambda_weights=regularization).configure(3)
    mse = MSE(lambda_weights=regularization).configure(3)

    latent_solution = latent.solve_leaf(gradient, np.full(3, count), count)
    mse_solution = mse.solve_leaf(gradient, np.full(3, count), count)

    np.testing.assert_allclose(latent_solution.value, mse_solution.value)
    assert latent_solution.gain == pytest.approx(mse_solution.gain)


def test_latent_response_stays_in_span_with_non_diagonal_precision() -> None:
    summation = np.array([[1.0], [1.0]])
    precision = np.array([[2.0, 0.3], [0.3, 1.0]])
    objective = LatentVariable(S=summation, precision=precision, lambda_weights=0.2).configure(2)
    raw_residuals = np.array([[2.0, -1.0], [1.0, 3.0], [-0.5, 2.0]])
    negative_gradient, _ = objective.gradient_hessian(raw_residuals, np.zeros_like(raw_residuals))

    solution = objective.solve_leaf(negative_gradient.sum(axis=0), np.full(2, len(raw_residuals)), len(raw_residuals))

    assert solution.value[0] == pytest.approx(solution.value[1])
    projected = summation @ np.linalg.lstsq(summation, solution.value, rcond=None)[0]
    np.testing.assert_allclose(solution.value, projected)


@pytest.mark.parametrize(
    "precision",
    [np.array([[1.0, 2.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, -1.0]])],
)
def test_latent_rejects_invalid_precision(precision: np.ndarray) -> None:
    with pytest.raises(ValueError):
        LatentVariable(S=np.eye(2), precision=precision)


def _smooth_quantile_value(
    prediction: np.ndarray,
    target: np.ndarray,
    alphas: np.ndarray,
    smoothing: float,
) -> float:
    residual = target - prediction
    shift = np.log((1.0 - alphas) / alphas)
    return float(np.sum(smoothing * np.logaddexp(0.0, residual / smoothing + shift) + (alphas - 1.0) * residual))


def _finite_difference_gradient_hessian(
    function: object, point: np.ndarray, epsilon: float = 1e-5
) -> tuple[np.ndarray, np.ndarray]:
    callable_function = function
    gradient = np.empty_like(point)
    hessian_diagonal = np.empty_like(point)
    centre = callable_function(point)  # type: ignore[operator]
    for index in range(point.size):
        step = np.zeros_like(point)
        step[index] = epsilon
        upper = callable_function(point + step)  # type: ignore[operator]
        lower = callable_function(point - step)  # type: ignore[operator]
        gradient[index] = (upper - lower) / (2.0 * epsilon)
        hessian_diagonal[index] = (upper - 2.0 * centre + lower) / epsilon**2
    return gradient, hessian_diagonal


def test_smoothed_quantile_gradient_and_hessian_match_finite_differences() -> None:
    objective = QuantileLoss(alphas=[0.2, 0.7], smoothing=0.8, min_hessian=1e-12).configure(1)
    target = np.array([[1.5, 1.5]])
    prediction = np.array([[-0.3, 0.4]])
    alphas = np.array([0.2, 0.7])

    negative_gradient, hessian = objective.gradient_hessian(target, prediction)
    numerical_gradient, numerical_hessian = _finite_difference_gradient_hessian(
        lambda values: _smooth_quantile_value(values, target[0], alphas, 0.8),
        prediction[0],
    )

    np.testing.assert_allclose(negative_gradient[0], -numerical_gradient, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(hessian[0], numerical_hessian, rtol=2e-4, atol=2e-5)
    residual = target - prediction
    expected_pinball = np.mean(np.maximum(alphas * residual, (alphas - 1.0) * residual))
    assert objective.loss(target, prediction) == pytest.approx(expected_pinball)


def _pseudo_huber_value(
    prediction: np.ndarray,
    target: np.ndarray,
    alphas: np.ndarray,
    delta: float,
) -> float:
    residual = target - prediction
    weights = np.where(residual >= 0.0, alphas, 1.0 - alphas)
    return float(np.sum(weights * (np.sqrt(residual**2 + delta**2) - delta)))


def test_pseudo_huber_quantile_gradient_and_hessian_match_finite_differences() -> None:
    objective = PseudoHuberQuantileLoss(alphas=[0.25, 0.8], delta=0.6, min_hessian=1e-12).configure(1)
    target = np.array([[1.2, 1.2]])
    prediction = np.array([[-0.5, 2.0]])
    alphas = np.array([0.25, 0.8])

    negative_gradient, hessian = objective.gradient_hessian(target, prediction)
    numerical_gradient, numerical_hessian = _finite_difference_gradient_hessian(
        lambda values: _pseudo_huber_value(values, target[0], alphas, 0.6),
        prediction[0],
    )

    np.testing.assert_allclose(negative_gradient[0], -numerical_gradient, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(hessian[0], numerical_hessian, rtol=2e-4, atol=2e-5)
    assert objective.loss(target, prediction) == pytest.approx(
        _pseudo_huber_value(prediction[0], target[0], alphas, 0.6) / prediction.size
    )


def test_published_quadratic_quantile_matches_partitioned_field() -> None:
    objective = QuadraticQuantileLoss(alphas=[0.25], quadratic_scale=8.0, min_hessian=1e-12).configure(1)
    target = np.array([[-3.0], [-1.0], [2.0], [4.0], [-2.0], [1.0]])
    prediction = np.zeros_like(target)
    partitions = np.array([0, 0, 0, 0, 1, 1], dtype=np.int64)

    gradient, hessian = objective.partitioned_gradient_hessian(target, prediction, partitions)
    expected_gradient = np.empty_like(target)
    expected_hessian = np.empty_like(target)
    for partition in np.unique(partitions):
        rows = np.flatnonzero(partitions == partition)
        residual = target[rows, 0]
        lower = residual < 0.0
        upper = residual > 0.0
        lower_sum = residual[lower].sum()
        upper_sum = residual[upper].sum()
        offset = 16.0 / len(rows)
        expected_gradient[rows[lower], 0] = -0.75 + 8.0 * residual[lower] / lower_sum - offset
        expected_gradient[rows[upper], 0] = 0.25 + 8.0 * residual[upper] / upper_sum - offset
        expected_hessian[rows[lower], 0] = abs(8.0 / lower_sum)
        expected_hessian[rows[upper], 0] = 8.0 / upper_sum

    np.testing.assert_allclose(gradient, expected_gradient)
    np.testing.assert_allclose(hessian, expected_hessian)
    with pytest.raises(ValueError, match="partition labels"):
        objective.gradient_hessian(target, prediction)
    with pytest.raises(ValueError, match="one label per sample"):
        objective.partitioned_gradient_hessian(target, prediction, np.array([0, 1]))


def test_published_quadratic_field_is_stationary_at_empirical_quantile() -> None:
    objective = QuadraticQuantileLoss(alphas=[0.5], quadratic_scale=8.0, min_hessian=1e-12).configure(1)
    target = np.array([[-3.0], [-1.0], [1.0], [3.0]])
    gradient, _ = objective.partitioned_gradient_hessian(
        target, np.zeros_like(target), np.zeros(len(target), dtype=np.int64)
    )

    assert np.sum(gradient) == pytest.approx(0.0)


def test_published_quadratic_quantile_fallback_is_finite_for_ties() -> None:
    objective = QuadraticQuantileLoss(alphas=[0.5]).configure(1)
    target = np.array([[-1.0], [0.0], [1.0], [2.0]])
    prediction = np.zeros_like(target)
    gradient, hessian = objective.partitioned_gradient_hessian(
        target, prediction, np.zeros(len(target), dtype=np.int64)
    )

    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.all(hessian > 0.0)
    np.testing.assert_allclose(gradient[:, 0], [-0.5, 0.0, 0.5, 0.5])
    np.testing.assert_allclose(hessian, objective.min_hessian)


@pytest.mark.parametrize("objective_type", [QuantileLoss, PseudoHuberQuantileLoss])
def test_quantile_derivatives_remain_finite_and_positive_for_extreme_residuals(
    objective_type: type[QuantileLoss],
) -> None:
    objective = objective_type(alphas=[0.1, 0.9]).configure(1)
    target = np.array([[-1e6, 1e6], [1e6, -1e6]])
    prediction = np.zeros_like(target)

    gradient, hessian = objective.gradient_hessian(target, prediction)

    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))
    assert np.all(hessian > 0.0)


@pytest.mark.parametrize(
    "objective",
    [
        QuantileLoss(alphas=[0.25, 0.5, 0.75], smoothing=0.4),
        QuadraticQuantileLoss(alphas=[0.25, 0.5, 0.75], quadratic_scale=100.0),
    ],
)
def test_multivariate_quantile_shapes_initialization_and_refit(
    objective: QuantileLoss,
) -> None:
    y = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    objective.configure(2)

    expanded = objective.prepare_target(y)
    initial = objective.initial_prediction(expanded)
    restored = objective.restore_prediction(np.broadcast_to(initial, expanded.shape).copy())

    assert expanded.shape == (4, 6)
    assert restored.shape == (4, 2, 3)
    empirical = np.quantile(y, [0.25, 0.5, 0.75], axis=0).T.reshape(-1)
    np.testing.assert_allclose(initial, empirical)

    residual = expanded - np.broadcast_to(initial, expanded.shape)
    refitted = objective.refit_leaf(residual)
    expected_refit = np.asarray(
        [np.quantile(residual[:, output], alpha) for output, alpha in enumerate(np.tile([0.25, 0.5, 0.75], 2))]
    )
    np.testing.assert_allclose(refitted, expected_refit)
    before = objective.loss(expanded, np.broadcast_to(initial, expanded.shape))
    after = objective.loss(expanded, np.broadcast_to(initial + refitted, expanded.shape))
    assert after <= before + 1e-15


def test_quantile_registry_keeps_published_and_pseudo_huber_names_distinct() -> None:
    assert LOSS_MAP["quadratic_quantile"] is QuadraticQuantileLoss
    assert LOSS_MAP["pseudo_huber_quantile"] is PseudoHuberQuantileLoss
    assert QuadraticQuantileLoss is not PseudoHuberQuantileLoss


def test_linear_leaf_matches_direct_ridge_solution_and_gain() -> None:
    design = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]], dtype=float)
    residual = np.column_stack((2.0 * design[:, 0] + 1.0, -design[:, 0] + 0.5))
    ridge = 0.4
    objective = LinRegLoss(lambda_weights=ridge, regularize_intercept=False).configure(2)
    gram = np.einsum("ni,nj->ij", design, design, optimize=True)
    linear_gradient = design.T @ residual

    solution = objective.solve_leaf(
        residual.sum(axis=0),
        np.full(2, len(design)),
        len(design),
        linear_gram=gram,
        linear_gradient=linear_gradient,
    )
    expected = np.linalg.solve(gram + np.diag([ridge, 0.0]), linear_gradient)

    np.testing.assert_allclose(solution.value, expected)
    assert solution.gain == pytest.approx(0.5 * np.sum(linear_gradient * expected))


def test_linear_leaf_is_stable_for_collinear_features_with_ridge() -> None:
    feature = np.arange(1.0, 6.0)
    design = np.column_stack((feature, 2.0 * feature, np.ones_like(feature)))
    residual = np.column_stack((3.0 * feature, -feature))
    objective = LinRegLoss(lambda_weights=0.5, regularize_intercept=True).configure(2)

    solution = objective.solve_leaf(
        residual.sum(axis=0),
        np.full(2, len(design)),
        len(design),
        linear_gram=design.T @ design,
        linear_gradient=design.T @ residual,
    )

    assert np.all(np.isfinite(solution.value))
    assert solution.gain >= 0.0


@pytest.mark.parametrize("regularize_intercept", [False, True])
def test_linear_binary_upper_bound_leaf_matches_direct_ridge_solution(
    regularize_intercept: bool,
) -> None:
    design = np.array(
        [
            [-1.0, 0.5, 1.0],
            [0.0, -0.5, 1.0],
            [1.0, 1.5, 1.0],
            [2.0, -1.0, 1.0],
        ]
    )
    negative_gradient = np.array(
        [
            [0.40, -0.30],
            [-0.20, 0.10],
            [0.50, 0.25],
            [-0.10, -0.40],
        ]
    )
    ridge = 0.35
    objective = LinearBinaryLogLoss(
        curvature="upper_bound",
        lambda_weights=ridge,
        regularize_intercept=regularize_intercept,
    ).configure(negative_gradient.shape[1])
    projected_gradient = design.T @ negative_gradient
    bounded_gram = 0.25 * design.T @ design
    penalty = np.full(design.shape[1], ridge)
    if not regularize_intercept:
        penalty[-1] = 0.0

    solution = objective.solve_leaf(
        negative_gradient.sum(axis=0),
        np.full(negative_gradient.shape[1], len(design) / 4.0),
        len(design),
        linear_gram=bounded_gram,
        linear_gradient=projected_gradient,
    )
    expected = np.linalg.solve(
        bounded_gram + np.diag(penalty),
        projected_gradient,
    )

    np.testing.assert_allclose(solution.value, expected, rtol=1e-13, atol=1e-14)
    assert solution.gain == pytest.approx(0.5 * np.sum(projected_gradient * expected))


@pytest.mark.parametrize("regularize_intercept", [False, True])
def test_linear_binary_exact_leaf_matches_direct_per_label_newton_solution(
    regularize_intercept: bool,
) -> None:
    design = np.array(
        [
            [-1.0, 0.5, 1.0],
            [0.0, -0.5, 1.0],
            [1.0, 1.5, 1.0],
            [2.0, -1.0, 1.0],
        ]
    )
    negative_gradient = np.array(
        [
            [0.40, -0.30],
            [-0.20, 0.10],
            [0.50, 0.25],
            [-0.10, -0.40],
        ]
    )
    hessian = np.array(
        [
            [0.08, 0.23],
            [0.25, 0.05],
            [0.12, 0.19],
            [0.04, 0.14],
        ]
    )
    ridge = 0.2
    objective = LinearBinaryLogLoss(
        curvature="exact",
        lambda_weights=ridge,
        regularize_intercept=regularize_intercept,
    ).configure(negative_gradient.shape[1])
    projected_gradient = design.T @ negative_gradient
    exact_grams = np.einsum("ni,no,nj->oij", design, hessian, design)
    penalty = np.full(design.shape[1], ridge)
    if not regularize_intercept:
        penalty[-1] = 0.0
    expected = np.column_stack(
        [
            np.linalg.solve(exact_grams[output] + np.diag(penalty), projected_gradient[:, output])
            for output in range(negative_gradient.shape[1])
        ]
    )

    solution = objective.solve_leaf(
        negative_gradient.sum(axis=0),
        hessian.sum(axis=0),
        len(design),
        linear_gram=exact_grams,
        linear_gradient=projected_gradient,
    )

    np.testing.assert_allclose(solution.value, expected, rtol=1e-13, atol=1e-14)
    assert solution.gain == pytest.approx(0.5 * np.sum(projected_gradient * expected))


def test_linear_binary_exact_and_upper_bound_solutions_agree_at_quarter_curvature() -> None:
    design = np.array([[-2.0, 1.0], [-0.5, 1.0], [0.75, 1.0], [1.5, 1.0]])
    negative_gradient = np.array([[0.2, -0.4], [0.3, 0.1], [-0.1, 0.35], [-0.25, -0.05]])
    projected_gradient = design.T @ negative_gradient
    bounded_gram = 0.25 * design.T @ design
    common = dict(lambda_weights=0.15, regularize_intercept=False)
    upper = (
        LinearBinaryLogLoss(curvature="upper_bound", **common)
        .configure(2)
        .solve_leaf(
            negative_gradient.sum(axis=0),
            np.full(2, len(design) / 4.0),
            len(design),
            linear_gram=bounded_gram,
            linear_gradient=projected_gradient,
        )
    )
    exact = (
        LinearBinaryLogLoss(curvature="exact", **common)
        .configure(2)
        .solve_leaf(
            negative_gradient.sum(axis=0),
            np.full(2, len(design) / 4.0),
            len(design),
            linear_gram=np.repeat(bounded_gram[None, :, :], 2, axis=0),
            linear_gradient=projected_gradient,
        )
    )

    np.testing.assert_allclose(exact.value, upper.value, rtol=1e-14, atol=1e-14)
    assert exact.gain == pytest.approx(upper.gain, rel=1e-14, abs=1e-14)
