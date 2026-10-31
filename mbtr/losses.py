"""Objectives, construction surrogates, and leaf solvers for MBTR."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cho_solve
from scipy.special import expit

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class LeafSolution:
    """Optimal parameters for one leaf and their positive quadratic gain."""

    value: FloatArray
    gain: float


def _as_float_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _stable_solve(matrix: FloatArray, rhs: FloatArray) -> FloatArray:
    """Solve a symmetric system, with a least-squares fallback."""

    try:
        factor = np.linalg.cholesky(matrix)
        return np.asarray(cho_solve((factor, True), rhs, check_finite=False), dtype=np.float64)
    except np.linalg.LinAlgError:
        return np.asarray(np.linalg.lstsq(matrix, rhs, rcond=None)[0], dtype=np.float64)


class Loss:
    """Base objective for constant-response multivariate trees.

    ``solve_leaf`` is the single source of truth for a construction response
    and its split gain. Objectives may explicitly apply a different
    post-structure refit through ``supports_leaf_refit`` and ``refit_leaf``.
    """

    required_pars: tuple[str, ...] = ()
    requires_response_features = False
    linear_response_mode: str | None = None
    linear_gram_scale = 1.0
    requires_partitions = False
    supports_leaf_refit = False
    is_quantile = False
    constant_hessian = False

    def __init__(
        self,
        lambda_weights: float = 0.1,
        lambda_leaves: float = 0.0,
        **kwargs: Any,
    ) -> None:
        if lambda_weights < 0 or lambda_leaves < 0:
            raise ValueError("Regularization parameters must be non-negative.")
        self.lambda_weights = float(lambda_weights)
        self.lambda_leaves = float(lambda_leaves)
        self.kwargs = dict(kwargs)
        self.n_outputs: int | None = None
        self.n_internal_outputs: int | None = None

    def configure(self, n_outputs: int) -> Loss:
        if int(n_outputs) < 1:
            raise ValueError("n_outputs must be at least one.")
        self.n_outputs = int(n_outputs)
        self.n_internal_outputs = int(n_outputs)
        return self

    def set_dimension(self, n_dims: int) -> None:  # historical API
        self.configure(n_dims)

    def prepare_target(self, y: ArrayLike) -> FloatArray:
        return _as_float_matrix(y, name="y")

    def restore_prediction(self, prediction: FloatArray) -> FloatArray:
        return prediction

    def initial_prediction(self, y: FloatArray) -> FloatArray:
        return np.mean(y, axis=0)

    def get_initial_guess(self, y: ArrayLike) -> FloatArray:  # historical API
        target = self.prepare_target(y)
        if self.n_outputs is None:
            self.configure(target.shape[1])
        return self.initial_prediction(target)

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        negative_gradient = y - prediction
        unit_hessian = np.broadcast_to(np.float64(1.0), negative_gradient.shape)
        return negative_gradient, unit_hessian

    def partitioned_gradient_hessian(
        self,
        y: FloatArray,
        prediction: FloatArray,
        partitions: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray]:
        """Evaluate derivatives, optionally conditioned on previous tree regions."""

        del partitions
        return self.gradient_hessian(y, prediction)

    def get_grad_and_hessian_diags(
        self,
        y: ArrayLike,
        y_hat: ArrayLike,
        iteration: int = 0,
        leaves_idx: Any = None,
    ) -> tuple[FloatArray, FloatArray]:  # historical API
        del iteration, leaves_idx
        negative_gradient, hessian = self.gradient_hessian(
            _as_float_matrix(y, name="y"), _as_float_matrix(y_hat, name="y_hat")
        )
        return -negative_gradient, hessian

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        *,
        linear_gram: FloatArray | None = None,
        linear_gradient: FloatArray | None = None,
    ) -> LeafSolution:
        del count, linear_gram, linear_gradient
        denominator = np.maximum(
            np.asarray(hessian_sum, dtype=np.float64) + self.lambda_weights,
            np.finfo(np.float64).eps,
        )
        response = np.asarray(gradient_sum, dtype=np.float64) / denominator
        gain = 0.5 * float(np.dot(np.asarray(gradient_sum), response))
        return LeafSolution(response, gain)

    def eval_optimal_response(self, gradient_sum: ArrayLike, hessian_sum: ArrayLike) -> FloatArray:
        return self.solve_leaf(
            np.asarray(gradient_sum, dtype=np.float64),
            np.asarray(hessian_sum, dtype=np.float64),
            1,
        ).value

    def eval_optimal_loss(self, gradient_squared: ArrayLike, hessian_sum: ArrayLike) -> float:
        """Legacy diagonal-objective helper; the new engine never squares G."""

        denominator = np.maximum(
            np.asarray(hessian_sum, dtype=np.float64) + self.lambda_weights,
            np.finfo(float).eps,
        )
        return -0.5 * float(np.sum(np.asarray(gradient_squared) / denominator))

    def refit_leaf(self, residual: FloatArray) -> FloatArray:
        return np.mean(residual, axis=0)

    def loss(self, y: FloatArray, prediction: FloatArray) -> float:
        return 0.5 * float(np.mean(np.square(y - prediction)))

    def tree_loss(self, y: ArrayLike, prediction: ArrayLike) -> float:
        target = _as_float_matrix(y, name="y")
        estimate = _as_float_matrix(prediction, name="prediction")
        return 0.5 * float(np.sum(np.square(target - estimate)))

    def eval(self, y: ArrayLike, prediction: ArrayLike, trees: Any = None) -> tuple[float, float]:
        del trees
        return self.tree_loss(y, prediction), 0.0


class MSE(Loss):
    """Squared-error objective with independent constant responses."""

    constant_hessian = True


class BinaryLogLoss(Loss):
    """Independent Bernoulli log loss evaluated on raw logits.

    The derivative formulas deliberately avoid subtracting a probability that
    has rounded to one.  This preserves the small, correctly signed training
    signal for confidently classified examples and is the binary analogue of
    cancellation-safe hard-target cross-entropy.
    """

    def prepare_target(self, y: ArrayLike) -> FloatArray:
        target = _as_float_matrix(y, name="y")
        if np.any((target != 0.0) & (target != 1.0)):
            raise ValueError("BinaryLogLoss targets must contain only 0 and 1.")
        return target

    def restore_prediction(self, prediction: FloatArray) -> FloatArray:
        return expit(prediction)

    def initial_prediction(self, y: FloatArray) -> FloatArray:
        probability = np.mean(y, axis=0)
        epsilon = np.finfo(np.float64).eps
        probability = np.clip(probability, epsilon, 1.0 - epsilon)
        return np.log(probability) - np.log1p(-probability)

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        # ``y - expit(prediction)`` loses the positive-class tail once expit
        # rounds to one.  These label-conditioned expressions are equivalent
        # in exact arithmetic and remain informative at large margins.
        negative_gradient = np.where(
            y == 1.0,
            expit(-prediction),
            -expit(prediction),
        )
        tail_probability = expit(-np.abs(prediction))
        hessian = tail_probability * (1.0 - tail_probability)
        return negative_gradient, hessian

    def loss(self, y: FloatArray, prediction: FloatArray) -> float:
        signed_target = 2.0 * y - 1.0
        return float(np.mean(np.logaddexp(0.0, -signed_target * prediction)))

    def tree_loss(self, y: ArrayLike, prediction: ArrayLike) -> float:
        target = self.prepare_target(y)
        estimate = _as_float_matrix(prediction, name="prediction")
        if estimate.shape != target.shape:
            raise ValueError("prediction must have the same shape as y.")
        signed_target = 2.0 * target - 1.0
        return float(np.sum(np.logaddexp(0.0, -signed_target * estimate)))


class TimeSmoother(MSE):
    """Squared error with a second-difference penalty per leaf response."""

    required_pars = ("lambda_smooth",)

    def __init__(self, lambda_smooth: float | None = None, **kwargs: Any) -> None:
        if lambda_smooth is None:
            lambda_smooth = kwargs.pop("smoothness", None)
        if lambda_smooth is None:
            raise TypeError("TimeSmoother requires lambda_smooth.")
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative.")
        super().__init__(**kwargs)
        self.lambda_smooth = float(lambda_smooth)
        self.difference_matrix_: FloatArray | None = None
        self.penalty_matrix_: FloatArray | None = None

    def configure(self, n_outputs: int) -> TimeSmoother:
        super().configure(n_outputs)
        difference = (
            np.zeros((0, n_outputs), dtype=np.float64)
            if n_outputs < 3
            else np.diff(np.eye(n_outputs, dtype=np.float64), n=2, axis=0)
        )
        self.difference_matrix_ = difference
        self.penalty_matrix_ = self.lambda_smooth * (difference.T @ difference)
        return self

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        **kwargs: Any,
    ) -> LeafSolution:
        del count, kwargs
        if self.penalty_matrix_ is None:
            raise RuntimeError("The objective has not been configured.")
        system = np.diag(np.asarray(hessian_sum) + self.lambda_weights)
        system += self.penalty_matrix_
        response = _stable_solve(system, np.asarray(gradient_sum, dtype=np.float64))
        return LeafSolution(response, 0.5 * float(np.dot(gradient_sum, response)))


class FourierLoss(MSE):
    """Squared error constrained to a non-redundant Fourier subspace."""

    required_pars = ("n_harmonics",)

    def __init__(self, n_harmonics: int | None = None, **kwargs: Any) -> None:
        if n_harmonics is None:
            raise TypeError("FourierLoss requires n_harmonics.")
        if int(n_harmonics) < 0:
            raise ValueError("n_harmonics must be non-negative.")
        super().__init__(**kwargs)
        self.n_harmonics = int(n_harmonics)
        self.basis_: FloatArray | None = None

    def configure(self, n_outputs: int) -> FourierLoss:
        super().configure(n_outputs)
        time = np.arange(n_outputs, dtype=np.float64)
        columns: list[FloatArray] = [np.ones(n_outputs, dtype=np.float64)]
        for harmonic in range(1, self.n_harmonics + 1):
            phase = 2.0 * np.pi * harmonic * time / n_outputs
            columns.extend((np.cos(phase), np.sin(phase)))
        candidate = np.column_stack(columns)
        left_vectors, singular_values, _ = np.linalg.svd(candidate, full_matrices=False)
        tolerance = np.finfo(float).eps * max(candidate.shape) * singular_values[0]
        self.basis_ = left_vectors[:, : int(np.sum(singular_values > tolerance))]
        return self

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        **kwargs: Any,
    ) -> LeafSolution:
        del count, kwargs
        if self.basis_ is None:
            raise RuntimeError("The objective has not been configured.")
        system = self.basis_.T @ (np.asarray(hessian_sum)[:, None] * self.basis_)
        system += self.lambda_weights * np.eye(system.shape[0])
        projected_gradient = self.basis_.T @ np.asarray(gradient_sum)
        weights = _stable_solve(system, projected_gradient)
        response = self.basis_ @ weights
        return LeafSolution(response, 0.5 * float(np.dot(projected_gradient, weights)))


class LatentVariable(MSE):
    """Generalized least squares response constrained to the span of ``S``."""

    required_pars = ("S", "precision")

    def __init__(
        self,
        S: ArrayLike | None = None,
        precision: ArrayLike | None = None,
        **kwargs: Any,
    ) -> None:
        if S is None or precision is None:
            raise TypeError("LatentVariable requires S and precision.")
        super().__init__(**kwargs)
        self.S = _as_float_matrix(S, name="S")
        self.precision = _as_float_matrix(precision, name="precision")
        if self.precision.shape[0] != self.precision.shape[1]:
            raise ValueError("precision must be square.")
        if not np.allclose(self.precision, self.precision.T, rtol=1e-10, atol=1e-12):
            raise ValueError("precision must be symmetric.")
        if self.S.shape[0] != self.precision.shape[0]:
            raise ValueError("S rows must match precision dimensions.")
        if np.min(np.linalg.eigvalsh(self.precision)) < -1e-10:
            raise ValueError("precision must be positive semidefinite.")
        self.parameter_hessian_: FloatArray | None = None

    def configure(self, n_outputs: int) -> LatentVariable:
        if n_outputs != self.S.shape[0]:
            raise ValueError(f"y has {n_outputs} outputs but S has {self.S.shape[0]} rows.")
        super().configure(n_outputs)
        self.parameter_hessian_ = self.S.T @ self.precision @ self.S
        return self

    def initial_prediction(self, y: FloatArray) -> FloatArray:
        if self.parameter_hessian_ is None:
            raise RuntimeError("The objective has not been configured.")
        rhs = self.S.T @ self.precision @ np.mean(y, axis=0)
        weights = np.linalg.lstsq(self.parameter_hessian_, rhs, rcond=None)[0]
        return self.S @ weights

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        return (y - prediction) @ self.precision.T, np.ones_like(y)

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        **kwargs: Any,
    ) -> LeafSolution:
        del hessian_sum, kwargs
        if self.parameter_hessian_ is None:
            raise RuntimeError("The objective has not been configured.")
        system = count * self.parameter_hessian_
        system += self.lambda_weights * np.eye(system.shape[0])
        parameter_gradient = self.S.T @ np.asarray(gradient_sum)
        weights = _stable_solve(system, parameter_gradient)
        response = self.S @ weights
        return LeafSolution(response, 0.5 * float(np.dot(parameter_gradient, weights)))

    def loss(self, y: FloatArray, prediction: FloatArray) -> float:
        residual = y - prediction
        return 0.5 * float(np.mean(np.einsum("ni,ij,nj->n", residual, self.precision, residual)))


class QuantileLoss(Loss):
    """Published logistic split surrogate with exact pinball leaf refits."""

    required_pars = ("alphas",)
    supports_leaf_refit = True
    is_quantile = True

    def __init__(
        self,
        alphas: ArrayLike | None = None,
        smoothing: float = 1.0,
        min_hessian: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        if alphas is None:
            raise TypeError("QuantileLoss requires alphas.")
        super().__init__(**kwargs)
        alpha_array = np.asarray(alphas, dtype=np.float64)
        if alpha_array.ndim != 1 or alpha_array.size == 0:
            raise ValueError("alphas must be a non-empty one-dimensional sequence.")
        if np.any((alpha_array <= 0.0) | (alpha_array >= 1.0)):
            raise ValueError("Every quantile level must be strictly between zero and one.")
        if np.any(np.diff(alpha_array) <= 0.0):
            raise ValueError("alphas must be strictly increasing.")
        if smoothing <= 0 or min_hessian <= 0:
            raise ValueError("smoothing and min_hessian must be positive.")
        self.alphas = alpha_array
        self.smoothing = float(smoothing)
        self.min_hessian = float(min_hessian)
        self._alpha_internal: FloatArray | None = None

    def configure(self, n_outputs: int) -> QuantileLoss:
        super().configure(n_outputs)
        self.n_internal_outputs = n_outputs * len(self.alphas)
        self._alpha_internal = np.tile(self.alphas, n_outputs)
        return self

    def prepare_target(self, y: ArrayLike) -> FloatArray:
        target = _as_float_matrix(y, name="y")
        if self.n_outputs is None:
            self.configure(target.shape[1])
        if target.shape[1] != self.n_outputs:
            raise ValueError("Target output dimension changed after objective configuration.")
        return np.repeat(target[:, :, None], len(self.alphas), axis=2).reshape(target.shape[0], -1)

    def restore_prediction(self, prediction: FloatArray) -> FloatArray:
        if self.n_outputs is None:
            raise RuntimeError("The objective has not been configured.")
        return prediction.reshape(prediction.shape[0], self.n_outputs, len(self.alphas))

    def initial_prediction(self, y: FloatArray) -> FloatArray:
        if self.n_outputs is None:
            raise RuntimeError("The objective has not been configured.")
        original = y[:, :: len(self.alphas)]
        return np.quantile(original, self.alphas, axis=0).T.reshape(-1)

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self._alpha_internal is None:
            raise RuntimeError("The objective has not been configured.")
        residual = y - prediction
        shift = np.log((1.0 - self._alpha_internal) / self._alpha_internal)
        probability = expit(residual / self.smoothing + shift)
        negative_gradient = probability + self._alpha_internal - 1.0
        hessian = probability * (1.0 - probability) / self.smoothing
        return negative_gradient, np.maximum(hessian, self.min_hessian)

    def refit_leaf(self, residual: FloatArray) -> FloatArray:
        if self._alpha_internal is None or residual.shape[0] == 0:
            raise ValueError("Cannot refit an unconfigured or empty quantile leaf.")
        return np.asarray(
            [np.quantile(residual[:, output], alpha) for output, alpha in enumerate(self._alpha_internal)],
            dtype=np.float64,
        )

    def loss(self, y: FloatArray, prediction: FloatArray) -> float:
        if self._alpha_internal is None:
            raise RuntimeError("The objective has not been configured.")
        residual = y - prediction
        return float(
            np.mean(
                np.maximum(
                    self._alpha_internal * residual,
                    (self._alpha_internal - 1.0) * residual,
                )
            )
        )


class QuadraticQuantileLoss(QuantileLoss):
    """Published leaf-dependent linear-quadratic quantile surrogate.

    The negative gradient follows equation 39 of Nespoli and Medici (2022)
    within regions from the preceding tree. Equation 40 is indefinite as
    printed, so split search uses its absolute curvature as a positive
    quasi-Newton stabilization. The public metric and terminal refit remain
    exact pinball loss. Partitions with residual ties or a missing side use
    the pinball subgradient because the published construction assumes
    non-zero mass on both sides of the candidate quantile.
    """

    requires_partitions = True

    def __init__(
        self,
        quadratic_scale: float = 1e5,
        denominator_floor: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        if quadratic_scale <= 0 or denominator_floor <= 0:
            raise ValueError("quadratic_scale and denominator_floor must be positive.")
        super().__init__(**kwargs)
        self.quadratic_scale = float(quadratic_scale)
        self.denominator_floor = float(denominator_floor)

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        raise ValueError("QuadraticQuantileLoss requires partition labels; use partitioned_gradient_hessian().")

    def partitioned_gradient_hessian(
        self,
        y: FloatArray,
        prediction: FloatArray,
        partitions: NDArray[np.int64],
    ) -> tuple[FloatArray, FloatArray]:
        if self._alpha_internal is None:
            raise RuntimeError("The objective has not been configured.")
        partition_array = np.asarray(partitions)
        if partition_array.ndim != 1 or partition_array.shape[0] != y.shape[0]:
            raise ValueError("partitions must be one-dimensional with one label per sample.")

        residual = y - prediction
        negative_gradient = np.empty_like(residual)
        hessian = np.empty_like(residual)
        k = self.quadratic_scale

        for partition in np.unique(partition_array):
            rows = np.flatnonzero(partition_array == partition)
            for output, alpha in enumerate(self._alpha_internal):
                values = residual[rows, output]
                lower = values < 0.0
                upper = values > 0.0
                ties = ~(lower | upper)
                lower_mass = -float(np.sum(values[lower]))
                upper_mass = float(np.sum(values[upper]))

                if (
                    np.any(ties)
                    or not np.any(lower)
                    or not np.any(upper)
                    or lower_mass <= self.denominator_floor
                    or upper_mass <= self.denominator_floor
                ):
                    local_gradient = np.where(lower, alpha - 1.0, np.where(upper, alpha, 0.0))
                    negative_gradient[rows, output] = local_gradient
                    hessian[rows, output] = self.min_hessian
                    continue

                offset = 2.0 * k / len(rows)
                negative_gradient[rows[lower], output] = alpha - 1.0 + k * values[lower] / -lower_mass - offset
                negative_gradient[rows[upper], output] = alpha + k * values[upper] / upper_mass - offset
                hessian[rows[lower], output] = k / lower_mass
                hessian[rows[upper], output] = k / upper_mass

        return negative_gradient, np.maximum(hessian, self.min_hessian)

    def get_grad_and_hessian_diags(
        self,
        y: ArrayLike,
        y_hat: ArrayLike,
        iteration: int = 0,
        leaves_idx: Any = None,
    ) -> tuple[FloatArray, FloatArray]:
        """Historical API accepting either labels or boolean leaf masks."""

        del iteration
        target = _as_float_matrix(y, name="y")
        prediction = _as_float_matrix(y_hat, name="y_hat")
        if leaves_idx is None:
            partitions: NDArray[np.int64] = np.zeros(len(target), dtype=np.int64)
        else:
            raw_partitions = np.asarray(leaves_idx)
            if raw_partitions.ndim == 1 and raw_partitions.shape[0] == len(target):
                partitions = np.asarray(raw_partitions, dtype=np.int64)
            elif raw_partitions.ndim == 2 and raw_partitions.shape[1] == len(target):
                partitions = np.full(len(target), -1, dtype=np.int64)
                for label, mask in enumerate(raw_partitions.astype(bool)):
                    partitions[mask] = label
                if np.any(partitions < 0):
                    raise ValueError("Boolean leaves_idx masks must cover every sample.")
            else:
                raise ValueError("leaves_idx must contain labels or boolean leaf masks.")
        negative_gradient, hessian = self.partitioned_gradient_hessian(target, prediction, partitions)
        return -negative_gradient, hessian


class PseudoHuberQuantileLoss(QuantileLoss):
    """Convex asymmetric pseudo-Huber quantile objective."""

    def __init__(self, delta: float = 1.0, **kwargs: Any) -> None:
        if delta <= 0:
            raise ValueError("delta must be positive.")
        super().__init__(**kwargs)
        self.delta = float(delta)

    def initial_prediction(self, y: FloatArray) -> FloatArray:
        if self.n_outputs is None:
            raise RuntimeError("The objective has not been configured.")
        return self.refit_leaf(y)

    def gradient_hessian(self, y: FloatArray, prediction: FloatArray) -> tuple[FloatArray, FloatArray]:
        if self._alpha_internal is None:
            raise RuntimeError("The objective has not been configured.")
        residual = y - prediction
        weights = np.where(residual >= 0.0, self._alpha_internal, 1.0 - self._alpha_internal)
        scale = np.hypot(residual, self.delta)
        negative_gradient = weights * residual / scale
        hessian = weights * np.square(self.delta / scale) / scale
        return negative_gradient, np.maximum(hessian, self.min_hessian)

    def refit_leaf(self, residual: FloatArray) -> FloatArray:
        if self._alpha_internal is None or residual.shape[0] == 0:
            raise ValueError("Cannot refit an unconfigured or empty quantile leaf.")
        response = np.empty(residual.shape[1], dtype=np.float64)
        for output, alpha in enumerate(self._alpha_internal):
            values = residual[:, output]
            margin = max(1.0, 40.0 * self.delta)
            lower = float(np.min(values) - margin)
            upper = float(np.max(values) + margin)
            for _ in range(80):
                midpoint = 0.5 * (lower + upper)
                centred = values - midpoint
                weights = np.where(centred >= 0.0, alpha, 1.0 - alpha)
                score = float(np.mean(weights * centred / np.hypot(centred, self.delta)))
                if score > 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            response[output] = 0.5 * (lower + upper)
        return response

    def loss(self, y: FloatArray, prediction: FloatArray) -> float:
        if self._alpha_internal is None:
            raise RuntimeError("The objective has not been configured.")
        residual = y - prediction
        weights = np.where(residual >= 0.0, self._alpha_internal, 1.0 - self._alpha_internal)
        return float(np.mean(weights * (np.hypot(residual, self.delta) - self.delta)))


class LinRegLoss(MSE):
    """Squared-error objective with a ridge response inside each leaf."""

    requires_response_features = True
    linear_response_mode = "shared"

    def __init__(self, regularize_intercept: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.regularize_intercept = bool(regularize_intercept)

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        *,
        linear_gram: FloatArray | None = None,
        linear_gradient: FloatArray | None = None,
    ) -> LeafSolution:
        del gradient_sum, hessian_sum, count
        if linear_gram is None or linear_gradient is None:
            raise ValueError("Linear leaves require Gram and gradient statistics.")
        penalty = np.full(linear_gram.shape[0], self.lambda_weights, dtype=np.float64)
        if not self.regularize_intercept:
            penalty[-1] = 0.0
        coefficients = _stable_solve(np.asarray(linear_gram) + np.diag(penalty), np.asarray(linear_gradient))
        gain = 0.5 * float(np.sum(np.asarray(linear_gradient) * coefficients))
        return LeafSolution(coefficients, gain)


class LinearBinaryLogLoss(BinaryLogLoss):
    """Bernoulli log loss with a ridge-linear logit response in each leaf.

    ``curvature="upper_bound"`` uses the global Bernoulli curvature bound
    ``p * (1 - p) <= 1/4``. This shares one Gram matrix across all labels.
    ``curvature="exact"`` uses one Hessian-weighted Gram matrix per label.
    Both modes evaluate and monotonically accept steps against the exact
    binary cross-entropy inherited from :class:`BinaryLogLoss`.
    """

    requires_response_features = True

    def __init__(
        self,
        curvature: str = "upper_bound",
        regularize_intercept: bool = False,
        **kwargs: Any,
    ) -> None:
        if curvature not in {"upper_bound", "exact"}:
            raise ValueError("curvature must be 'upper_bound' or 'exact'.")
        super().__init__(**kwargs)
        self.curvature = curvature
        self.regularize_intercept = bool(regularize_intercept)
        self.linear_response_mode = "shared" if curvature == "upper_bound" else "per_output"
        self.linear_gram_scale = 0.25 if curvature == "upper_bound" else 1.0

    def solve_leaf(
        self,
        gradient_sum: FloatArray,
        hessian_sum: FloatArray,
        count: int,
        *,
        linear_gram: FloatArray | None = None,
        linear_gradient: FloatArray | None = None,
    ) -> LeafSolution:
        del gradient_sum, hessian_sum, count
        if linear_gram is None or linear_gradient is None:
            raise ValueError("Linear-logistic leaves require Gram and gradient statistics.")
        gram = np.asarray(linear_gram, dtype=np.float64)
        projected_gradient = np.asarray(linear_gradient, dtype=np.float64)
        if projected_gradient.ndim != 2:
            raise ValueError("linear_gradient must have shape (design_width, n_outputs).")
        design_width, n_outputs = projected_gradient.shape
        penalty = np.full(design_width, self.lambda_weights, dtype=np.float64)
        if not self.regularize_intercept:
            penalty[-1] = 0.0
        regularizer = np.diag(penalty)

        if self.curvature == "upper_bound":
            if gram.shape != (design_width, design_width):
                raise ValueError("Upper-bound curvature requires one shared square Gram matrix.")
            coefficients = _stable_solve(gram + regularizer, projected_gradient)
        else:
            if gram.shape != (n_outputs, design_width, design_width):
                raise ValueError("Exact curvature requires one square Gram matrix per output.")
            coefficients = np.empty_like(projected_gradient)
            for output in range(n_outputs):
                coefficients[:, output] = _stable_solve(
                    gram[output] + regularizer,
                    projected_gradient[:, output],
                )
        gain = 0.5 * float(np.sum(projected_gradient * coefficients))
        return LeafSolution(coefficients, gain)


LOSS_MAP: Mapping[str, type[Loss]] = {
    "mse": MSE,
    "squared_error": MSE,
    "time_smoother": TimeSmoother,
    "latent_variable": LatentVariable,
    "linear_regression": LinRegLoss,
    "linear-regression": LinRegLoss,
    "fourier": FourierLoss,
    "quantile": QuantileLoss,
    "quadratic_quantile": QuadraticQuantileLoss,
    "pseudo_huber_quantile": PseudoHuberQuantileLoss,
    "pseudo-huber-quantile": PseudoHuberQuantileLoss,
}


def make_loss(
    objective: str | Loss,
    *,
    lambda_weights: float,
    lambda_leaves: float,
    objective_params: Mapping[str, Any] | None = None,
) -> Loss:
    if isinstance(objective, Loss):
        return objective
    if not isinstance(objective, str):
        raise TypeError("objective must be a registered name or a Loss instance.")
    try:
        objective_type = LOSS_MAP[objective]
    except KeyError as exc:
        choices = ", ".join(sorted(LOSS_MAP))
        raise ValueError(f"Unknown objective {objective!r}. Available: {choices}.") from exc
    parameters = dict(objective_params or {})
    parameters.setdefault("lambda_weights", lambda_weights)
    parameters.setdefault("lambda_leaves", lambda_leaves)
    return objective_type(**parameters)


__all__ = [
    "LOSS_MAP",
    "LeafSolution",
    "Loss",
    "MSE",
    "BinaryLogLoss",
    "LinearBinaryLogLoss",
    "TimeSmoother",
    "LatentVariable",
    "QuantileLoss",
    "QuadraticQuantileLoss",
    "PseudoHuberQuantileLoss",
    "FourierLoss",
    "LinRegLoss",
    "make_loss",
]
