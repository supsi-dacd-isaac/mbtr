"""Public multivariate boosted-tree estimator."""

from __future__ import annotations

import copy
import importlib.util
import warnings
from collections.abc import Callable, Generator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit

from ._tree import RegressionTree
from .losses import BinaryLogLoss, LinearBinaryLogLoss, Loss, make_loss

FloatArray = NDArray[np.float64]
Callback = Callable[[Mapping[str, Any]], bool | None]


def _as_feature_matrix(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array)


def _as_target_matrix(value: ArrayLike, *, name: str) -> tuple[FloatArray, bool]:
    array = np.asarray(value, dtype=np.float64)
    was_one_dimensional = array.ndim == 1
    if was_one_dimensional:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional; got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(array), was_one_dimensional


def _fit_bin_thresholds(x: FloatArray, max_bins: int) -> tuple[FloatArray, ...]:
    levels = np.arange(1, max_bins, dtype=np.float64) / max_bins
    thresholds: list[FloatArray] = []
    for feature in range(x.shape[1]):
        column = x[:, feature]
        if np.all(column == column[0]):
            thresholds.append(np.empty(0, dtype=np.float64))
            continue
        feature_thresholds = np.unique(np.quantile(column, levels))
        feature_thresholds = feature_thresholds[
            (feature_thresholds > np.min(column)) & (feature_thresholds <= np.max(column))
        ]
        thresholds.append(np.asarray(feature_thresholds, dtype=np.float64))
    return tuple(thresholds)


def _bin_features(x: FloatArray, thresholds: Sequence[FloatArray]) -> NDArray[np.integer[Any]]:
    max_thresholds = max((len(values) for values in thresholds), default=0)
    dtype: Any = np.uint8 if max_thresholds < np.iinfo(np.uint8).max else np.uint16
    binned = np.empty(x.shape, dtype=dtype)
    for feature, feature_thresholds in enumerate(thresholds):
        # side="right" means bins <= k correspond exactly to x < threshold[k].
        binned[:, feature] = np.searchsorted(feature_thresholds, x[:, feature], side="right")
    return binned


class MBTRegressor:
    """Multivariate gradient-boosted regression trees.

    Parameters
    ----------
    objective:
        Registered objective name or a configured :class:`mbtr.losses.Loss`.
    n_estimators:
        Maximum number of boosting rounds.
    max_bins:
        Number of global quantile bins used for split candidates.
    engine:
        ``"numpy"``, ``"numba"`` or ``"auto"``.  The reference NumPy engine
        is always available; ``"auto"`` selects the fastest verified engine.
    output_sketch_size:
        Optional number of highest-gradient outputs used only for split
        search. Leaves and monotonic acceptance always use every output.
    """

    _SERIALIZATION_VERSION = 3

    def __init__(
        self,
        *,
        objective: str | Loss = "squared_error",
        objective_params: Mapping[str, Any] | None = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_bins: int = 64,
        min_samples_leaf: int = 20,
        max_depth: int | None = None,
        max_leaves: int | None = None,
        min_split_gain: float = 0.0,
        l2_regularization: float = 0.1,
        leaf_regularization: float = 0.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.0,
        random_state: int | None = None,
        engine: str = "auto",
        output_sketch_size: int | None = None,
        refit_leaves: bool = True,
        verbose: int = 0,
    ) -> None:
        self.objective = objective
        self.objective_params = objective_params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_bins = max_bins
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_split_gain = min_split_gain
        self.l2_regularization = l2_regularization
        self.leaf_regularization = leaf_regularization
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.engine = engine
        self.output_sketch_size = output_sketch_size
        self.refit_leaves = refit_leaves
        self.verbose = verbose

    def _validate_hyperparameters(self) -> None:
        if not isinstance(self.n_estimators, int) or self.n_estimators < 1:
            raise ValueError("n_estimators must be a positive integer.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if not isinstance(self.max_bins, int) or not 2 <= self.max_bins <= 65535:
            raise ValueError("max_bins must be an integer between 2 and 65535.")
        if not isinstance(self.min_samples_leaf, int) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be a positive integer.")
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 1):
            raise ValueError("max_depth must be None or a positive integer.")
        if self.max_leaves is not None and (not isinstance(self.max_leaves, int) or self.max_leaves < 2):
            raise ValueError("max_leaves must be None or an integer of at least two.")
        if self.min_split_gain < 0 or self.l2_regularization < 0 or self.leaf_regularization < 0:
            raise ValueError("Split and regularization parameters must be non-negative.")
        if self.early_stopping_rounds is not None and (
            not isinstance(self.early_stopping_rounds, int) or self.early_stopping_rounds < 1
        ):
            raise ValueError("early_stopping_rounds must be None or a positive integer.")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.engine not in {"auto", "numpy", "numba"}:
            raise ValueError("engine must be 'auto', 'numpy', or 'numba'.")
        if self.output_sketch_size is not None and (
            not isinstance(self.output_sketch_size, int)
            or isinstance(self.output_sketch_size, (bool, np.bool_))
            or self.output_sketch_size < 1
        ):
            raise ValueError("output_sketch_size must be None or a positive integer.")
        if self.verbose not in {0, 1, 2}:
            raise ValueError("verbose must be 0, 1, or 2.")

    @staticmethod
    def _validate_response_features(
        value: ArrayLike | None,
        *,
        n_samples: int,
        required: bool,
        name: str,
    ) -> FloatArray | None:
        if value is None:
            if required:
                raise ValueError(f"{name} is required by the selected objective.")
            return None
        features = _as_feature_matrix(value, name=name)
        if features.shape[0] != n_samples:
            raise ValueError(f"{name} must have {n_samples} rows; got {features.shape[0]}.")
        return features

    def _make_objective(self, n_outputs: int) -> Loss:
        objective = copy.deepcopy(
            make_loss(
                self.objective,
                lambda_weights=self.l2_regularization,
                lambda_leaves=self.leaf_regularization,
                objective_params=self.objective_params,
            )
        )
        return objective.configure(n_outputs)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        eval_set: tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        callbacks: Sequence[Callback] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> MBTRegressor:
        """Fit the boosted model.

        ``x_lr`` is accepted for one compatibility cycle; new code should use
        the keyword-only ``response_features`` argument.
        """

        self._validate_hyperparameters()
        if x_lr is not None:
            if response_features is not None:
                raise ValueError("Pass only response_features, not both response_features and x_lr.")
            warnings.warn("x_lr is deprecated; use response_features.", DeprecationWarning, stacklevel=2)
            response_features = x_lr

        x_array = _as_feature_matrix(x, name="x")
        y_array, self._target_was_one_dimensional_ = _as_target_matrix(y, name="y")
        if x_array.shape[0] != y_array.shape[0]:
            raise ValueError("x and y must contain the same number of samples.")
        objective = self._make_objective(y_array.shape[1])
        response_array = self._validate_response_features(
            response_features,
            n_samples=x_array.shape[0],
            required=objective.requires_response_features,
            name="response_features",
        )

        eval_x: FloatArray | None = None
        eval_y: FloatArray | None = None
        eval_response: FloatArray | None = None
        if eval_set is not None and self.validation_fraction:
            raise ValueError("Use either eval_set or validation_fraction, not both.")
        if eval_set is not None:
            if len(eval_set) not in {2, 3}:
                raise ValueError("eval_set must be (x, y) or (x, y, response_features).")
            eval_x = _as_feature_matrix(eval_set[0], name="eval_set[0]")
            eval_y, _ = _as_target_matrix(eval_set[1], name="eval_set[1]")
            if eval_x.shape[0] != eval_y.shape[0]:
                raise ValueError("Validation x and y must have the same number of rows.")
            if eval_x.shape[1] != x_array.shape[1] or eval_y.shape[1] != y_array.shape[1]:
                raise ValueError("Validation data must match training feature/output dimensions.")
            eval_response = self._validate_response_features(
                eval_set[2] if len(eval_set) == 3 else None,
                n_samples=eval_x.shape[0],
                required=objective.requires_response_features,
                name="eval_set response_features",
            )
            if (
                response_array is not None
                and eval_response is not None
                and eval_response.shape[1] != response_array.shape[1]
            ):
                raise ValueError(
                    "eval_set response_features must have the same number of columns as training response_features."
                )
        elif self.validation_fraction:
            n_validation = int(np.ceil(self.validation_fraction * len(x_array)))
            if n_validation < 1 or len(x_array) - n_validation < 1:
                raise ValueError("validation_fraction leaves an empty train or validation set.")
            generator = np.random.default_rng(self.random_state)
            indices = generator.permutation(len(x_array))
            validation_indices = indices[:n_validation]
            training_indices = indices[n_validation:]
            eval_x, eval_y = x_array[validation_indices], y_array[validation_indices]
            x_array, y_array = x_array[training_indices], y_array[training_indices]
            if response_array is not None:
                eval_response = response_array[validation_indices]
                response_array = response_array[training_indices]

        if len(x_array) < 2:
            raise ValueError("At least two training samples are required.")

        self.objective_ = objective
        self.n_features_in_ = x_array.shape[1]
        self.n_outputs_ = y_array.shape[1]
        self.n_response_features_ = None if response_array is None else response_array.shape[1]
        self.bin_thresholds_ = _fit_bin_thresholds(x_array, self.max_bins)
        binned_x = _bin_features(x_array, self.bin_thresholds_)
        internal_y = objective.prepare_target(y_array)
        self.base_prediction_ = np.asarray(objective.initial_prediction(internal_y), dtype=np.float64)
        prediction = np.broadcast_to(self.base_prediction_, internal_y.shape).copy()

        internal_eval_y: FloatArray | None = None
        eval_prediction: FloatArray | None = None
        if eval_y is not None:
            internal_eval_y = objective.prepare_target(eval_y)
            eval_prediction = np.broadcast_to(self.base_prediction_, internal_eval_y.shape).copy()

        self.trees_: list[RegressionTree] = []
        self.evals_result_: dict[str, list[float]] = {
            "training": [],
            "validation": [],
        }
        # Iteration ``-1`` is the fitted intercept/base prediction.  Treat it
        # as a genuine early-stopping candidate so a harmful first tree cannot
        # displace a better base-only model.  When an evaluation set is
        # present, every candidate (including the base) is scored on that same
        # set; otherwise training loss is the monitored metric.
        if eval_prediction is not None and internal_eval_y is not None:
            best_score = objective.loss(internal_eval_y, eval_prediction)
        else:
            best_score = objective.loss(internal_y, prediction)
        best_iteration = -1
        callback_list = tuple(callbacks or ())
        partitions: NDArray[np.int64] = np.zeros(len(x_array), dtype=np.int64)

        # The reference engine is authoritative.  A verified accelerated engine
        # may replace only its histogram aggregation, never objective semantics.
        numba_available = importlib.util.find_spec("numba") is not None
        if self.engine == "numba" and not numba_available:
            raise ImportError("engine='numba' requires the optional mbtr[fast] dependency.")
        if response_array is None:
            backend_work_units = x_array.shape[0] * x_array.shape[1] * internal_y.shape[1]
        else:
            design_width = response_array.shape[1] + 1
            if objective.linear_response_mode == "per_output":
                backend_work_units = (
                    x_array.shape[0] * x_array.shape[1] * internal_y.shape[1] * design_width * (design_width + 1)
                )
            else:
                backend_work_units = (
                    x_array.shape[0] * x_array.shape[1] * design_width * (design_width + internal_y.shape[1])
                )
        # Stay with NumPy until repeated histogram work is large enough to
        # repay Numba's import, dispatch, and first-compilation costs.  The
        # bins and kernel signatures are reused across boosting rounds, so the
        # estimator count is part of the amortization decision.
        amortized_backend_work_units = backend_work_units * self.n_estimators
        auto_uses_numba = numba_available and amortized_backend_work_units >= 10_000_000
        self.engine_ = "numba" if self.engine == "numba" or (self.engine == "auto" and auto_uses_numba) else "numpy"

        for iteration in range(self.n_estimators):
            if objective.requires_partitions:
                gradient, hessian = objective.partitioned_gradient_hessian(internal_y, prediction, partitions)
            else:
                gradient, hessian = objective.gradient_hessian(internal_y, prediction)
            split_gradient: FloatArray | None = None
            split_hessian: FloatArray | None = None
            split_output_indices: NDArray[np.int64] | None = None
            if self.output_sketch_size is not None and self.output_sketch_size < gradient.shape[1]:
                output_norm = np.einsum("no,no->o", gradient, gradient, optimize=True)
                output_index = np.arange(gradient.shape[1], dtype=np.int64)
                ranked = np.lexsort((output_index, -output_norm))
                split_output_indices = np.asarray(
                    ranked[: self.output_sketch_size],
                    dtype=np.int64,
                )
                split_gradient = np.ascontiguousarray(gradient[:, split_output_indices])
                split_hessian = np.ascontiguousarray(hessian[:, split_output_indices])
            tree = RegressionTree(
                objective=objective,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.max_depth,
                max_leaves=self.max_leaves,
                min_split_gain=self.min_split_gain,
                learning_rate=self.learning_rate,
                engine=self.engine_,
            ).fit(
                binned_x,
                self.bin_thresholds_,
                gradient,
                hessian,
                response_features=response_array,
                split_gradient=split_gradient,
                split_hessian=split_hessian,
                split_output_indices=split_output_indices,
            )
            if self.refit_leaves and objective.supports_leaf_refit:
                tree.refit_leaves(x_array, internal_y - prediction)
                update = tree.predict(x_array, response_features=response_array)
                tree.take_training_prediction()
            else:
                update = tree.take_training_prediction()
            previous_loss = objective.loss(internal_y, prediction)
            candidate = prediction + update
            candidate_loss = objective.loss(internal_y, candidate)

            # Backtracking makes accepted boosting rounds monotonically improve
            # the declared training metric, while preserving the fitted split.
            scale = 1.0
            while (not np.isfinite(candidate_loss) or candidate_loss > previous_loss + 1e-14) and scale > 2.0**-30:
                scale *= 0.5
                candidate = prediction + scale * update
                candidate_loss = objective.loss(internal_y, candidate)
            if not np.isfinite(candidate_loss) or candidate_loss > previous_loss + 1e-14:
                if self.verbose:
                    print(f"Stopping at iteration {iteration}: no improving step found.")
                break
            if scale != 1.0:
                for node in tree.nodes:
                    if node.is_leaf:
                        node.value *= scale

            prediction = candidate
            self.trees_.append(tree)
            self.evals_result_["training"].append(candidate_loss)
            if objective.requires_partitions:
                partitions = tree.apply(x_array)

            if eval_prediction is not None and eval_x is not None and internal_eval_y is not None:
                eval_prediction += tree.predict(eval_x, response_features=eval_response)
                monitored_score = objective.loss(internal_eval_y, eval_prediction)
                self.evals_result_["validation"].append(monitored_score)
            else:
                monitored_score = candidate_loss

            if monitored_score < best_score - 1e-14:
                best_score = monitored_score
                best_iteration = iteration

            state = {
                "iteration": iteration,
                "training_loss": candidate_loss,
                "validation_loss": None if eval_prediction is None else self.evals_result_["validation"][-1],
                "tree": tree,
                "estimator": self,
            }
            should_stop = any(callback(state) is True for callback in callback_list)
            if self.verbose == 2 or (self.verbose == 1 and iteration % 10 == 0):
                message = f"[{iteration + 1}/{self.n_estimators}] train={candidate_loss:.6g}"
                if eval_prediction is not None:
                    message += f" validation={monitored_score:.6g}"
                print(message)
            if should_stop:
                break
            if self.early_stopping_rounds is not None and iteration - best_iteration >= self.early_stopping_rounds:
                break

        if self.early_stopping_rounds is not None and best_iteration + 1 < len(self.trees_):
            self.trees_ = self.trees_[: best_iteration + 1]

        self.best_iteration_ = best_iteration
        self.best_score_ = float(best_score)
        self.n_estimators_ = len(self.trees_)
        importance = np.zeros(self.n_features_in_, dtype=np.float64)
        for tree in self.trees_:
            assert tree.feature_importances_ is not None
            importance += tree.feature_importances_
        total_importance = float(np.sum(importance))
        self.feature_importances_ = importance / total_importance if total_importance else importance
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "objective_"):
            raise RuntimeError(f"This {type(self).__name__} instance is not fitted.")

    def _validate_predict_inputs(
        self,
        x: ArrayLike,
        response_features: ArrayLike | None,
        x_lr: ArrayLike | None,
    ) -> tuple[FloatArray, FloatArray | None]:
        self._check_fitted()
        if x_lr is not None:
            if response_features is not None:
                raise ValueError("Pass only response_features, not both response_features and x_lr.")
            warnings.warn("x_lr is deprecated; use response_features.", DeprecationWarning, stacklevel=3)
            response_features = x_lr
        x_array = _as_feature_matrix(x, name="x")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(f"x must have {self.n_features_in_} features.")
        response_array = self._validate_response_features(
            response_features,
            n_samples=x_array.shape[0],
            required=self.objective_.requires_response_features,
            name="response_features",
        )
        if response_array is not None and response_array.shape[1] != self.n_response_features_:
            raise ValueError(f"response_features must have {self.n_response_features_} columns.")
        return x_array, response_array

    def _predict_internal(
        self,
        x: FloatArray,
        response_features: FloatArray | None,
        stop: int | None = None,
    ) -> FloatArray:
        prediction = np.broadcast_to(self.base_prediction_, (x.shape[0], self.base_prediction_.size)).copy()
        trees = self.trees_ if stop is None else self.trees_[:stop]
        for tree in trees:
            prediction += tree.predict(x, response_features=response_features)
        return prediction

    def predict(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        iteration_range: tuple[int, int] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> FloatArray:
        x_array, response_array = self._validate_predict_inputs(x, response_features, x_lr)
        start, stop = (0, len(self.trees_)) if iteration_range is None else iteration_range
        if start != 0 or not 0 <= stop <= len(self.trees_):
            raise ValueError("iteration_range currently supports only (0, stop) within fitted trees.")
        restored = self.objective_.restore_prediction(self._predict_internal(x_array, response_array, stop))
        if self._target_was_one_dimensional_ and not self.objective_.is_quantile:
            return restored[:, 0]
        return restored

    def staged_predict(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> Generator[FloatArray, None, None]:
        x_array, response_array = self._validate_predict_inputs(x, response_features, None)
        prediction = np.broadcast_to(self.base_prediction_, (x_array.shape[0], self.base_prediction_.size)).copy()
        for tree in self.trees_:
            prediction += tree.predict(x_array, response_features=response_array)
            restored = self.objective_.restore_prediction(prediction.copy())
            if self._target_was_one_dimensional_ and not self.objective_.is_quantile:
                restored = restored[:, 0]
            yield restored

    def apply(self, x: ArrayLike) -> NDArray[np.int64]:
        self._check_fitted()
        x_array = _as_feature_matrix(x, name="x")
        if x_array.shape[1] != self.n_features_in_:
            raise ValueError(f"x must have {self.n_features_in_} features.")
        if not self.trees_:
            return np.empty((len(x_array), 0), dtype=np.int64)
        return np.column_stack([tree.apply(x_array) for tree in self.trees_])

    def score(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> float:
        prediction = self.predict(x, response_features=response_features)
        target, _ = _as_target_matrix(y, name="y")
        if self.objective_.is_quantile:
            internal_target = self.objective_.prepare_target(target)
            internal_prediction = np.asarray(prediction).reshape(len(target), -1)
            return -self.objective_.loss(internal_target, internal_prediction)
        estimate = np.asarray(prediction)
        if estimate.ndim == 1:
            estimate = estimate[:, None]
        residual_sum = float(np.sum(np.square(target - estimate)))
        total_sum = float(np.sum(np.square(target - np.mean(target, axis=0))))
        return 1.0 - residual_sum / total_sum if total_sum else float(residual_sum == 0.0)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        del deep
        return {
            "objective": self.objective,
            "objective_params": self.objective_params,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_bins": self.max_bins,
            "min_samples_leaf": self.min_samples_leaf,
            "max_depth": self.max_depth,
            "max_leaves": self.max_leaves,
            "min_split_gain": self.min_split_gain,
            "l2_regularization": self.l2_regularization,
            "leaf_regularization": self.leaf_regularization,
            "early_stopping_rounds": self.early_stopping_rounds,
            "validation_fraction": self.validation_fraction,
            "random_state": self.random_state,
            "engine": self.engine,
            "output_sketch_size": self.output_sketch_size,
            "refit_leaves": self.refit_leaves,
            "verbose": self.verbose,
        }

    def set_params(self, **params: Any) -> MBTRegressor:
        valid = self.get_params()
        unknown = sorted(set(params) - set(valid))
        if unknown:
            raise ValueError(f"Unknown parameter(s): {', '.join(unknown)}.")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "objective_")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_serialization_version"] = self._SERIALIZATION_VERSION
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        version = state.pop("_serialization_version", None)
        if version != self._SERIALIZATION_VERSION:
            raise ValueError("Unsupported serialized MBTR model version. Refit or migrate the model.")
        self.__dict__.update(state)


class MBTClassifier(MBTRegressor):
    """Shared-tree binary and multilabel gradient-boosted classifier.

    This first classification interface accepts either a one-dimensional
    binary target or a two-dimensional multilabel indicator matrix.  It uses
    independent Bernoulli log losses while sharing every tree partition across
    outputs.  Multiclass softmax targets are intentionally not inferred from a
    one-hot matrix.

    Parameters are the common tree and boosting controls from
    :class:`MBTRegressor`, plus classification and linear-response controls.
    Supplying ``response_features`` to :meth:`fit` selects ridge-linear logit
    responses inside leaves; omitting them retains constant logit responses.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_bins: int = 64,
        min_samples_leaf: int = 20,
        max_depth: int | None = None,
        max_leaves: int | None = None,
        min_split_gain: float = 0.0,
        l2_regularization: float = 0.1,
        leaf_regularization: float = 0.0,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.0,
        random_state: int | None = None,
        engine: str = "auto",
        output_sketch_size: int | None = None,
        decision_threshold: float = 0.5,
        linear_response_curvature: str = "upper_bound",
        regularize_intercept: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            objective=BinaryLogLoss(),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_bins=max_bins,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_leaves=max_leaves,
            min_split_gain=min_split_gain,
            l2_regularization=l2_regularization,
            leaf_regularization=leaf_regularization,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=validation_fraction,
            random_state=random_state,
            engine=engine,
            output_sketch_size=output_sketch_size,
            refit_leaves=False,
            verbose=verbose,
        )
        self.decision_threshold = decision_threshold
        self.linear_response_curvature = linear_response_curvature
        self.regularize_intercept = regularize_intercept

    def _decision_threshold_value(self) -> float:
        if isinstance(self.decision_threshold, (bool, np.bool_)):
            raise ValueError("decision_threshold must be strictly between zero and one.")
        try:
            threshold = float(self.decision_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("decision_threshold must be strictly between zero and one.") from exc
        if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise ValueError("decision_threshold must be strictly between zero and one.")
        return threshold

    def _decision_logit_threshold(self) -> float:
        """Return the probability threshold in cancellation-safe logit space."""

        threshold = self._decision_threshold_value()
        return float(np.log(threshold) - np.log1p(-threshold))

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self._decision_threshold_value()
        if self.linear_response_curvature not in {"upper_bound", "exact"}:
            raise ValueError("linear_response_curvature must be 'upper_bound' or 'exact'.")
        if not isinstance(self.regularize_intercept, (bool, np.bool_)):
            raise ValueError("regularize_intercept must be Boolean.")

    def _make_objective(self, n_outputs: int) -> BinaryLogLoss:
        if getattr(self, "_linear_response_requested_", False):
            objective: BinaryLogLoss = LinearBinaryLogLoss(
                curvature=self.linear_response_curvature,
                regularize_intercept=bool(self.regularize_intercept),
                lambda_weights=self.l2_regularization,
                lambda_leaves=self.leaf_regularization,
            )
        else:
            objective = BinaryLogLoss(
                lambda_weights=self.l2_regularization,
                lambda_leaves=self.leaf_regularization,
            )
        objective.configure(n_outputs)
        return objective

    @staticmethod
    def _validate_binary_target(value: ArrayLike, *, name: str) -> FloatArray:
        try:
            target, was_one_dimensional = _as_target_matrix(value, name=name)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be a one- or two-dimensional binary indicator containing only 0 and 1."
            ) from exc
        if np.any((target != 0.0) & (target != 1.0)):
            raise ValueError(f"{name} must contain only binary values 0 and 1.")
        if was_one_dimensional:
            return np.ascontiguousarray(target[:, 0])
        return np.ascontiguousarray(target)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        eval_set: tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        callbacks: Sequence[Callback] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> MBTClassifier:
        """Fit a binary or multilabel model from hard ``0``/``1`` targets."""

        self._validate_hyperparameters()
        if response_features is not None and x_lr is not None:
            raise ValueError("Pass only response_features, not both response_features and x_lr.")
        self._linear_response_requested_ = response_features is not None or x_lr is not None
        target = self._validate_binary_target(y, name="y")
        validated_eval_set: tuple[ArrayLike, ArrayLike] | tuple[ArrayLike, ArrayLike, ArrayLike] | None = None
        if eval_set is not None:
            if self._linear_response_requested_:
                if len(eval_set) != 3:
                    raise ValueError("eval_set must be (x, y, response_features) for this MBTClassifier fit.")
                eval_x, eval_target, eval_response = eval_set
                validated_target = self._validate_binary_target(eval_target, name="eval_set[1]")
                validated_eval_set = (eval_x, validated_target, eval_response)
            else:
                if len(eval_set) != 2:
                    raise ValueError("eval_set must be (x, y) for this MBTClassifier fit.")
                eval_x, eval_target = eval_set
                validated_target = self._validate_binary_target(eval_target, name="eval_set[1]")
                validated_eval_set = (eval_x, validated_target)

        super().fit(
            x,
            target,
            response_features=response_features,
            eval_set=validated_eval_set,
            callbacks=callbacks,
            x_lr=x_lr,
        )
        self.linear_response_ = self.objective_.requires_response_features
        self.n_labels_ = self.n_outputs_
        if self._target_was_one_dimensional_:
            classes: NDArray[np.int64] | list[NDArray[np.int64]] = np.array([0, 1], dtype=np.int64)
            n_classes: int | NDArray[np.int64] = 2
        else:
            classes = [np.array([0, 1], dtype=np.int64) for _ in range(self.n_labels_)]
            n_classes = np.full(self.n_labels_, 2, dtype=np.int64)
        self.classes_ = classes
        self.n_classes_ = n_classes
        return self

    def _decision_matrix(
        self,
        x: ArrayLike,
        response_features: ArrayLike | None,
        iteration_range: tuple[int, int] | None,
        x_lr: ArrayLike | None = None,
    ) -> FloatArray:
        self._check_fitted()
        if not self.objective_.requires_response_features and (response_features is not None or x_lr is not None):
            raise ValueError("This fitted classifier uses constant leaves and does not accept response_features.")
        x_array, response_array = self._validate_predict_inputs(x, response_features, x_lr)
        start, stop = (0, len(self.trees_)) if iteration_range is None else iteration_range
        if start != 0 or not 0 <= stop <= len(self.trees_):
            raise ValueError("iteration_range currently supports only (0, stop) within fitted trees.")
        return self._predict_internal(x_array, response_array, stop)

    def decision_function(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        iteration_range: tuple[int, int] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> FloatArray:
        """Return raw Bernoulli logits before thresholding."""

        decision = self._decision_matrix(x, response_features, iteration_range, x_lr)
        if self._target_was_one_dimensional_:
            return decision[:, 0]
        return decision

    @staticmethod
    def _probabilities_from_decision(decision: FloatArray) -> FloatArray:
        if decision.ndim == 1:
            # Calculate both tails directly instead of forming ``1 - p``.
            return np.column_stack((expit(-decision), expit(decision)))
        return expit(decision)

    def predict_proba(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        iteration_range: tuple[int, int] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> FloatArray:
        """Return class probabilities for binary or positive-label probabilities."""

        return self._probabilities_from_decision(
            self.decision_function(
                x,
                response_features=response_features,
                iteration_range=iteration_range,
                x_lr=x_lr,
            )
        )

    def predict(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
        iteration_range: tuple[int, int] | None = None,
        x_lr: ArrayLike | None = None,
    ) -> NDArray[Any]:
        """Return hard labels using ``decision_threshold``."""

        decision = self.decision_function(
            x,
            response_features=response_features,
            iteration_range=iteration_range,
            x_lr=x_lr,
        )
        return np.asarray(decision >= self._decision_logit_threshold(), dtype=np.int64)

    def staged_decision_function(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> Generator[FloatArray, None, None]:
        """Yield raw logits after each accepted boosting round."""

        self._check_fitted()
        if not self.objective_.requires_response_features and response_features is not None:
            raise ValueError("This fitted classifier uses constant leaves and does not accept response_features.")
        x_array, response_array = self._validate_predict_inputs(x, response_features, None)
        prediction = np.broadcast_to(self.base_prediction_, (x_array.shape[0], self.base_prediction_.size)).copy()
        for tree in self.trees_:
            prediction += tree.predict(x_array, response_features=response_array)
            if self._target_was_one_dimensional_:
                yield prediction[:, 0].copy()
            else:
                yield prediction.copy()

    def staged_predict_proba(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> Generator[FloatArray, None, None]:
        """Yield probabilities after each accepted boosting round."""

        for decision in self.staged_decision_function(x, response_features=response_features):
            yield self._probabilities_from_decision(decision)

    def staged_predict(
        self,
        x: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> Generator[NDArray[Any], None, None]:
        """Yield hard label predictions after each accepted boosting round."""

        logit_threshold = self._decision_logit_threshold()
        for decision in self.staged_decision_function(x, response_features=response_features):
            yield np.asarray(decision >= logit_threshold, dtype=np.int64)

    def score(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        response_features: ArrayLike | None = None,
    ) -> float:
        """Return exact-match accuracy (subset accuracy for multilabel data)."""

        target = self._validate_binary_target(y, name="y")
        target_matrix = target[:, None] if target.ndim == 1 else target
        prediction = self.predict(x, response_features=response_features)
        prediction_matrix = prediction[:, None] if prediction.ndim == 1 else prediction
        if target_matrix.shape != prediction_matrix.shape:
            raise ValueError("y must match the fitted model's number of samples and output labels.")
        return float(np.mean(np.all(target_matrix == prediction_matrix, axis=1)))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        parameters = super().get_params(deep=deep)
        parameters.pop("objective")
        parameters.pop("objective_params")
        parameters.pop("refit_leaves")
        parameters["decision_threshold"] = self.decision_threshold
        parameters["linear_response_curvature"] = self.linear_response_curvature
        parameters["regularize_intercept"] = self.regularize_intercept
        return parameters

    def __setstate__(self, state: dict[str, Any]) -> None:
        state.setdefault("linear_response_curvature", "upper_bound")
        state.setdefault("regularize_intercept", False)
        super().__setstate__(state)


__all__ = ["MBTRegressor", "MBTClassifier"]
