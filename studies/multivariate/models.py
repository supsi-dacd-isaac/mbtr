"""Serializable selected-feature linear-response model wrappers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mbtr import MBTClassifier, MBTRegressor

FloatArray = NDArray[np.float64]


class SelectedLinearResponseRegressor:
    """Fit MBTR linear leaves on standardized selected input columns.

    Tree split candidates continue to use every column in ``x``. Only the
    within-leaf design matrix is restricted to ``selected_indices``.
    """

    def __init__(
        self,
        selected_indices: tuple[int, ...],
        estimator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        self.selected_indices = tuple(int(index) for index in selected_indices)
        self.estimator_parameters = dict(estimator_parameters or {})

    def _validate_x(self, value: ArrayLike, *, name: str) -> FloatArray:
        x = np.asarray(value, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be a non-empty finite two-dimensional matrix.")
        if hasattr(self, "n_features_in_") and x.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have {self.n_features_in_} columns.")
        return np.ascontiguousarray(x)

    def _design(self, x: FloatArray) -> FloatArray:
        selected = x[:, self.selected_indices_]
        return np.ascontiguousarray((selected - self.design_mean_) / self.design_scale_)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> SelectedLinearResponseRegressor:
        x_array = self._validate_x(x, name="x")
        if not self.selected_indices:
            raise ValueError("selected_indices must contain at least one feature.")
        indices = np.asarray(self.selected_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= x_array.shape[1]) or len(np.unique(indices)) != len(indices):
            raise ValueError("selected_indices must be unique valid x column positions.")
        self.n_features_in_ = x_array.shape[1]
        self.selected_indices_ = tuple(int(index) for index in indices)
        selected = x_array[:, indices]
        self.design_mean_ = np.mean(selected, axis=0)
        scale = np.std(selected, axis=0)
        self.design_scale_ = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        response_features = self._design(x_array)
        parameters = dict(self.estimator_parameters)
        parameters.pop("objective", None)
        parameters.pop("objective_params", None)
        self.linear_model_ = MBTRegressor(
            objective="linear_regression",
            objective_params={"regularize_intercept": False},
            **parameters,
        )
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            eval_x = self._validate_x(eval_set[0], name="eval_set[0]")
            fit_kwargs["eval_set"] = (eval_x, eval_set[1], self._design(eval_x))
        self.linear_model_.fit(x_array, y, response_features=response_features, **fit_kwargs)
        self.n_outputs_ = self.linear_model_.n_outputs_
        self.n_response_features_ = len(self.selected_indices_)
        self.feature_importances_ = self.linear_model_.feature_importances_.copy()
        return self

    def predict(self, x: ArrayLike) -> FloatArray:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response regressor is not fitted.")
        x_array = self._validate_x(x, name="x")
        return np.asarray(
            self.linear_model_.predict(x_array, response_features=self._design(x_array)),
            dtype=np.float64,
        )

    @property
    def engine_(self) -> str:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response regressor is not fitted.")
        return str(self.linear_model_.engine_)


class SelectedLinearResponseClassifier:
    """Fit linear-logistic MBTR leaves on standardized selected columns."""

    def __init__(
        self,
        selected_indices: tuple[int, ...],
        estimator_parameters: Mapping[str, Any] | None = None,
        *,
        linear_response_curvature: str = "upper_bound",
    ) -> None:
        self.selected_indices = tuple(int(index) for index in selected_indices)
        self.estimator_parameters = dict(estimator_parameters or {})
        self.linear_response_curvature = linear_response_curvature

    def _validate_x(self, value: ArrayLike, *, name: str) -> FloatArray:
        x = np.asarray(value, dtype=np.float64)
        if x.ndim != 2 or not len(x) or not x.shape[1] or not np.all(np.isfinite(x)):
            raise ValueError(f"{name} must be a non-empty finite two-dimensional matrix.")
        if hasattr(self, "n_features_in_") and x.shape[1] != self.n_features_in_:
            raise ValueError(f"{name} must have {self.n_features_in_} columns.")
        return np.ascontiguousarray(x)

    def _design(self, x: FloatArray) -> FloatArray:
        selected = x[:, self.selected_indices_]
        return np.ascontiguousarray((selected - self.design_mean_) / self.design_scale_)

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> SelectedLinearResponseClassifier:
        x_array = self._validate_x(x, name="x")
        if not self.selected_indices:
            raise ValueError("selected_indices must contain at least one feature.")
        indices = np.asarray(self.selected_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= x_array.shape[1]) or len(np.unique(indices)) != len(indices):
            raise ValueError("selected_indices must be unique valid x column positions.")
        self.n_features_in_ = x_array.shape[1]
        self.selected_indices_ = tuple(int(index) for index in indices)
        selected = x_array[:, indices]
        self.design_mean_ = np.mean(selected, axis=0)
        scale = np.std(selected, axis=0)
        self.design_scale_ = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
        response_features = self._design(x_array)
        parameters = dict(self.estimator_parameters)
        parameters.pop("objective", None)
        parameters.pop("objective_params", None)
        self.linear_model_ = MBTClassifier(
            linear_response_curvature=self.linear_response_curvature,
            regularize_intercept=False,
            **parameters,
        )
        fit_kwargs: dict[str, Any] = {}
        if eval_set is not None:
            eval_x = self._validate_x(eval_set[0], name="eval_set[0]")
            fit_kwargs["eval_set"] = (eval_x, eval_set[1], self._design(eval_x))
        self.linear_model_.fit(
            x_array,
            y,
            response_features=response_features,
            **fit_kwargs,
        )
        self.n_outputs_ = self.linear_model_.n_outputs_
        self.n_labels_ = self.linear_model_.n_labels_
        self.n_response_features_ = len(self.selected_indices_)
        self.feature_importances_ = self.linear_model_.feature_importances_.copy()
        return self

    def decision_function(self, x: ArrayLike) -> FloatArray:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response classifier is not fitted.")
        x_array = self._validate_x(x, name="x")
        return np.asarray(
            self.linear_model_.decision_function(
                x_array,
                response_features=self._design(x_array),
            ),
            dtype=np.float64,
        )

    def predict_proba(self, x: ArrayLike) -> FloatArray:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response classifier is not fitted.")
        x_array = self._validate_x(x, name="x")
        return np.asarray(
            self.linear_model_.predict_proba(
                x_array,
                response_features=self._design(x_array),
            ),
            dtype=np.float64,
        )

    def predict(self, x: ArrayLike) -> NDArray[Any]:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response classifier is not fitted.")
        x_array = self._validate_x(x, name="x")
        return np.asarray(
            self.linear_model_.predict(
                x_array,
                response_features=self._design(x_array),
            )
        )

    def score(self, x: ArrayLike, y: ArrayLike) -> float:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response classifier is not fitted.")
        x_array = self._validate_x(x, name="x")
        return float(
            self.linear_model_.score(
                x_array,
                y,
                response_features=self._design(x_array),
            )
        )

    @property
    def engine_(self) -> str:
        if not hasattr(self, "linear_model_"):
            raise RuntimeError("The selected linear-response classifier is not fitted.")
        return str(self.linear_model_.engine_)


__all__ = ["SelectedLinearResponseClassifier", "SelectedLinearResponseRegressor"]
