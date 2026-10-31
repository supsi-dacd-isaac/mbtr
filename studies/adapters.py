"""Shared-tree and independent-output estimators for controlled comparisons."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mbtr import MBTClassifier, MBTRegressor


class IndependentOutputsRegressor:
    """Fit one fresh scalar regressor per response column."""

    def __init__(self, estimator_factory: Callable[[], Any]) -> None:
        self.estimator_factory = estimator_factory

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> IndependentOutputsRegressor:
        target = np.asarray(y, dtype=np.float64)
        self.target_was_one_dimensional_ = target.ndim == 1
        if self.target_was_one_dimensional_:
            target = target[:, None]
        if target.ndim != 2 or not target.size:
            raise ValueError("y must be a non-empty one- or two-dimensional array.")
        validation_x: ArrayLike | None = None
        validation_y: NDArray[np.float64] | None = None
        if eval_set is not None:
            validation_x = eval_set[0]
            validation_y = np.asarray(eval_set[1], dtype=np.float64)
            if validation_y.ndim == 1:
                validation_y = validation_y[:, None]
            if validation_y.ndim != 2 or validation_y.shape[1] != target.shape[1]:
                raise ValueError("Validation targets must match the training output dimension.")
        self.estimators_ = []
        for output in range(target.shape[1]):
            estimator = self.estimator_factory()
            fit_kwargs = {}
            if validation_y is not None:
                fit_kwargs["eval_set"] = (validation_x, validation_y[:, output])
            estimator.fit(x, target[:, output], **fit_kwargs)
            self.estimators_.append(estimator)
        self.n_outputs_ = target.shape[1]
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "estimators_"):
            raise ValueError("The independent-output regressor is not fitted.")
        prediction = np.column_stack([np.asarray(estimator.predict(x)) for estimator in self.estimators_])
        return prediction[:, 0] if self.target_was_one_dimensional_ else prediction


class IndependentLabelsClassifier:
    """Fit one fresh binary classifier per label column."""

    def __init__(self, estimator_factory: Callable[[], Any]) -> None:
        self.estimator_factory = estimator_factory

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        *,
        eval_set: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> IndependentLabelsClassifier:
        target = np.asarray(y)
        self.target_was_one_dimensional_ = target.ndim == 1
        if self.target_was_one_dimensional_:
            target = target[:, None]
        if target.ndim != 2 or not target.size or np.any((target != 0) & (target != 1)):
            raise ValueError("y must be a non-empty binary indicator array.")
        validation_x: ArrayLike | None = None
        validation_y: NDArray[Any] | None = None
        if eval_set is not None:
            validation_x = eval_set[0]
            validation_y = np.asarray(eval_set[1])
            if validation_y.ndim == 1:
                validation_y = validation_y[:, None]
            if validation_y.ndim != 2 or validation_y.shape[1] != target.shape[1]:
                raise ValueError("Validation targets must match the training label dimension.")
        self.estimators_ = []
        for label in range(target.shape[1]):
            estimator = self.estimator_factory()
            fit_kwargs = {}
            if validation_y is not None:
                fit_kwargs["eval_set"] = (validation_x, validation_y[:, label])
            estimator.fit(x, target[:, label], **fit_kwargs)
            self.estimators_.append(estimator)
        self.n_outputs_ = target.shape[1]
        return self

    @staticmethod
    def _positive_probability(estimator: Any, x: ArrayLike) -> NDArray[np.float64]:
        probability = np.asarray(estimator.predict_proba(x), dtype=np.float64)
        if probability.ndim == 2 and probability.shape[1] == 2:
            return probability[:, 1]
        if probability.ndim == 1:
            return probability
        raise ValueError("A scalar binary estimator must return one or two probability columns.")

    def predict_proba(self, x: ArrayLike) -> NDArray[np.float64]:
        if not hasattr(self, "estimators_"):
            raise ValueError("The independent-label classifier is not fitted.")
        probability = np.column_stack(
            [self._positive_probability(estimator, x) for estimator in self.estimators_]
        )
        return probability[:, 0] if self.target_was_one_dimensional_ else probability

    def predict(self, x: ArrayLike, *, threshold: float | None = None) -> NDArray[np.int64]:
        probability = self.predict_proba(x)
        if threshold is not None:
            if not 0.0 < threshold < 1.0:
                raise ValueError("threshold must be strictly between zero and one.")
            return np.asarray(probability >= threshold, dtype=np.int64)
        fitted_thresholds = np.asarray(
            [float(getattr(estimator, "decision_threshold", 0.5)) for estimator in self.estimators_]
        )
        if probability.ndim == 1:
            return np.asarray(probability >= fitted_thresholds[0], dtype=np.int64)
        return np.asarray(probability >= fitted_thresholds[None, :], dtype=np.int64)


def make_estimator(
    task: str,
    variant: str,
    parameters: Mapping[str, Any] | None = None,
) -> MBTRegressor | MBTClassifier | IndependentOutputsRegressor | IndependentLabelsClassifier:
    """Construct a shared or independent MBTR estimator for a study run."""

    settings = dict(parameters or {})
    if task == "regression":
        if variant == "shared":
            return MBTRegressor(**settings)
        if variant == "independent":
            return IndependentOutputsRegressor(partial(MBTRegressor, **settings))
    elif task in {"binary", "multilabel"}:
        if variant == "shared":
            return MBTClassifier(**settings)
        if variant == "independent":
            return IndependentLabelsClassifier(partial(MBTClassifier, **settings))
    else:
        raise ValueError("task must be 'regression', 'binary', or 'multilabel'.")
    raise ValueError("variant must be 'shared' or 'independent'.")


__all__ = ["IndependentLabelsClassifier", "IndependentOutputsRegressor", "make_estimator"]
