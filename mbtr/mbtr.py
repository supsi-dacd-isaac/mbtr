"""Compatibility imports for the historical :mod:`mbtr.mbtr` module."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._tree import RegressionTree
from .estimator import MBTRegressor, _as_feature_matrix, _bin_features, _fit_bin_thresholds
from .losses import Loss, make_loss


class Tree:
    """Backward-compatible single-tree facade.

    New applications should use :class:`MBTRegressor`; direct tree access is
    intentionally a low-level API.
    """

    def __init__(
        self,
        n_q: int = 10,
        min_leaf: int = 100,
        loss_type: str | Loss = "mse",
        lambda_weights: float = 0.1,
        lambda_leaves: float = 0.1,
        max_depth: int | None = None,
        max_leaves: int | None = None,
        min_split_gain: float = 0.0,
        **loss_kwargs: Any,
    ) -> None:
        if n_q < 1:
            raise ValueError("n_q must be positive.")
        self.n_q = n_q
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_split_gain = min_split_gain
        self.loss = make_loss(
            loss_type,
            lambda_weights=lambda_weights,
            lambda_leaves=lambda_leaves,
            objective_params=loss_kwargs,
        )
        self._tree: RegressionTree | None = None

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        hessian: ArrayLike | None = None,
        learning_rate: float = 1.0,
        x_lr: ArrayLike | None = None,
    ) -> Tree:
        x_array = _as_feature_matrix(x, name="x")
        gradient = np.asarray(y, dtype=np.float64)
        if gradient.ndim == 1:
            gradient = gradient[:, None]
        if gradient.ndim != 2 or gradient.shape[0] != x_array.shape[0]:
            raise ValueError("y must have the same number of rows as x.")
        if not np.all(np.isfinite(gradient)):
            raise ValueError("y must contain only finite values.")
        curvature = (
            np.ones_like(gradient)
            if hessian is None
            else np.asarray(hessian, dtype=np.float64)
        )
        if curvature.shape != gradient.shape or np.any(curvature < 0):
            raise ValueError("hessian must match y and be non-negative.")
        self.loss.configure(gradient.shape[1])
        thresholds = _fit_bin_thresholds(x_array, self.n_q + 1)
        response_features = None if x_lr is None else _as_feature_matrix(x_lr, name="x_lr")
        self._tree = RegressionTree(
            objective=self.loss,
            min_samples_leaf=self.min_leaf,
            max_depth=self.max_depth,
            max_leaves=self.max_leaves,
            min_split_gain=self.min_split_gain,
            learning_rate=learning_rate,
            engine="numpy",
        ).fit(
            _bin_features(x_array, thresholds),
            thresholds,
            gradient,
            curvature,
            response_features=response_features,
        )
        return self

    def predict(self, x: ArrayLike, x_lr: ArrayLike | None = None) -> NDArray[np.float64]:
        if self._tree is None:
            raise RuntimeError("This Tree instance is not fitted.")
        return self._tree.predict(
            _as_feature_matrix(x, name="x"),
            response_features=None
            if x_lr is None
            else _as_feature_matrix(x_lr, name="x_lr"),
        )

    def _refit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        discount: float = 1.0,
        x_lr: ArrayLike | None = None,
    ) -> None:
        del x_lr
        if self._tree is None:
            raise RuntimeError("This Tree instance is not fitted.")
        previous_rate = self._tree.learning_rate
        self._tree.learning_rate = discount
        self._tree.refit_leaves(
            _as_feature_matrix(x, name="x"), np.asarray(y, dtype=np.float64)
        )
        self._tree.learning_rate = previous_rate

    def _find_regions(self, x: ArrayLike, y: ArrayLike | None = None) -> list[NDArray[np.bool_]]:
        del y
        if self._tree is None:
            raise RuntimeError("This Tree instance is not fitted.")
        leaves = self._tree.apply(_as_feature_matrix(x, name="x"))
        return [leaves == leaf for leaf in np.unique(leaves)]

    @property
    def g(self) -> Any:
        if self._tree is None:
            raise RuntimeError("This Tree instance is not fitted.")
        return self._tree.to_networkx()


class MBT(MBTRegressor):
    """Compatibility estimator using the original constructor names."""

    def __init__(
        self,
        n_boosts: int = 20,
        early_stopping_rounds: int | None = 3,
        learning_rate: float = 0.1,
        val_ratio: float = 0.0,
        n_q: int = 10,
        min_leaf: int = 100,
        loss_type: str | Loss = "mse",
        lambda_weights: float = 0.1,
        lambda_leaves: float = 0.1,
        verbose: int = 0,
        refit: bool = True,
        random_state: int | None = None,
        **loss_kwargs: Any,
    ) -> None:
        self.n_boosts = n_boosts
        self.val_ratio = val_ratio
        self.n_q = n_q
        self.min_leaf = min_leaf
        self.loss_type = loss_type
        self.lambda_weights = lambda_weights
        self.lambda_leaves = lambda_leaves
        self.refit = refit
        self.loss_kwargs = dict(loss_kwargs)
        super().__init__(
            objective=loss_type,
            objective_params=loss_kwargs,
            n_estimators=n_boosts,
            learning_rate=learning_rate,
            max_bins=max(2, n_q + 1),
            min_samples_leaf=min_leaf,
            l2_regularization=lambda_weights,
            leaf_regularization=lambda_leaves,
            early_stopping_rounds=early_stopping_rounds,
            validation_fraction=val_ratio,
            random_state=random_state,
            refit_leaves=refit,
            verbose=verbose,
        )

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        do_plot: bool = False,
        x_lr: ArrayLike | None = None,
        **kwargs: Any,
    ) -> MBT:
        if do_plot:
            warnings.warn(
                "Training plots were removed from the estimator; plot evals_result_ externally.",
                DeprecationWarning,
                stacklevel=2,
            )
        response_features = kwargs.pop("response_features", x_lr)
        super().fit(x, y, response_features=response_features, **kwargs)
        return self

    def predict(
        self,
        x: ArrayLike,
        n: int | None = None,
        x_lr: ArrayLike | None = None,
        **kwargs: Any,
    ) -> NDArray[np.float64]:
        response_features = kwargs.pop("response_features", x_lr)
        iteration_range = None if n is None else (0, n)
        return super().predict(
            x,
            response_features=response_features,
            iteration_range=iteration_range,
            **kwargs,
        )

    @property
    def trees(self) -> list[RegressionTree]:
        return getattr(self, "trees_", [])

    @property
    def y_0(self) -> NDArray[np.float64] | None:
        return getattr(self, "base_prediction_", None)


def bin_sums(y: ArrayLike, edges: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
    """Cumulative sums in half-open bins, retained for source compatibility."""

    values = np.asarray(y, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.float64)
    feature = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or feature.ndim != 1 or len(feature) != len(values):
        raise ValueError("y must be 2-D and x a matching 1-D array.")
    result = np.zeros((len(edge_array) - 1, values.shape[1]), dtype=np.float64)
    running = np.zeros(values.shape[1], dtype=np.float64)
    for index in range(len(edge_array) - 1):
        mask = (edge_array[index] <= feature) & (feature < edge_array[index + 1])
        running = running + np.sum(values[mask], axis=0)
        result[index] = running
    return result


def leaf_stats(
    y: ArrayLike, edges: ArrayLike, x: ArrayLike, order: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    cumulative = bin_sums(y, edges, x)
    if order == 1:
        return np.square(cumulative), np.square(cumulative[-1] - cumulative)
    if order == 2:
        return cumulative, cumulative[-1] - cumulative
    raise ValueError("order must be 1 or 2.")


__all__ = ["MBT", "MBTRegressor", "Tree", "bin_sums", "leaf_stats"]
