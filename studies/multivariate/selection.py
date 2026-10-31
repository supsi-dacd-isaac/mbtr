"""Leakage-safe response-feature selection and linear-cost preflight checks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SelectionPolicy:
    """Pre-registered rule for selecting leaf-level response features."""

    importance_coverage: float = 0.8
    max_features: int = 8
    min_features: int = 1

    def validate(self) -> None:
        if not 0.0 < self.importance_coverage <= 1.0:
            raise ValueError("importance_coverage must be in (0, 1].")
        if self.max_features < 1:
            raise ValueError("max_features must be positive.")
        if self.min_features < 1 or self.min_features > self.max_features:
            raise ValueError("min_features must be between one and max_features.")


@dataclass(frozen=True)
class LinearCostEstimate:
    """Conservative implementation-specific work and temporary-memory estimate."""

    response_features: int
    design_width: int
    leaf_kind: str
    curvature_mode: str
    depth: int | None
    internal_nodes: int | None
    work_units: float | None
    predicted_seconds: float | None
    temporary_bytes: int
    status: str
    reason: str | None


@dataclass(frozen=True)
class ResponseFeatureSelection:
    """Stable feature ranking plus the subset admitted by the cost guard."""

    ranked_indices: tuple[int, ...]
    ranked_importances: tuple[float, ...]
    importance_selected_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    selected_importance: float
    cost: LinearCostEstimate
    status: str
    reason: str | None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cost"] = asdict(self.cost)
        return value


def _validated_importances(value: ArrayLike) -> FloatArray:
    importance = np.asarray(value, dtype=np.float64)
    if importance.ndim != 1 or not len(importance):
        raise ValueError("feature_importances must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(importance)) or np.any(importance < 0.0):
        raise ValueError("feature_importances must contain finite non-negative values.")
    return importance


def rank_feature_importances(value: ArrayLike) -> tuple[IntArray, FloatArray]:
    """Rank positive importances descending, breaking ties by column index."""

    importance = _validated_importances(value)
    indices = np.arange(len(importance), dtype=np.int64)
    order = np.lexsort((indices, -importance))
    positive = order[importance[order] > 0.0]
    return positive, importance[positive]


def importance_subset(value: ArrayLike, policy: SelectionPolicy) -> tuple[IntArray, FloatArray, float]:
    """Select the smallest stable top-k attaining the requested gain coverage."""

    policy.validate()
    ranked_indices, ranked_importance = rank_feature_importances(value)
    if not len(ranked_indices):
        return ranked_indices, ranked_importance, 0.0
    normalized = ranked_importance / np.sum(ranked_importance)
    required = int(np.searchsorted(np.cumsum(normalized), policy.importance_coverage, side="left") + 1)
    count = min(policy.max_features, max(policy.min_features, required))
    selected = ranked_indices[:count]
    selected_importance = float(np.sum(normalized[:count]))
    return selected, ranked_importance, selected_importance


def estimate_linear_response_cost(
    *,
    n_samples: int,
    n_split_features: int,
    n_outputs: int,
    n_response_features: int,
    n_estimators: int,
    max_bins: int,
    max_depth: int | None,
    leaf_kind: str = "regression",
    curvature_mode: str = "upper_bound",
) -> LinearCostEstimate:
    """Estimate linear-histogram work for this implementation.

    The estimate counts row-level Gram/gradient accumulation plus ridge solves
    at every candidate split.  It is a guard and comparison metadata, not a
    wall-clock prediction.
    """

    dimensions = {
        "n_samples": n_samples,
        "n_split_features": n_split_features,
        "n_outputs": n_outputs,
        "n_response_features": n_response_features,
        "n_estimators": n_estimators,
        "max_bins": max_bins,
    }
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in dimensions.values()):
        raise ValueError(f"Linear cost dimensions must be positive integers: {dimensions}.")
    if leaf_kind not in {"regression", "classification"}:
        raise ValueError("leaf_kind must be 'regression' or 'classification'.")
    if curvature_mode not in {"upper_bound", "exact"}:
        raise ValueError("curvature_mode must be 'upper_bound' or 'exact'.")
    exact_logistic = leaf_kind == "classification" and curvature_mode == "exact"
    design_width = n_response_features + 1
    if exact_logistic:
        temporary_bytes = int(
            16 * max_bins * n_outputs * design_width * (design_width + 1) + 8 * n_samples * design_width
        )
    else:
        temporary_bytes = int(16 * max_bins * design_width * (design_width + n_outputs) + 8 * n_samples * design_width)
    if max_depth is None:
        return LinearCostEstimate(
            response_features=n_response_features,
            design_width=design_width,
            leaf_kind=leaf_kind,
            curvature_mode=curvature_mode,
            depth=None,
            internal_nodes=None,
            work_units=None,
            predicted_seconds=None,
            temporary_bytes=temporary_bytes,
            status="skip_unbounded_depth",
            reason="Linear-response preflight requires a finite max_depth unless explicitly overridden.",
        )
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise ValueError("max_depth must be None or a positive integer.")
    internal_nodes = 2**max_depth - 1
    if exact_logistic:
        row_histogram_work = n_samples * n_split_features * max_depth * n_outputs * design_width * (design_width + 1)
        candidate_solve_work = (
            internal_nodes * n_split_features * (max_bins - 1) * n_outputs * design_width**2 * (design_width + 1)
        )
        candidate_seconds = (
            internal_nodes
            * n_split_features
            * (max_bins - 1)
            * n_outputs
            * (15e-6 + 1e-9 * design_width**2 * (design_width + 1))
        )
    else:
        row_histogram_work = n_samples * n_split_features * max_depth * design_width * (design_width + n_outputs)
        candidate_solve_work = (
            internal_nodes * n_split_features * (max_bins - 1) * design_width**2 * (design_width + n_outputs)
        )
        candidate_seconds = (
            internal_nodes
            * n_split_features
            * (max_bins - 1)
            * (15e-6 + 1e-9 * design_width**2 * (design_width + n_outputs))
        )
    work_units = float(n_estimators * (row_histogram_work + candidate_solve_work))
    predicted_seconds = float(n_estimators * (5.5e-10 * row_histogram_work + candidate_seconds))
    return LinearCostEstimate(
        response_features=n_response_features,
        design_width=design_width,
        leaf_kind=leaf_kind,
        curvature_mode=curvature_mode,
        depth=max_depth,
        internal_nodes=internal_nodes,
        work_units=work_units,
        predicted_seconds=predicted_seconds,
        temporary_bytes=temporary_bytes,
        status="estimated",
        reason=None,
    )


def select_response_features(
    feature_importances: ArrayLike,
    *,
    policy: SelectionPolicy,
    n_samples: int,
    n_outputs: int,
    n_estimators: int,
    max_bins: int,
    max_depth: int | None,
    work_budget: float = 2.0e11,
    time_budget_seconds: float = 300.0,
    memory_budget_bytes: int = 1024**3,
    force: bool = False,
    leaf_kind: str = "regression",
    curvature_mode: str = "upper_bound",
) -> ResponseFeatureSelection:
    """Select relevant columns and shrink top-k until the linear fit is admissible."""

    importance = _validated_importances(feature_importances)
    policy.validate()
    if not math.isfinite(work_budget) or work_budget <= 0.0:
        raise ValueError("work_budget must be finite and positive.")
    if not math.isfinite(time_budget_seconds) or time_budget_seconds <= 0.0:
        raise ValueError("time_budget_seconds must be finite and positive.")
    if memory_budget_bytes < 1:
        raise ValueError("memory_budget_bytes must be positive.")
    ranked_indices, ranked_values = rank_feature_importances(importance)
    selected, _, _ = importance_subset(importance, policy)
    if not len(selected):
        cost = estimate_linear_response_cost(
            n_samples=n_samples,
            n_split_features=len(importance),
            n_outputs=n_outputs,
            n_response_features=1,
            n_estimators=n_estimators,
            max_bins=max_bins,
            max_depth=max_depth,
            leaf_kind=leaf_kind,
            curvature_mode=curvature_mode,
        )
        return ResponseFeatureSelection(
            ranked_indices=(),
            ranked_importances=(),
            importance_selected_indices=(),
            selected_indices=(),
            selected_importance=0.0,
            cost=cost,
            status="skip_no_importance",
            reason="The standard MBTR produced no positive split-gain importances.",
        )

    importance_selected = tuple(int(index) for index in selected)
    total_positive = float(np.sum(ranked_values))
    candidate = list(importance_selected)
    if len(candidate) < policy.min_features:
        cost = estimate_linear_response_cost(
            n_samples=n_samples,
            n_split_features=len(importance),
            n_outputs=n_outputs,
            n_response_features=len(candidate),
            n_estimators=n_estimators,
            max_bins=max_bins,
            max_depth=max_depth,
            leaf_kind=leaf_kind,
            curvature_mode=curvature_mode,
        )
        return ResponseFeatureSelection(
            ranked_indices=tuple(int(index) for index in ranked_indices),
            ranked_importances=tuple(float(value) for value in ranked_values),
            importance_selected_indices=importance_selected,
            selected_indices=(),
            selected_importance=0.0,
            cost=cost,
            status="skip_insufficient_importance",
            reason=(
                f"Only {len(candidate)} positive-gain feature(s) are available, below the "
                f"configured minimum of {policy.min_features}."
            ),
        )
    last_cost: LinearCostEstimate | None = None
    while len(candidate) >= policy.min_features:
        last_cost = estimate_linear_response_cost(
            n_samples=n_samples,
            n_split_features=len(importance),
            n_outputs=n_outputs,
            n_response_features=len(candidate),
            n_estimators=n_estimators,
            max_bins=max_bins,
            max_depth=max_depth,
            leaf_kind=leaf_kind,
            curvature_mode=curvature_mode,
        )
        work_allowed = last_cost.work_units is not None and last_cost.work_units <= work_budget
        time_allowed = last_cost.predicted_seconds is not None and last_cost.predicted_seconds <= time_budget_seconds
        memory_allowed = last_cost.temporary_bytes <= memory_budget_bytes
        bounded = last_cost.status != "skip_unbounded_depth"
        if force or (bounded and work_allowed and time_allowed and memory_allowed):
            selected_importance = float(np.sum(importance[candidate]) / total_positive)
            admitted_cost = LinearCostEstimate(
                **{
                    **asdict(last_cost),
                    "status": "forced"
                    if force and not (bounded and work_allowed and time_allowed and memory_allowed)
                    else "run",
                    "reason": None
                    if bounded and work_allowed and time_allowed and memory_allowed
                    else "Linear response was forced past the preflight guard.",
                }
            )
            return ResponseFeatureSelection(
                ranked_indices=tuple(int(index) for index in ranked_indices),
                ranked_importances=tuple(float(value) for value in ranked_values),
                importance_selected_indices=importance_selected,
                selected_indices=tuple(candidate),
                selected_importance=selected_importance,
                cost=admitted_cost,
                status="run",
                reason=admitted_cost.reason,
            )
        candidate.pop()

    assert last_cost is not None
    if last_cost.status == "skip_unbounded_depth":
        status = "skip_unbounded_depth"
        reason = last_cost.reason
    elif last_cost.temporary_bytes > memory_budget_bytes:
        status = "skip_memory"
        reason = f"Estimated temporary memory {last_cost.temporary_bytes} exceeds budget {memory_budget_bytes}."
    elif last_cost.predicted_seconds is not None and last_cost.predicted_seconds > time_budget_seconds:
        status = "skip_time"
        reason = f"Predicted fit time {last_cost.predicted_seconds:.3g}s exceeds budget {time_budget_seconds:.3g}s."
    else:
        status = "skip_work"
        reason = f"Estimated work {last_cost.work_units} exceeds budget {work_budget}."
    return ResponseFeatureSelection(
        ranked_indices=tuple(int(index) for index in ranked_indices),
        ranked_importances=tuple(float(value) for value in ranked_values),
        importance_selected_indices=importance_selected,
        selected_indices=(),
        selected_importance=0.0,
        cost=last_cost,
        status=status,
        reason=reason,
    )


def fit_response_design(
    x_train: ArrayLike,
    partitions: dict[str, ArrayLike],
    selected_indices: tuple[int, ...],
) -> tuple[dict[str, FloatArray], FloatArray, FloatArray]:
    """Fit train-only standardization and transform selected columns."""

    training = np.asarray(x_train, dtype=np.float64)
    if training.ndim != 2 or not len(selected_indices):
        raise ValueError("x_train must be two-dimensional and selected_indices must not be empty.")
    indices = np.asarray(selected_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= training.shape[1]) or len(np.unique(indices)) != len(indices):
        raise ValueError("selected_indices must be unique valid feature positions.")
    selected_training = training[:, indices]
    offset = np.mean(selected_training, axis=0)
    scale = np.std(selected_training, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    transformed: dict[str, FloatArray] = {"train": np.ascontiguousarray((selected_training - offset) / scale)}
    for name, value in partitions.items():
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != training.shape[1]:
            raise ValueError(f"Partition {name!r} must match the training feature dimension.")
        transformed[name] = np.ascontiguousarray((matrix[:, indices] - offset) / scale)
    return transformed, np.asarray(offset, dtype=np.float64), np.asarray(scale, dtype=np.float64)


__all__ = [
    "LinearCostEstimate",
    "ResponseFeatureSelection",
    "SelectionPolicy",
    "estimate_linear_response_cost",
    "fit_response_design",
    "importance_subset",
    "rank_feature_importances",
    "select_response_features",
]
