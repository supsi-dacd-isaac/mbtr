"""Shared deterministic fixtures and benchmark parameter profiles."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

FULL_PROFILE = os.environ.get("MBTR_BENCHMARK_PROFILE", "quick").lower() == "full"

ROW_SIZES = [1_000, 10_000, 100_000, 1_000_000] if FULL_PROFILE else [1_000, 10_000]
FEATURE_SIZES = [4, 16, 64, 256] if FULL_PROFILE else [4, 16, 64]
OUTPUT_SIZES = [1, 8, 32, 128] if FULL_PROFILE else [1, 8, 32]
BIN_SIZES = [16, 64, 256] if FULL_PROFILE else [16, 64]
LINEAR_FEATURE_SIZES = [2, 8, 32] if FULL_PROFILE else [2, 8]
TREE_SIZES = [1, 10, 50] if FULL_PROFILE else [1, 5]
PREDICTION_BATCH_SIZES = [1, 1_000, 100_000] if FULL_PROFILE else [1, 1_000]
DTYPES = ["float32", "float64"]


@dataclass(frozen=True)
class Baseline:
    n_samples: int = 10_000
    n_features: int = 32
    n_outputs: int = 8
    max_bins: int = 64
    n_linear_features: int = 8
    n_estimators: int = 10
    min_samples_leaf: int = 20


BASELINE = Baseline()


def make_regression(
    n_samples: int,
    n_features: int,
    n_outputs: int,
    *,
    dtype: str = "float64",
    seed: int = 1729,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic data containing linear and non-linear split signal."""

    rng = np.random.default_rng(seed)
    np_dtype = np.dtype(dtype)
    x = rng.normal(size=(n_samples, n_features)).astype(np_dtype, copy=False)

    informative = min(n_features, 12)
    coefficients = rng.normal(size=(informative, n_outputs)).astype(np_dtype, copy=False)
    y = x[:, :informative] @ coefficients
    y += np.sin(x[:, [0]]) * np.linspace(0.2, 1.0, n_outputs, dtype=np_dtype)
    if n_features > 1:
        y += (x[:, [1]] > 0).astype(np_dtype) * np.linspace(
            -0.4, 0.4, n_outputs, dtype=np_dtype
        )
    noise = rng.normal(scale=0.02, size=y.shape).astype(np_dtype, copy=False)
    y = (y + noise).astype(np_dtype, copy=False)
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def make_response_features(
    x: np.ndarray,
    n_linear_features: int,
    *,
    seed: int = 2718,
) -> np.ndarray:
    """Create response-model features independently of tree split dimensions."""

    rng = np.random.default_rng(seed)
    projection = rng.normal(size=(x.shape[1], n_linear_features)).astype(x.dtype, copy=False)
    response_features = x @ projection / np.sqrt(x.shape[1])
    return np.ascontiguousarray(response_features, dtype=x.dtype)


def make_model(
    *,
    n_estimators: int = BASELINE.n_estimators,
    max_bins: int = BASELINE.max_bins,
    objective: str = "squared_error",
    engine: str = "numpy",
):
    """Construct an estimator without importing MBTR during benchmark discovery."""

    from mbtr import MBTRegressor

    return MBTRegressor(
        objective=objective,
        n_estimators=n_estimators,
        learning_rate=0.05,
        max_bins=max_bins,
        min_samples_leaf=BASELINE.min_samples_leaf,
        early_stopping_rounds=None,
        engine=engine,
        random_state=0,
        verbose=0,
    )
