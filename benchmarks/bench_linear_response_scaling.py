"""Fixed-topology scaling benchmarks for ridge-linear tree leaves.

The existing end-to-end benchmark intentionally follows representative model
defaults.  These cases instead cap every tree at the same depth and vary one
matrix dimension at a time, so timing changes can be attributed to that axis
rather than to a different number of fitted nodes.  Regression and Bernoulli
logit-response leaves share the same upper-bound Gram histogram; small exact
curvature cases separately expose the additional per-label cost.
"""

from __future__ import annotations

import numpy as np

from .common import FULL_PROFILE

ROW_SIZES = [1_000, 10_000, 50_000] if FULL_PROFILE else [1_000, 10_000]
SPLIT_FEATURE_SIZES = [8, 32, 64, 128] if FULL_PROFILE else [8, 32, 64]
OUTPUT_SIZES = [1, 8, 32, 128] if FULL_PROFILE else [1, 8, 32]
BIN_SIZES = [16, 64, 256] if FULL_PROFILE else [16, 64]
RESPONSE_FEATURE_SIZES = [2, 4, 8, 16, 32] if FULL_PROFILE else [2, 8, 16]
TREE_SIZES = [1, 5, 20] if FULL_PROFILE else [1, 5]
LOGISTIC_CURVATURE_OUTPUT_SIZES = [1, 4, 8] if FULL_PROFILE else [1, 4]
LOGISTIC_CURVATURE_MODES = ["upper_bound", "exact"]

BASE_ROWS = 3_000
BASE_SPLIT_FEATURES = 32
BASE_OUTPUTS = 8
BASE_BINS = 64
BASE_RESPONSE_FEATURES = 4
FIXED_DEPTH = 3


def _make_problem(
    n_samples: int,
    n_features: int,
    n_outputs: int,
    n_response_features: int,
    *,
    seed: int = 1729,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic data with signal that still benefits from splits."""

    if n_response_features > n_features:
        raise ValueError("n_response_features must not exceed n_features.")

    generator = np.random.default_rng(seed)
    x = generator.normal(size=(n_samples, n_features))
    output_scale = np.linspace(0.6, 1.4, n_outputs)
    nonlinear = np.sin(2.5 * x[:, 0]) + 0.7 * (x[:, 1] > 0.2) + 0.3 * x[:, 2] * x[:, 3]
    linear_column = x[:, min(4, n_features - 1)]
    y = nonlinear[:, None] * output_scale[None, :]
    y += 0.15 * linear_column[:, None] * np.linspace(-1.0, 1.0, n_outputs)[None, :]
    y += generator.normal(scale=0.03, size=y.shape)
    response_features = np.ascontiguousarray(x[:, :n_response_features])
    return np.ascontiguousarray(x), np.ascontiguousarray(y), response_features


def _make_classification_problem(
    n_samples: int,
    n_features: int,
    n_outputs: int,
    n_response_features: int,
    *,
    seed: int = 2718,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a deterministic multilabel problem with split and logit signal."""

    if n_response_features > n_features:
        raise ValueError("n_response_features must not exceed n_features.")

    generator = np.random.default_rng(seed)
    x = generator.normal(size=(n_samples, n_features))
    response_features = np.ascontiguousarray(x[:, :n_response_features])

    coefficients = generator.normal(
        scale=0.45 / np.sqrt(n_response_features),
        size=(n_response_features, n_outputs),
    )
    output_scale = np.linspace(0.7, 1.3, n_outputs)
    routed_signal = 0.8 * np.sin(1.8 * x[:, 0]) + 0.55 * (x[:, 1] > 0.15) - 0.35 * (x[:, 2] * x[:, 3] > 0.0)
    logits = response_features @ coefficients
    logits += routed_signal[:, None] * output_scale[None, :]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = generator.binomial(1, probabilities).astype(np.float64)
    return np.ascontiguousarray(x), np.ascontiguousarray(y), response_features


def _warm_numba_linear_histogram() -> None:
    """Compile the shape-generic linear histogram kernel outside timings."""

    from mbtr._numba import histogram_linear_statistics

    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    design = np.ones((4, 3), dtype=np.float64)
    gradient = np.ones((4, 2), dtype=np.float64)
    histogram_linear_statistics(bins, design, gradient, 2)


def _warm_numba_linear_logistic_histograms() -> None:
    """Compile both Bernoulli response-leaf kernels outside timings."""

    from mbtr._numba import histogram_linear_logistic_statistics

    _warm_numba_linear_histogram()
    bins = np.array([0, 1, 0, 1], dtype=np.int64)
    design = np.ones((4, 3), dtype=np.float64)
    gradient = np.ones((4, 2), dtype=np.float64)
    hessian = np.full((4, 2), 0.25, dtype=np.float64)
    histogram_linear_logistic_statistics(bins, design, gradient, hessian, 2)


class _LinearResponseScalingBase:
    number = 1
    repeat = 3
    timeout = 900

    n_estimators = 1
    max_bins = BASE_BINS

    def _set_problem(
        self,
        *,
        n_samples: int = BASE_ROWS,
        n_features: int = BASE_SPLIT_FEATURES,
        n_outputs: int = BASE_OUTPUTS,
        n_response_features: int = BASE_RESPONSE_FEATURES,
    ) -> None:
        self.x, self.y, self.response_features = _make_problem(
            n_samples,
            n_features,
            n_outputs,
            n_response_features,
        )
        _warm_numba_linear_histogram()

    def _fit(self, engine: str) -> None:
        from mbtr import MBTRegressor

        MBTRegressor(
            objective="linear_regression",
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_bins=self.max_bins,
            min_samples_leaf=20,
            max_depth=FIXED_DEPTH,
            early_stopping_rounds=None,
            engine=engine,
            random_state=0,
            verbose=0,
        ).fit(self.x, self.y, response_features=self.response_features)


class TimeFitLinearRows(_LinearResponseScalingBase):
    params = ROW_SIZES
    param_names = ["n_samples"]

    def setup(self, n_samples):
        self._set_problem(n_samples=n_samples)

    def time_fit_numpy(self, n_samples):
        self._fit("numpy")

    def time_fit_numba(self, n_samples):
        self._fit("numba")


class TimeFitLinearSplitFeatures(_LinearResponseScalingBase):
    params = SPLIT_FEATURE_SIZES
    param_names = ["n_features"]

    def setup(self, n_features):
        self._set_problem(n_features=n_features)

    def time_fit_numpy(self, n_features):
        self._fit("numpy")

    def time_fit_numba(self, n_features):
        self._fit("numba")


class TimeFitLinearOutputs(_LinearResponseScalingBase):
    params = OUTPUT_SIZES
    param_names = ["n_outputs"]

    def setup(self, n_outputs):
        self._set_problem(n_outputs=n_outputs)

    def time_fit_numpy(self, n_outputs):
        self._fit("numpy")

    def time_fit_numba(self, n_outputs):
        self._fit("numba")


class TimeFitLinearBins(_LinearResponseScalingBase):
    params = BIN_SIZES
    param_names = ["max_bins"]

    def setup(self, max_bins):
        self.max_bins = max_bins
        self._set_problem()

    def time_fit_numpy(self, max_bins):
        self._fit("numpy")

    def time_fit_numba(self, max_bins):
        self._fit("numba")


class TimeFitLinearResponseFeatures(_LinearResponseScalingBase):
    params = RESPONSE_FEATURE_SIZES
    param_names = ["n_response_features"]

    def setup(self, n_response_features):
        self._set_problem(
            n_features=max(BASE_SPLIT_FEATURES, n_response_features),
            n_response_features=n_response_features,
        )

    def time_fit_numpy(self, n_response_features):
        self._fit("numpy")

    def time_fit_numba(self, n_response_features):
        self._fit("numba")

    def peakmem_fit_numba(self, n_response_features):
        self._fit("numba")


class TimeFitLinearTrees(_LinearResponseScalingBase):
    params = TREE_SIZES
    param_names = ["n_estimators"]

    def setup(self, n_estimators):
        self.n_estimators = n_estimators
        self._set_problem(n_samples=2_000, n_features=16, n_outputs=4)

    def time_fit_numpy(self, n_estimators):
        self._fit("numpy")

    def time_fit_numba(self, n_estimators):
        self._fit("numba")


class _LinearLogisticResponseScalingBase:
    number = 1
    repeat = 3
    timeout = 900

    n_estimators = 1
    max_bins = BASE_BINS
    curvature = "upper_bound"

    def _set_problem(
        self,
        *,
        n_samples: int = BASE_ROWS,
        n_features: int = BASE_SPLIT_FEATURES,
        n_outputs: int = BASE_OUTPUTS,
        n_response_features: int = BASE_RESPONSE_FEATURES,
    ) -> None:
        self.x, self.y, self.response_features = _make_classification_problem(
            n_samples,
            n_features,
            n_outputs,
            n_response_features,
        )
        _warm_numba_linear_logistic_histograms()

    def _fit(self, engine: str) -> None:
        from mbtr import MBTClassifier

        MBTClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_bins=self.max_bins,
            min_samples_leaf=20,
            max_depth=FIXED_DEPTH,
            early_stopping_rounds=None,
            engine=engine,
            linear_response_curvature=self.curvature,
            random_state=0,
            verbose=0,
        ).fit(self.x, self.y, response_features=self.response_features)


class TimeFitLinearLogisticRows(_LinearLogisticResponseScalingBase):
    params = ROW_SIZES
    param_names = ["n_samples"]

    def setup(self, n_samples):
        self._set_problem(n_samples=n_samples)

    def time_fit_numpy(self, n_samples):
        self._fit("numpy")

    def time_fit_numba(self, n_samples):
        self._fit("numba")


class TimeFitLinearLogisticSplitFeatures(_LinearLogisticResponseScalingBase):
    params = SPLIT_FEATURE_SIZES
    param_names = ["n_features"]

    def setup(self, n_features):
        self._set_problem(n_features=n_features)

    def time_fit_numpy(self, n_features):
        self._fit("numpy")

    def time_fit_numba(self, n_features):
        self._fit("numba")


class TimeFitLinearLogisticOutputs(_LinearLogisticResponseScalingBase):
    params = OUTPUT_SIZES
    param_names = ["n_outputs"]

    def setup(self, n_outputs):
        self._set_problem(n_outputs=n_outputs)

    def time_fit_numpy(self, n_outputs):
        self._fit("numpy")

    def time_fit_numba(self, n_outputs):
        self._fit("numba")


class TimeFitLinearLogisticBins(_LinearLogisticResponseScalingBase):
    params = BIN_SIZES
    param_names = ["max_bins"]

    def setup(self, max_bins):
        self.max_bins = max_bins
        self._set_problem()

    def time_fit_numpy(self, max_bins):
        self._fit("numpy")

    def time_fit_numba(self, max_bins):
        self._fit("numba")


class TimeFitLinearLogisticResponseFeatures(_LinearLogisticResponseScalingBase):
    params = RESPONSE_FEATURE_SIZES
    param_names = ["n_response_features"]

    def setup(self, n_response_features):
        self._set_problem(
            n_features=max(BASE_SPLIT_FEATURES, n_response_features),
            n_response_features=n_response_features,
        )

    def time_fit_numpy(self, n_response_features):
        self._fit("numpy")

    def time_fit_numba(self, n_response_features):
        self._fit("numba")

    def peakmem_fit_numba(self, n_response_features):
        self._fit("numba")


class TimeFitLinearLogisticTrees(_LinearLogisticResponseScalingBase):
    params = TREE_SIZES
    param_names = ["n_estimators"]

    def setup(self, n_estimators):
        self.n_estimators = n_estimators
        self._set_problem(n_samples=2_000, n_features=16, n_outputs=4)

    def time_fit_numpy(self, n_estimators):
        self._fit("numpy")

    def time_fit_numba(self, n_estimators):
        self._fit("numba")


class TimeFitLinearLogisticCurvature(_LinearLogisticResponseScalingBase):
    """Compare shared upper-bound and exact per-label curvature at small sizes."""

    params = [LOGISTIC_CURVATURE_MODES, LOGISTIC_CURVATURE_OUTPUT_SIZES]
    param_names = ["curvature", "n_outputs"]

    def setup(self, curvature, n_outputs):
        self.curvature = curvature
        self.max_bins = 32
        self._set_problem(
            n_samples=1_000,
            n_features=16,
            n_outputs=n_outputs,
            n_response_features=4,
        )

    def time_fit_numpy(self, curvature, n_outputs):
        self._fit("numpy")

    def time_fit_numba(self, curvature, n_outputs):
        self._fit("numba")

    def peakmem_fit_numba(self, curvature, n_outputs):
        self._fit("numba")
