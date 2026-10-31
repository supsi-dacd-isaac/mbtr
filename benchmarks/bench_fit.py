"""Fit-time and peak-memory scaling benchmarks."""

from __future__ import annotations

import pickle

from .common import (
    BASELINE,
    BIN_SIZES,
    DTYPES,
    FEATURE_SIZES,
    LINEAR_FEATURE_SIZES,
    OUTPUT_SIZES,
    ROW_SIZES,
    TREE_SIZES,
    make_model,
    make_regression,
    make_response_features,
)


class _FitBase:
    number = 1
    repeat = 3
    timeout = 600

    def _fit(self):
        return make_model(
            n_estimators=self.n_estimators,
            max_bins=self.max_bins,
            engine="numpy",
        ).fit(self.x, self.y)


class TimeFitRows(_FitBase):
    params = ROW_SIZES
    param_names = ["n_samples"]

    def setup(self, n_samples):
        self.n_estimators = 1
        self.max_bins = BASELINE.max_bins
        self.x, self.y = make_regression(n_samples, BASELINE.n_features, BASELINE.n_outputs)

    def time_fit_rows(self, n_samples):
        self._fit()

    def peakmem_fit_rows(self, n_samples):
        self._fit()


class TimeFitFeatures(_FitBase):
    params = FEATURE_SIZES
    param_names = ["n_features"]

    def setup(self, n_features):
        self.n_estimators = 1
        self.max_bins = BASELINE.max_bins
        self.x, self.y = make_regression(BASELINE.n_samples, n_features, BASELINE.n_outputs)

    def time_fit_features(self, n_features):
        self._fit()


class TimeFitOutputs(_FitBase):
    params = OUTPUT_SIZES
    param_names = ["n_outputs"]

    def setup(self, n_outputs):
        self.n_estimators = 1
        self.max_bins = BASELINE.max_bins
        self.x, self.y = make_regression(BASELINE.n_samples, BASELINE.n_features, n_outputs)

    def time_fit_outputs(self, n_outputs):
        self._fit()


class TimeFitBins(_FitBase):
    params = BIN_SIZES
    param_names = ["max_bins"]

    def setup(self, max_bins):
        self.n_estimators = 1
        self.max_bins = max_bins
        self.x, self.y = make_regression(
            BASELINE.n_samples, BASELINE.n_features, BASELINE.n_outputs
        )

    def time_fit_bins(self, max_bins):
        self._fit()


class TimeFitTrees(_FitBase):
    params = TREE_SIZES
    param_names = ["n_estimators"]

    def setup(self, n_estimators):
        self.n_estimators = n_estimators
        self.max_bins = BASELINE.max_bins
        self.x, self.y = make_regression(2_000, 16, 4)

    def time_fit_trees(self, n_estimators):
        self._fit()


class TimeFitDtype(_FitBase):
    params = DTYPES
    param_names = ["dtype"]

    def setup(self, dtype):
        self.n_estimators = 1
        self.max_bins = BASELINE.max_bins
        self.x, self.y = make_regression(
            BASELINE.n_samples, BASELINE.n_features, BASELINE.n_outputs, dtype=dtype
        )

    def time_fit_dtype(self, dtype):
        self._fit()


class TimeFitLinearResponse:
    params = LINEAR_FEATURE_SIZES
    param_names = ["n_response_features"]
    number = 1
    repeat = 3
    timeout = 600

    def setup(self, n_response_features):
        self.x, self.y = make_regression(2_000, 16, 4)
        self.response_features = make_response_features(self.x, n_response_features)

    def time_fit_linear_response(self, n_response_features):
        make_model(
            n_estimators=1,
            objective="linear_regression",
            engine="numpy",
        ).fit(self.x, self.y, response_features=self.response_features)


class TrackModelSize:
    params = TREE_SIZES
    param_names = ["n_estimators"]
    timeout = 600

    def setup(self, n_estimators):
        x, y = make_regression(2_000, 16, 4)
        self.model = make_model(n_estimators=n_estimators, engine="numpy").fit(x, y)

    def track_pickle_bytes(self, n_estimators):
        return len(pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL))

    track_pickle_bytes.unit = "bytes"
