"""Prediction, leaf-routing, and response-feature scaling benchmarks."""

from __future__ import annotations

from .common import (
    BASELINE,
    PREDICTION_BATCH_SIZES,
    make_model,
    make_regression,
    make_response_features,
)


class TimePredictBatch:
    params = PREDICTION_BATCH_SIZES
    param_names = ["batch_size"]
    number = 3
    repeat = 5
    timeout = 300

    def setup(self, batch_size):
        x_train, y_train = make_regression(5_000, BASELINE.n_features, BASELINE.n_outputs)
        self.model = make_model(n_estimators=5, engine="numpy").fit(x_train, y_train)
        self.x, _ = make_regression(
            batch_size, BASELINE.n_features, BASELINE.n_outputs, seed=1730
        )
        self.model.predict(self.x[: min(len(self.x), 8)])

    def time_predict(self, batch_size):
        self.model.predict(self.x)

    def peakmem_predict(self, batch_size):
        self.model.predict(self.x)

    def time_apply(self, batch_size):
        self.model.apply(self.x)


class TimePredictLinearResponse:
    params = PREDICTION_BATCH_SIZES
    param_names = ["batch_size"]
    number = 3
    repeat = 5
    timeout = 300

    def setup(self, batch_size):
        x_train, y_train = make_regression(2_000, 16, 4)
        z_train = make_response_features(x_train, 8)
        self.model = make_model(
            n_estimators=3,
            objective="linear_regression",
            engine="numpy",
        ).fit(x_train, y_train, response_features=z_train)
        self.x, _ = make_regression(batch_size, 16, 4, seed=1730)
        self.response_features = make_response_features(self.x, 8)

    def time_predict_linear_response(self, batch_size):
        self.model.predict(self.x, response_features=self.response_features)
