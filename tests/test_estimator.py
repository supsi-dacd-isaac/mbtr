from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

from mbtr import MBT, MBTRegressor, Tree
from mbtr.losses import MSE


def regression_data(
    n_samples: int = 120,
    n_features: int = 4,
    n_outputs: int = 2,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    y_columns = []
    for output in range(n_outputs):
        signal = (output + 1.0) * (x[:, 0] > 0.0)
        signal += (0.7 + 0.1 * output) * x[:, 1]
        signal -= 0.25 * x[:, 2] ** 2
        y_columns.append(signal + rng.normal(scale=0.03, size=n_samples))
    return x, np.column_stack(y_columns)


def small_model(**overrides: object) -> MBTRegressor:
    parameters: dict[str, object] = {
        "n_estimators": 8,
        "learning_rate": 0.2,
        "max_bins": 12,
        "min_samples_leaf": 6,
        "max_depth": 3,
        "random_state": 5,
        "early_stopping_rounds": None,
        "engine": "numpy",
    }
    parameters.update(overrides)
    return MBTRegressor(**parameters)


def test_fit_predict_improves_over_constant_baseline_and_preserves_shape() -> None:
    x, y = regression_data()
    model = small_model().fit(x, y)

    prediction = model.predict(x)
    baseline = np.broadcast_to(np.mean(y, axis=0), y.shape)

    assert prediction.shape == y.shape
    assert np.mean((y - prediction) ** 2) < np.mean((y - baseline) ** 2) * 0.45
    assert model.n_features_in_ == x.shape[1]
    assert model.n_outputs_ == y.shape[1]
    assert model.n_estimators_ == len(model.trees_)
    assert np.isclose(np.sum(model.feature_importances_), 1.0)


def test_one_dimensional_target_returns_one_dimensional_prediction() -> None:
    x, y = regression_data(n_outputs=1)
    model = small_model().fit(x, y[:, 0])
    assert model.predict(x).shape == (len(x),)


def test_accepted_training_rounds_are_monotonic() -> None:
    x, y = regression_data()
    model = small_model(n_estimators=12).fit(x, y)
    losses = np.asarray(model.evals_result_["training"])

    assert losses.size > 0
    assert losses.size == model.n_estimators_
    assert np.all(np.diff(losses) <= 1e-12)


@pytest.mark.parametrize(
    ("objective", "objective_params"),
    [
        ("quantile", {"alphas": [0.2, 0.8], "smoothing": 0.5}),
        ("quadratic_quantile", {"alphas": [0.2, 0.8], "quadratic_scale": 100.0}),
        ("pseudo_huber_quantile", {"alphas": [0.2, 0.8], "delta": 0.5}),
    ],
)
def test_quantile_rounds_are_monotonic_in_the_public_metric(
    objective: str, objective_params: dict[str, object]
) -> None:
    x, y = regression_data(n_samples=90, n_outputs=2)
    model = small_model(
        objective=objective,
        objective_params=objective_params,
        n_estimators=8,
    ).fit(x, y)
    losses = np.asarray(model.evals_result_["training"])

    assert losses.size > 0
    assert losses.size == model.n_estimators_
    assert np.all(np.diff(losses) <= 1e-12)


@pytest.mark.parametrize(
    ("objective", "objective_params"),
    [
        ("quantile", {"alphas": [0.2, 0.8], "smoothing": 0.5}),
        ("quadratic_quantile", {"alphas": [0.2, 0.8], "quadratic_scale": 100.0}),
    ],
)
def test_published_quantile_training_metric_is_staged_pinball(
    objective: str, objective_params: dict[str, object]
) -> None:
    x, y = regression_data(n_samples=100, n_outputs=2)
    alphas = np.asarray(objective_params["alphas"])
    model = small_model(
        objective=objective,
        objective_params=objective_params,
        n_estimators=6,
    ).fit(x, y)

    staged_losses = []
    for prediction in model.staged_predict(x):
        residual = y[:, :, None] - prediction
        staged_losses.append(
            np.mean(
                np.maximum(
                    alphas[None, None, :] * residual,
                    (alphas[None, None, :] - 1.0) * residual,
                )
            )
        )

    np.testing.assert_allclose(staged_losses, model.evals_result_["training"])
    assert np.all(np.diff(staged_losses) <= 1e-12)


def test_min_samples_leaf_is_an_invariant_for_every_tree() -> None:
    x, y = regression_data(n_samples=150)
    minimum = 9
    model = small_model(min_samples_leaf=minimum, max_depth=5).fit(x, y)

    for tree in model.trees_:
        leaf_sizes = [node.sample_count for node in tree.nodes if node.is_leaf]
        assert leaf_sizes
        assert min(leaf_sizes) >= minimum
        assert sum(leaf_sizes) == len(x)


def test_validation_split_and_early_stopping_are_deterministic() -> None:
    x, y = regression_data(n_samples=160, seed=21)
    parameters = {
        "n_estimators": 18,
        "learning_rate": 0.25,
        "max_bins": 10,
        "min_samples_leaf": 5,
        "max_depth": 4,
        "validation_fraction": 0.25,
        "early_stopping_rounds": 3,
        "random_state": 77,
        "engine": "numpy",
    }

    first = MBTRegressor(**parameters).fit(x, y)
    second = MBTRegressor(**parameters).fit(x, y)

    assert first.best_iteration_ == second.best_iteration_
    assert first.n_estimators_ == second.n_estimators_
    assert first.evals_result_ == second.evals_result_
    np.testing.assert_allclose(first.predict(x), second.predict(x), rtol=0, atol=0)
    assert first.evals_result_["validation"]


def test_explicit_eval_set_records_validation_metric() -> None:
    x, y = regression_data(n_samples=140)
    train = np.arange(100)
    validation = np.arange(100, 140)
    model = small_model(early_stopping_rounds=2).fit(x[train], y[train], eval_set=(x[validation], y[validation]))

    assert model.evals_result_["validation"]
    assert np.all(np.isfinite(model.evals_result_["validation"]))
    assert model.best_iteration_ < len(model.evals_result_["validation"])


def test_early_stopping_can_retain_the_base_only_regressor() -> None:
    coordinate = np.linspace(-1.0, 1.0, 80)
    x = coordinate[:, None]
    y = np.where(coordinate < 0.0, 0.0, 2.0)
    validation_y = np.ones_like(y)

    model = small_model(
        n_estimators=8,
        learning_rate=0.5,
        max_depth=1,
        early_stopping_rounds=2,
    ).fit(x, y, eval_set=(x, validation_y))

    assert model.n_estimators_ == 0
    assert model.best_iteration_ == -1
    assert model.best_score_ == pytest.approx(0.0)
    assert len(model.evals_result_["validation"]) == 2
    np.testing.assert_allclose(model.predict(x), np.ones_like(y))
    assert model.apply(x).shape == (len(x), 0)


def test_early_stopping_discards_the_patience_tail_and_restores_the_best_prefix() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(180, 4))
    y = np.column_stack(
        (
            x[:, 0] + 0.2 * rng.normal(size=180),
            x[:, 1] ** 2 + 0.2 * rng.normal(size=180),
        )
    )
    parameters = {
        "n_estimators": 80,
        "learning_rate": 0.3,
        "max_bins": 16,
        "min_samples_leaf": 5,
        "max_depth": 4,
        "engine": "numpy",
    }

    stopped = MBTRegressor(**parameters, early_stopping_rounds=5).fit(
        x[:100],
        y[:100],
        eval_set=(x[100:140], y[100:140]),
    )
    full = MBTRegressor(**parameters).fit(
        x[:100],
        y[:100],
        eval_set=(x[100:140], y[100:140]),
    )

    evaluated_rounds = len(stopped.evals_result_["validation"])
    assert stopped.best_iteration_ == 18
    assert stopped.n_estimators_ == 19
    assert evaluated_rounds == 24
    assert evaluated_rounds - stopped.n_estimators_ == 5
    assert stopped.best_score_ == pytest.approx(min(stopped.evals_result_["validation"]))
    np.testing.assert_allclose(
        stopped.predict(x[140:]),
        full.predict(x[140:], iteration_range=(0, stopped.n_estimators_)),
        rtol=0.0,
        atol=0.0,
    )


def test_rejected_first_tree_keeps_validation_base_score() -> None:
    class RejectNonconstantCandidate(MSE):
        def loss(self, y: np.ndarray, prediction: np.ndarray) -> float:
            if np.any(np.ptp(prediction, axis=0) > 0.0):
                return float("inf")
            return super().loss(y, prediction)

    coordinate = np.linspace(-1.0, 1.0, 60)
    x = coordinate[:, None]
    y = np.where(coordinate < 0.0, 0.0, 2.0)
    validation_y = np.full_like(y, 3.0)
    model = small_model(
        objective=RejectNonconstantCandidate(),
        n_estimators=2,
        refit_leaves=False,
    ).fit(x, y, eval_set=(x, validation_y))

    expected_validation_base_loss = 0.5 * float(np.mean(np.square(validation_y - 1.0)))
    training_base_loss = 0.5 * float(np.mean(np.square(y - 1.0)))
    assert model.n_estimators_ == 0
    assert model.best_iteration_ == -1
    assert model.evals_result_ == {"training": [], "validation": []}
    assert model.best_score_ == pytest.approx(expected_validation_base_loss)
    assert model.best_score_ != pytest.approx(training_base_loss)


def test_linear_response_features_fit_a_linear_leaf_model() -> None:
    rng = np.random.default_rng(4)
    response_features = rng.normal(size=(100, 2))
    # Constant split features force the ridge response to do all useful work.
    x = np.ones((100, 2))
    coefficients = np.array([[2.0, -1.0], [0.5, 3.0]])
    intercept = np.array([1.25, -0.75])
    y = response_features @ coefficients + intercept

    model = small_model(
        objective="linear_regression",
        n_estimators=2,
        learning_rate=1.0,
        l2_regularization=1e-10,
        min_samples_leaf=5,
    ).fit(x, y, response_features=response_features)

    prediction = model.predict(x, response_features=response_features)
    np.testing.assert_allclose(prediction, y, atol=1e-8, rtol=1e-8)
    with pytest.raises(ValueError, match="response_features"):
        model.predict(x)


@pytest.mark.parametrize("linear", [False, True])
def test_output_sketch_uses_full_output_leaf_values(linear: bool) -> None:
    x, y = regression_data(n_samples=120, n_outputs=4)
    response_features = x[:, :2]
    parameters: dict[str, object] = {
        "n_estimators": 4,
        "output_sketch_size": 2,
        "max_depth": 2,
        "min_samples_leaf": 8,
    }
    if linear:
        parameters["objective"] = "linear_regression"
        model = small_model(**parameters).fit(x, y, response_features=response_features)
        prediction = model.predict(x, response_features=response_features)
    else:
        model = small_model(**parameters).fit(x, y)
        prediction = model.predict(x)

    assert prediction.shape == y.shape
    assert np.all(np.diff(model.evals_result_["training"]) <= 1e-12)
    for tree in model.trees_:
        np.testing.assert_equal(tree.split_output_indices_.shape, (2,))
        for node in tree.nodes:
            if node.is_leaf:
                expected_shape = (response_features.shape[1] + 1, y.shape[1]) if linear else (y.shape[1],)
                assert node.value.shape == expected_shape


def test_inactive_output_sketch_is_identical_to_full_output_search() -> None:
    x, y = regression_data(n_samples=110, n_outputs=3)
    parameters = dict(n_estimators=4, max_depth=2, min_samples_leaf=8)
    ordinary = small_model(**parameters).fit(x, y)
    inactive = small_model(output_sketch_size=y.shape[1], **parameters).fit(x, y)

    np.testing.assert_array_equal(ordinary.predict(x), inactive.predict(x))
    assert all(tree.split_output_indices_ is None for tree in inactive.trees_)


def test_large_linear_response_contractions_do_not_emit_runtime_warnings() -> None:
    rng = np.random.default_rng(41)
    response_features = rng.normal(size=(3_000, 8))
    coefficients = rng.normal(size=(8, 3))
    y = np.einsum("ni,io->no", response_features, coefficients) + np.array([0.5, -1.0, 2.0])
    x = np.ones((len(y), 1))

    # Accelerate-backed NumPy 2.2 wheels can report false floating-point
    # errors for the equivalent large matrix products.  Linear-response MBTR
    # must remain warning-free while preserving finite fitted predictions.
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = small_model(
            objective="linear_regression",
            n_estimators=1,
            learning_rate=1.0,
            l2_regularization=1e-10,
            min_samples_leaf=5,
        ).fit(x, y, response_features=response_features)
        prediction = model.predict(x, response_features=response_features)

    assert np.all(np.isfinite(prediction))
    np.testing.assert_allclose(prediction, y, atol=1e-9, rtol=1e-9)


def test_quantile_prediction_has_explicit_three_dimensional_shape() -> None:
    x, y = regression_data(n_samples=100, n_outputs=2)
    levels = [0.1, 0.5, 0.9]
    model = small_model(
        objective="quantile",
        objective_params={"alphas": levels, "smoothing": 0.5},
        n_estimators=4,
    ).fit(x, y)

    prediction = model.predict(x)
    assert prediction.shape == (len(x), y.shape[1], len(levels))
    assert np.all(np.isfinite(prediction))
    residual = y[:, :, None] - prediction
    expected_pinball = np.mean(
        np.maximum(
            np.asarray(levels)[None, None, :] * residual,
            (np.asarray(levels)[None, None, :] - 1.0) * residual,
        )
    )
    assert model.score(x, y) == pytest.approx(-expected_pinball)


@pytest.mark.parametrize("objective", ["quantile", "quadratic_quantile"])
def test_quantile_quality_on_known_heteroscedastic_distribution(objective: str) -> None:
    def sample(seed: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(-2.0, 2.0, size=(size, 5))
        location = np.column_stack(
            (
                np.sin(x[:, 0]) + 0.25 * x[:, 1] ** 2,
                0.7 * np.sin(x[:, 0]) + np.cos(x[:, 2]),
            )
        )
        scale = np.column_stack(
            (
                0.25 + 0.5 / (1.0 + np.exp(-x[:, 3])),
                0.3 + 0.2 * np.abs(x[:, 4]),
            )
        )
        noise = rng.exponential(size=size) - 1.0
        return x, location + scale * noise[:, None]

    alphas = np.array([0.1, 0.5, 0.9])
    x_train, y_train = sample(2024, 800)
    x_test, y_test = sample(2025, 2_000)
    model = MBTRegressor(
        objective=objective,
        objective_params={"alphas": alphas, "smoothing": 0.7},
        n_estimators=40,
        learning_rate=0.08,
        max_bins=32,
        min_samples_leaf=20,
        max_depth=3,
        early_stopping_rounds=None,
        engine="numpy",
        random_state=0,
    ).fit(x_train, y_train)
    prediction = model.predict(x_test)
    residual = y_test[:, :, None] - prediction
    pinball = float(
        np.mean(
            np.maximum(
                alphas[None, None, :] * residual,
                (alphas[None, None, :] - 1.0) * residual,
            )
        )
    )
    baseline_values = np.quantile(y_train, alphas, axis=0).T
    baseline_residual = y_test[:, :, None] - baseline_values[None, :, :]
    baseline_pinball = float(
        np.mean(
            np.maximum(
                alphas[None, None, :] * baseline_residual,
                (alphas[None, None, :] - 1.0) * baseline_residual,
            )
        )
    )
    coverage = np.mean(y_test[:, :, None] <= prediction, axis=0)
    calibration_error = float(np.mean(np.abs(coverage - alphas[None, :])))
    crossing_rate = float(np.mean(prediction[:, :, :-1] > prediction[:, :, 1:]))

    assert calibration_error < 0.06
    assert crossing_rate < 0.01
    assert pinball < 0.8 * baseline_pinball


def test_constant_and_duplicate_features_are_handled_deterministically() -> None:
    x = np.column_stack(
        (
            np.ones(80),
            np.repeat([0.0, 1.0, 2.0, 3.0], 20),
            np.repeat([0.0, 1.0, 2.0, 3.0], 20),
        )
    )
    y = (x[:, 1] > 1.0).astype(float)

    first = small_model(n_estimators=3, min_samples_leaf=8).fit(x, y)
    second = small_model(n_estimators=3, min_samples_leaf=8).fit(x, y)

    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    assert first.feature_importances_[0] == 0.0


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_common_input_dtypes_are_normalized(dtype: np.dtype) -> None:
    x, y = regression_data(n_samples=90, n_outputs=1)
    x_scaled = np.round(x * 10).astype(dtype)
    model = small_model(n_estimators=2).fit(x_scaled, y.astype(dtype))
    prediction = model.predict(x_scaled)
    assert prediction.dtype == np.float64
    assert np.all(np.isfinite(prediction))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_estimators": 0}, "n_estimators"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"max_bins": 1}, "max_bins"),
        ({"min_samples_leaf": 0}, "min_samples_leaf"),
        ({"max_depth": 0}, "max_depth"),
        ({"engine": "gpu"}, "engine"),
        ({"output_sketch_size": 0}, "output_sketch_size"),
        ({"output_sketch_size": True}, "output_sketch_size"),
    ],
)
def test_invalid_hyperparameters_fail_early(kwargs: dict[str, object], message: str) -> None:
    x, y = regression_data(n_samples=30, n_outputs=1)
    with pytest.raises(ValueError, match=message):
        small_model(**kwargs).fit(x, y)


def test_invalid_data_and_predict_before_fit_have_clear_errors() -> None:
    x, y = regression_data(n_samples=40, n_outputs=1)
    model = small_model(n_estimators=1)
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict(x)
    with pytest.raises(ValueError, match="same number"):
        model.fit(x[:-1], y)
    x_bad = x.copy()
    x_bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(x_bad, y)


def test_staged_predict_apply_and_iteration_range_are_consistent() -> None:
    x, y = regression_data(n_samples=90)
    model = small_model(n_estimators=5).fit(x, y)

    staged = list(model.staged_predict(x))
    assert len(staged) == model.n_estimators_
    np.testing.assert_allclose(staged[-1], model.predict(x))
    np.testing.assert_allclose(staged[1], model.predict(x, iteration_range=(0, 2)))
    leaves = model.apply(x)
    assert leaves.shape == (len(x), model.n_estimators_)
    assert np.issubdtype(leaves.dtype, np.integer)


def test_get_set_params_and_pickle_roundtrip() -> None:
    x, y = regression_data(n_samples=80)
    model = small_model(n_estimators=3)
    params = model.get_params()
    assert params["n_estimators"] == 3
    assert model.set_params(n_estimators=4) is model
    assert model.n_estimators == 4
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.set_params(does_not_exist=True)

    model.fit(x, y)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(x), model.predict(x), rtol=0, atol=0)


def test_legacy_mbt_and_tree_entry_points_remain_usable() -> None:
    x, y = regression_data(n_samples=80, n_outputs=1)
    legacy = MBT(
        n_boosts=3,
        early_stopping_rounds=None,
        n_q=8,
        min_leaf=5,
        verbose=0,
    ).fit(x, y)
    assert legacy.predict(x).shape == y.shape
    assert len(legacy.trees) == legacy.n_estimators_

    residual = y - np.mean(y, axis=0)
    tree = Tree(n_q=8, min_leaf=5, max_depth=2).fit(x, residual)
    assert tree.predict(x).shape == y.shape


def test_scored_partition_matches_fitted_and_prediction_partition() -> None:
    x = np.array([[0.0], [0.0], [1.0], [1.0], [2.0], [2.0], [3.0], [3.0]])
    gradient = np.array([[0.0], [0.0], [0.0], [0.0], [5.0], [5.0], [5.0], [5.0]])
    tree = Tree(
        n_q=4,
        min_leaf=2,
        lambda_weights=0.0,
        lambda_leaves=0.0,
        max_depth=1,
    ).fit(x, gradient)

    assert tree._tree is not None
    root = tree._tree.nodes[0]
    assert not root.is_leaf
    expected_left = int(np.sum(x[:, root.feature] < root.threshold))
    assert tree._tree.nodes[root.left].sample_count == expected_left
    assert tree._tree.nodes[root.right].sample_count == len(x) - expected_left

    # Values equal to the threshold must follow the right branch everywhere.
    at_threshold = np.array([[root.threshold]])
    right_probe = np.nextafter(root.threshold, np.inf)
    np.testing.assert_allclose(tree.predict(at_threshold), tree.predict(np.array([[right_probe]])))
