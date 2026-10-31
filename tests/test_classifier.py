from __future__ import annotations

import pickle

import numpy as np
import pytest

from mbtr import MBTClassifier
from mbtr.losses import BinaryLogLoss, LinearBinaryLogLoss


def classification_data(
    n_samples: int = 180,
    n_features: int = 5,
    n_labels: int = 3,
    seed: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic, learnable binary indicator targets."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    scores = [
        1.8 * x[:, 0] - 0.7 * x[:, 1],
        1.4 * x[:, 1] + 0.9 * x[:, 2],
        1.2 * x[:, 0] * x[:, 2] - 0.8 * x[:, 3],
    ]
    if n_labels > len(scores):
        scores.extend(
            x[:, label % n_features] + 0.35 * x[:, (label + 1) % n_features] for label in range(len(scores), n_labels)
        )
    y = np.column_stack([score > np.median(score) for score in scores[:n_labels]])
    return x, y.astype(np.int64)


def small_classifier(**overrides: object) -> MBTClassifier:
    parameters: dict[str, object] = {
        "n_estimators": 12,
        "learning_rate": 0.25,
        "max_bins": 16,
        "min_samples_leaf": 6,
        "max_depth": 3,
        "random_state": 7,
        "early_stopping_rounds": None,
        "engine": "numpy",
    }
    parameters.update(overrides)
    return MBTClassifier(**parameters)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _binary_cross_entropy_from_logits(target: np.ndarray, logits: np.ndarray) -> float:
    target_matrix = target[:, None] if target.ndim == 1 else target
    logit_matrix = logits[:, None] if logits.ndim == 1 else logits
    signed_target = 2.0 * target_matrix - 1.0
    return float(np.mean(np.logaddexp(0.0, -signed_target * logit_matrix)))


def linear_classification_data(
    n_samples: int = 160,
    seed: int = 417,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create labels carried by response features, independently of split features."""

    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, 3))
    response_features = rng.normal(size=(n_samples, 2))
    scores = np.column_stack(
        (
            1.8 * response_features[:, 0] - 0.7 * response_features[:, 1],
            -0.9 * response_features[:, 0] + 1.5 * response_features[:, 1],
        )
    )
    target = (scores > np.median(scores, axis=0)).astype(np.int64)
    return x, target, response_features


def test_multilabel_fit_predict_probability_and_score_contract() -> None:
    x, y = classification_data()
    model = small_classifier().fit(x, y)

    logits = model.decision_function(x)
    probabilities = model.predict_proba(x)
    prediction = model.predict(x)

    assert logits.shape == y.shape
    assert probabilities.shape == y.shape
    assert prediction.shape == y.shape
    assert np.issubdtype(prediction.dtype, np.integer)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities, _sigmoid(logits), rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(prediction, probabilities >= model.decision_threshold)
    assert model.n_features_in_ == x.shape[1]
    assert model.n_outputs_ == y.shape[1]
    assert len(model.classes_) == y.shape[1]
    for classes in model.classes_:
        np.testing.assert_array_equal(classes, np.array([0, 1]))
    np.testing.assert_array_equal(model.n_classes_, np.full(y.shape[1], 2))
    assert model.score(x, y) == pytest.approx(np.mean(np.all(prediction == y, axis=1)))


def test_binary_target_uses_sklearn_probability_shapes() -> None:
    x, y = classification_data(n_labels=1)
    target = y[:, 0]
    model = small_classifier().fit(x, target)

    logits = model.decision_function(x)
    probabilities = model.predict_proba(x)
    prediction = model.predict(x)

    assert logits.shape == (len(x),)
    assert probabilities.shape == (len(x), 2)
    assert prediction.shape == (len(x),)
    np.testing.assert_allclose(np.sum(probabilities, axis=1), 1.0)
    np.testing.assert_allclose(probabilities[:, 1], _sigmoid(logits), rtol=1e-13, atol=1e-15)
    np.testing.assert_array_equal(prediction, probabilities[:, 1] >= model.decision_threshold)
    np.testing.assert_array_equal(model.classes_, np.array([0, 1]))


def test_training_binary_cross_entropy_is_monotonic_and_matches_probabilities() -> None:
    x, y = classification_data(n_samples=220)
    model = small_classifier(n_estimators=18).fit(x, y)
    losses = np.asarray(model.evals_result_["training"])

    assert losses.size == model.n_estimators_
    assert losses.size > 0
    assert np.all(np.diff(losses) <= 1e-12)

    probabilities = np.clip(model.predict_proba(x), np.finfo(float).eps, 1.0 - np.finfo(float).eps)
    expected = -np.mean(y * np.log(probabilities) + (1 - y) * np.log1p(-probabilities))
    assert losses[-1] == pytest.approx(expected, rel=1e-12, abs=1e-14)


def test_classifier_learns_better_than_the_prevalence_baseline() -> None:
    x, y = classification_data(n_samples=240)
    model = small_classifier(n_estimators=20).fit(x, y)
    probabilities = np.clip(model.predict_proba(x), 1e-15, 1.0 - 1e-15)
    prevalence = np.clip(np.mean(y, axis=0), 1e-15, 1.0 - 1e-15)

    model_loss = -np.mean(y * np.log(probabilities) + (1 - y) * np.log1p(-probabilities))
    baseline_loss = -np.mean(y * np.log(prevalence) + (1 - y) * np.log1p(-prevalence))

    assert model_loss < 0.7 * baseline_loss


def test_decision_threshold_is_configurable_and_validated() -> None:
    x, y = classification_data()
    model = small_classifier(decision_threshold=0.7).fit(x, y)

    np.testing.assert_array_equal(model.predict(x), model.predict_proba(x) >= 0.7)
    assert model.get_params()["decision_threshold"] == 0.7
    assert model.set_params(decision_threshold=0.35) is model
    np.testing.assert_array_equal(model.predict(x), model.predict_proba(x) >= 0.35)

    model.set_params(decision_threshold=1.1)
    with pytest.raises(ValueError, match="decision_threshold"):
        model.predict(x)

    for invalid in (-0.1, 0.0, 1.0, 1.1, np.nan):
        with pytest.raises(ValueError, match="decision_threshold"):
            small_classifier(decision_threshold=invalid).fit(x, y)


@pytest.mark.parametrize(
    "threshold",
    [np.nextafter(0.0, 1.0), np.nextafter(1.0, 0.0)],
)
def test_extreme_decision_thresholds_are_compared_in_logit_space(threshold: float) -> None:
    x, y = classification_data(n_samples=40, n_labels=1)
    model = small_classifier(n_estimators=2, decision_threshold=threshold).fit(x, y[:, 0])
    decision = model.decision_function(x)
    logit_threshold = np.log(threshold) - np.log1p(-threshold)

    np.testing.assert_array_equal(model.predict(x), decision >= logit_threshold)


def test_two_dimensional_single_label_target_preserves_multilabel_shapes() -> None:
    x, y = classification_data(n_samples=60, n_labels=1)
    model = small_classifier(n_estimators=3).fit(x, y)

    assert model.decision_function(x).shape == (len(x), 1)
    assert model.predict_proba(x).shape == (len(x), 1)
    assert model.predict(x).shape == (len(x), 1)
    assert len(model.classes_) == 1
    np.testing.assert_array_equal(model.classes_[0], np.array([0, 1]))
    np.testing.assert_array_equal(model.n_classes_, np.array([2]))


@pytest.mark.parametrize(
    "bad_target",
    [
        np.array([0.0, 0.5, 1.0, 0.0]),
        np.array([0, 1, 2, 0]),
        np.array([[0, 1], [1, -1], [0, 0], [1, 1]]),
    ],
)
def test_non_binary_targets_are_rejected(bad_target: np.ndarray) -> None:
    x = np.arange(12, dtype=np.float64).reshape(4, 3)
    with pytest.raises(ValueError, match="binary|0 and 1|0 or 1"):
        small_classifier(n_estimators=1, min_samples_leaf=1).fit(x, bad_target)


def test_eval_set_targets_are_validated_as_binary() -> None:
    x, y = classification_data(n_samples=80)
    validation_y = y[:20].astype(np.float64)
    validation_y[0, 0] = 0.5

    with pytest.raises(ValueError, match="binary|0 and 1|0 or 1"):
        small_classifier(n_estimators=2).fit(x[20:], y[20:], eval_set=(x[:20], validation_y))


def test_boolean_targets_and_constant_labels_are_supported() -> None:
    x, y = classification_data(n_samples=120, n_labels=2)
    target = np.column_stack((y[:, 0], np.zeros(len(y)), np.ones(len(y)))).astype(bool)
    model = small_classifier(n_estimators=6).fit(x, target)

    probabilities = model.predict_proba(x)
    prediction = model.predict(x)
    assert np.all(np.isfinite(model.decision_function(x)))
    assert np.all(np.isfinite(probabilities))
    assert np.all(prediction[:, 1] == 0)
    assert np.all(prediction[:, 2] == 1)


def test_classifier_is_deterministic_cloneable_and_pickleable() -> None:
    x, y = classification_data(n_samples=140)
    first = small_classifier(n_estimators=7).fit(x, y)
    clone = MBTClassifier(**first.get_params()).fit(x, y)

    np.testing.assert_allclose(first.decision_function(x), clone.decision_function(x), rtol=0, atol=0)
    np.testing.assert_array_equal(first.predict(x), clone.predict(x))

    restored = pickle.loads(pickle.dumps(first))
    np.testing.assert_allclose(restored.predict_proba(x), first.predict_proba(x), rtol=0, atol=0)
    np.testing.assert_array_equal(restored.predict(x), first.predict(x))


def test_predict_methods_before_fit_have_clear_errors() -> None:
    x, _ = classification_data(n_samples=20)
    model = small_classifier(n_estimators=1)

    for method in (model.decision_function, model.predict_proba, model.predict):
        with pytest.raises(RuntimeError, match="not fitted"):
            method(x)


def test_staged_predictions_and_iteration_range_are_consistent() -> None:
    x, y = classification_data(n_samples=120, n_labels=2)
    model = small_classifier(n_estimators=6).fit(x, y)

    staged_decisions = list(model.staged_decision_function(x))
    staged_probabilities = list(model.staged_predict_proba(x))
    staged_predictions = list(model.staged_predict(x))

    assert len(staged_decisions) == model.n_estimators_
    assert len(staged_probabilities) == model.n_estimators_
    assert len(staged_predictions) == model.n_estimators_
    np.testing.assert_allclose(staged_decisions[-1], model.decision_function(x))
    np.testing.assert_allclose(staged_probabilities[-1], model.predict_proba(x))
    np.testing.assert_array_equal(staged_predictions[-1], model.predict(x))
    np.testing.assert_allclose(staged_decisions[1], model.decision_function(x, iteration_range=(0, 2)))
    np.testing.assert_allclose(staged_probabilities[1], model.predict_proba(x, iteration_range=(0, 2)))
    np.testing.assert_array_equal(staged_predictions[1], model.predict(x, iteration_range=(0, 2)))


def test_classifier_eval_set_records_binary_cross_entropy() -> None:
    x, y = classification_data(n_samples=140)
    model = small_classifier(n_estimators=8, early_stopping_rounds=3).fit(x[:100], y[:100], eval_set=(x[100:], y[100:]))

    assert model.evals_result_["validation"]
    assert np.all(np.isfinite(model.evals_result_["validation"]))
    base_logits = np.broadcast_to(model.base_prediction_, y[100:].shape)
    base_loss = _binary_cross_entropy_from_logits(y[100:], base_logits)
    assert model.best_score_ == pytest.approx(
        min(base_loss, *model.evals_result_["validation"])
    )


def test_linear_classifier_early_stopping_can_retain_base_only_model() -> None:
    response_features = np.linspace(-2.0, 2.0, 80)[:, None]
    x = np.ones((len(response_features), 1))
    target = (response_features[:, 0] > 0.0).astype(np.int64)
    validation_target = 1 - target

    model = small_classifier(
        n_estimators=8,
        learning_rate=0.5,
        max_depth=1,
        early_stopping_rounds=2,
        linear_response_curvature="exact",
    ).fit(
        x,
        target,
        response_features=response_features,
        eval_set=(x, validation_target, response_features),
    )

    assert model.linear_response_
    assert model.n_estimators_ == 0
    assert model.best_iteration_ == -1
    assert model.best_score_ == pytest.approx(np.log(2.0))
    assert len(model.evals_result_["validation"]) == 2
    np.testing.assert_allclose(
        model.predict_proba(x, response_features=response_features),
        np.full((len(x), 2), 0.5),
    )


@pytest.mark.parametrize(
    ("target_kind", "expected_decision_width", "expected_probability_width"),
    [
        ("one_dimensional", None, 2),
        ("single_column", 1, 1),
        ("multilabel", 2, 2),
    ],
)
def test_linear_response_classifier_preserves_binary_and_multilabel_api_shapes(
    target_kind: str,
    expected_decision_width: int | None,
    expected_probability_width: int,
) -> None:
    x, multilabel_target, response_features = linear_classification_data(n_samples=120)
    if target_kind == "one_dimensional":
        target = multilabel_target[:, 0]
        expected_decision_shape = (len(x),)
    elif target_kind == "single_column":
        target = multilabel_target[:, :1]
        expected_decision_shape = (len(x), expected_decision_width)
    else:
        target = multilabel_target
        expected_decision_shape = (len(x), expected_decision_width)

    model = small_classifier(n_estimators=5).fit(
        x,
        target,
        response_features=response_features,
    )
    decision = model.decision_function(x, response_features=response_features)
    probability = model.predict_proba(x, response_features=response_features)
    prediction = model.predict(x, response_features=response_features)
    staged_decisions = list(model.staged_decision_function(x, response_features=response_features))
    staged_probabilities = list(model.staged_predict_proba(x, response_features=response_features))
    staged_predictions = list(model.staged_predict(x, response_features=response_features))

    assert model.linear_response_
    assert isinstance(model.objective_, LinearBinaryLogLoss)
    assert model.n_response_features_ == response_features.shape[1]
    assert decision.shape == expected_decision_shape
    assert prediction.shape == expected_decision_shape
    assert probability.shape == (len(x), expected_probability_width)
    prediction_matrix = np.asarray(prediction).reshape(len(x), -1)
    target_matrix = np.asarray(target).reshape(len(x), -1)
    assert model.score(x, target, response_features=response_features) == pytest.approx(
        np.mean(np.all(prediction_matrix == target_matrix, axis=1))
    )
    assert len(staged_decisions) == model.n_estimators_
    np.testing.assert_allclose(staged_decisions[-1], decision)
    np.testing.assert_allclose(staged_probabilities[-1], probability)
    np.testing.assert_array_equal(staged_predictions[-1], prediction)


@pytest.mark.parametrize("curvature", ["upper_bound", "exact"])
def test_linear_response_training_bce_is_monotonic_and_matches_staged_logits(
    curvature: str,
) -> None:
    x, target, response_features = linear_classification_data(n_samples=150)
    model = small_classifier(
        n_estimators=8,
        linear_response_curvature=curvature,
    ).fit(x, target, response_features=response_features)

    staged_losses = np.asarray(
        [
            _binary_cross_entropy_from_logits(target, decision)
            for decision in model.staged_decision_function(x, response_features=response_features)
        ]
    )

    assert model.objective_.curvature == curvature
    assert staged_losses.size == model.n_estimators_
    assert staged_losses.size > 0
    assert np.all(np.isfinite(staged_losses))
    assert np.all(np.diff(staged_losses) <= 1e-12)
    np.testing.assert_allclose(
        staged_losses,
        model.evals_result_["training"],
        rtol=1e-12,
        atol=1e-14,
    )


def test_linear_response_explicit_validation_uses_response_feature_triple() -> None:
    x, target, response_features = linear_classification_data(n_samples=140)
    model = small_classifier(
        n_estimators=6,
        linear_response_curvature="exact",
    ).fit(
        x[:100],
        target[:100],
        response_features=response_features[:100],
        eval_set=(x[100:], target[100:], response_features[100:]),
    )

    validation_logits = model.decision_function(
        x[100:],
        response_features=response_features[100:],
    )
    expected = _binary_cross_entropy_from_logits(target[100:], validation_logits)
    assert len(model.evals_result_["validation"]) == model.n_estimators_
    assert np.all(np.isfinite(model.evals_result_["validation"]))
    assert model.evals_result_["validation"][-1] == pytest.approx(expected, rel=1e-12, abs=1e-14)

    with pytest.raises(ValueError, match="x, y, response_features"):
        small_classifier(n_estimators=2).fit(
            x[:100],
            target[:100],
            response_features=response_features[:100],
            eval_set=(x[100:], target[100:]),
        )
    with pytest.raises(ValueError, match="response_features|columns"):
        small_classifier(n_estimators=2).fit(
            x[:100],
            target[:100],
            response_features=response_features[:100],
            eval_set=(x[100:], target[100:], response_features[100:, :1]),
        )


def test_linear_response_feature_validation_is_consistent_across_classifier_methods() -> None:
    x, target, response_features = linear_classification_data(n_samples=80)

    with pytest.raises(ValueError, match="two-dimensional"):
        small_classifier(n_estimators=2).fit(x, target, response_features=response_features[:, 0])
    with pytest.raises(ValueError, match="80 rows"):
        small_classifier(n_estimators=2).fit(x, target, response_features=response_features[:-1])
    nonfinite = response_features.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        small_classifier(n_estimators=2).fit(x, target, response_features=nonfinite)

    model = small_classifier(n_estimators=3).fit(x, target, response_features=response_features)
    methods = (
        lambda: model.decision_function(x),
        lambda: model.predict_proba(x),
        lambda: model.predict(x),
        lambda: list(model.staged_decision_function(x)),
        lambda: list(model.staged_predict_proba(x)),
        lambda: list(model.staged_predict(x)),
        lambda: model.score(x, target),
    )
    for call in methods:
        with pytest.raises(ValueError, match="response_features"):
            call()

    with pytest.raises(ValueError, match="80 rows"):
        model.decision_function(x, response_features=response_features[:-1])
    with pytest.raises(ValueError, match="2 columns"):
        model.predict_proba(x, response_features=response_features[:, :1])
    with pytest.raises(ValueError, match="finite"):
        model.predict(x, response_features=nonfinite)
    with pytest.raises(ValueError, match="both response_features and x_lr"):
        model.decision_function(
            x,
            response_features=response_features,
            x_lr=response_features,
        )

    constant_model = small_classifier(n_estimators=2).fit(x, target)
    with pytest.raises(ValueError, match="constant leaves"):
        constant_model.decision_function(x, response_features=response_features)


@pytest.mark.parametrize("curvature", ["upper_bound", "exact"])
def test_linear_response_classifier_is_cloneable_and_pickleable(curvature: str) -> None:
    x, target, response_features = linear_classification_data(n_samples=100)
    model = small_classifier(
        n_estimators=4,
        linear_response_curvature=curvature,
        regularize_intercept=True,
    ).fit(x, target, response_features=response_features)
    clone = MBTClassifier(**model.get_params()).fit(
        x,
        target,
        response_features=response_features,
    )
    restored = pickle.loads(pickle.dumps(model))

    assert model.get_params()["linear_response_curvature"] == curvature
    assert model.get_params()["regularize_intercept"] is True
    assert clone.objective_.curvature == curvature
    assert restored.objective_.curvature == curvature
    assert restored.n_response_features_ == response_features.shape[1]
    np.testing.assert_allclose(
        clone.decision_function(x, response_features=response_features),
        model.decision_function(x, response_features=response_features),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        restored.predict_proba(x, response_features=response_features),
        model.predict_proba(x, response_features=response_features),
        rtol=0,
        atol=0,
    )
    np.testing.assert_array_equal(
        restored.predict(x, response_features=response_features),
        model.predict(x, response_features=response_features),
    )


def test_linear_response_leaves_improve_when_split_features_are_constant() -> None:
    train_coordinate = np.linspace(-3.0, 3.0, 240)
    test_coordinate = np.linspace(-2.9875, 2.9875, 200)
    x_train = np.ones((len(train_coordinate), 1))
    x_test = np.ones((len(test_coordinate), 1))
    response_train = train_coordinate[:, None]
    response_test = test_coordinate[:, None]
    y_train = np.column_stack((train_coordinate > 0.0, train_coordinate < 0.0)).astype(np.int64)
    y_test = np.column_stack((test_coordinate > 0.0, test_coordinate < 0.0)).astype(np.int64)
    parameters = dict(
        n_estimators=10,
        learning_rate=0.5,
        max_bins=8,
        min_samples_leaf=10,
        max_depth=2,
        l2_regularization=0.1,
        early_stopping_rounds=None,
        engine="numpy",
        random_state=0,
    )

    constant_model = MBTClassifier(**parameters).fit(x_train, y_train)
    linear_model = MBTClassifier(**parameters).fit(
        x_train,
        y_train,
        response_features=response_train,
    )
    constant_loss = _binary_cross_entropy_from_logits(
        y_test,
        constant_model.decision_function(x_test),
    )
    linear_loss = _binary_cross_entropy_from_logits(
        y_test,
        linear_model.decision_function(x_test, response_features=response_test),
    )

    assert linear_loss < 0.35 * constant_loss


def test_exact_linear_response_remains_finite_at_extreme_logits() -> None:
    rng = np.random.default_rng(19)
    x = np.ones((60, 1))
    response_features = rng.normal(size=(60, 2))
    target = np.ones((60, 2), dtype=np.int64)
    model = small_classifier(
        n_estimators=3,
        linear_response_curvature="exact",
        l2_regularization=0.2,
    ).fit(x, target, response_features=response_features)
    decision = model.decision_function(x, response_features=response_features)

    assert np.min(decision) > 30.0
    assert np.all(np.isfinite(decision))
    assert np.all(np.isfinite(model.evals_result_["training"]))
    assert np.all(np.diff(model.evals_result_["training"]) <= 1e-12)


def test_linear_response_hyperparameters_are_validated() -> None:
    x, target, response_features = linear_classification_data(n_samples=40)

    with pytest.raises(ValueError, match="linear_response_curvature"):
        small_classifier(linear_response_curvature="diagonal").fit(
            x,
            target,
            response_features=response_features,
        )
    with pytest.raises(ValueError, match="regularize_intercept"):
        small_classifier(regularize_intercept="yes").fit(  # type: ignore[arg-type]
            x,
            target,
            response_features=response_features,
        )


def test_nonfinite_linear_response_candidate_is_not_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinate = np.linspace(-2.0, 2.0, 60)
    x = np.ones((len(coordinate), 1))
    response_features = coordinate[:, None]
    target = (coordinate > 0.0).astype(np.int64)
    original_loss = LinearBinaryLogLoss.loss

    def nonfinite_away_from_base(
        self: LinearBinaryLogLoss,
        y: np.ndarray,
        prediction: np.ndarray,
    ) -> float:
        if np.any(prediction != 0.0):
            return float("nan")
        return original_loss(self, y, prediction)

    monkeypatch.setattr(LinearBinaryLogLoss, "loss", nonfinite_away_from_base)
    model = small_classifier(n_estimators=2).fit(
        x,
        target,
        response_features=response_features,
    )

    assert model.n_estimators_ == 0
    assert model.evals_result_["training"] == []
    assert np.all(np.isfinite(model.decision_function(x, response_features=response_features)))


@pytest.mark.parametrize(
    ("target", "logit", "gradient_sign"),
    [
        (1.0, 40.0, 1.0),
        (1.0, 50.0, 1.0),
        (0.0, -40.0, -1.0),
        (0.0, -50.0, -1.0),
    ],
)
def test_binary_log_loss_preserves_high_margin_signal(target: float, logit: float, gradient_sign: float) -> None:
    """Correct, high-confidence examples must not underflow to exact zero."""

    objective = BinaryLogLoss(lambda_weights=0.0).configure(1)
    y = np.array([[target]])
    raw_prediction = np.array([[logit]])

    negative_gradient, hessian = objective.gradient_hessian(y, raw_prediction)
    loss = objective.loss(y, raw_prediction)
    tail = np.exp(-abs(logit))

    assert np.isfinite(loss)
    assert loss > 0.0
    assert gradient_sign * negative_gradient[0, 0] > 0.0
    assert hessian[0, 0] > 0.0
    assert loss == pytest.approx(np.logaddexp(0.0, -abs(logit)), rel=1e-14)
    assert abs(negative_gradient[0, 0]) == pytest.approx(tail / (1.0 + tail), rel=1e-14)
    assert hessian[0, 0] == pytest.approx(tail / (1.0 + tail) ** 2, rel=1e-14)
