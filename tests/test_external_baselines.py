from __future__ import annotations

import importlib

import numpy as np
import pytest

from studies.multivariate.baselines import (
    ExternalBaselineConfig,
    _base_prediction,
    _base_raw_prediction,
    _catboost_objective,
    _objective,
    _SelectionState,
    fit_external_baseline,
)


def test_selection_state_can_retain_the_intercept() -> None:
    state = _SelectionState(best_loss=1.0, patience=3)

    assert state.observe(1.1) is False
    assert state.observe(1.0) is False
    assert state.observe(1.2) is True
    assert state.best_round == 0
    assert state.evaluated_rounds == 3


def test_selection_state_uses_strict_tolerance_and_common_rounds() -> None:
    state = _SelectionState(best_loss=1.0, patience=2)

    assert state.observe(0.9) is False
    assert state.best_round == 1
    assert state.observe(0.9 - 0.5e-14) is False
    assert state.observe(0.91) is True
    assert state.best_round == 1


def test_matrix_objectives_and_catboost_rmse_conversion_agree() -> None:
    target = np.asarray([[0.0, 2.0], [2.0, 4.0]])
    prediction = np.asarray([[1.0, 2.0], [1.0, 2.0]])
    expected = 0.5 * np.mean(np.square(target - prediction))
    catboost_multi_rmse = np.sqrt(np.sum(np.square(target - prediction)) / len(target))

    assert _objective(target, prediction, "regression") == pytest.approx(expected)
    assert _catboost_objective(catboost_multi_rmse, "regression", 2) == pytest.approx(expected)


def test_multilabel_base_prediction_uses_per_label_prevalence() -> None:
    target = np.asarray([[0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    raw = _base_raw_prediction(target, "classification")
    probability = _base_prediction(raw, 3, "classification")

    np.testing.assert_allclose(probability[:, 0], 0.5)
    assert np.all(probability[:, 1] > 1.0 - 1e-14)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("n_estimators", 0),
        ("learning_rate", 0.0),
        ("max_bins", 1),
        ("min_samples_leaf", 0),
        ("l2_regularization", -1.0),
        ("early_stopping_rounds", 0),
        ("native_threads", 0),
    ],
)
def test_external_config_rejects_invalid_controls(field: str, value: object) -> None:
    values = ExternalBaselineConfig().__dict__ | {field: value}
    with pytest.raises(ValueError):
        ExternalBaselineConfig(**values).validate()


def test_unknown_baseline_fails_before_importing_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_import(name: str) -> object:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    arrays = {
        "x_train": np.zeros((3, 1)),
        "y_train": np.zeros((3, 1)),
        "x_validation": np.zeros((2, 1)),
        "y_validation": np.zeros((2, 1)),
        "x_test": np.zeros((2, 1)),
        "y_test": np.zeros((2, 1)),
    }
    with pytest.raises(ValueError, match="Unknown external baseline"):
        fit_external_baseline("not-a-model", arrays, task="regression", config=ExternalBaselineConfig())


def _tiny_external_arrays(task: str) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(901)
    features = generator.normal(size=(72, 4))
    continuous = np.column_stack(
        (
            1.2 * features[:, 0] - 0.4 * features[:, 1],
            features[:, 2] + 0.7 * features[:, 3],
        )
    )
    target = continuous if task == "regression" else (continuous > 0.0).astype(np.float64)
    return {
        "x_train": features[:40],
        "y_train": target[:40],
        "x_validation": features[40:56],
        "y_validation": target[40:56],
        "x_test": features[56:],
        "y_test": target[56:],
    }


@pytest.mark.parametrize(
    ("name", "package", "trees_per_round"),
    [
        ("xgboost", "xgboost", 2),
        ("xgboost-multi-output", "xgboost", 1),
        ("lightgbm", "lightgbm", 2),
        ("catboost", "catboost", 1),
    ],
)
@pytest.mark.parametrize("task", ["regression", "classification"])
def test_optional_external_adapter_smoke(
    name: str,
    package: str,
    trees_per_round: int,
    task: str,
) -> None:
    pytest.importorskip(package)
    arrays = _tiny_external_arrays(task)
    config = ExternalBaselineConfig(
        n_estimators=3,
        learning_rate=0.1,
        max_bins=8,
        min_samples_leaf=2,
        max_depth=2,
        early_stopping_rounds=2,
        native_threads=1,
    )

    run = fit_external_baseline(name, arrays, task=task, config=config)  # type: ignore[arg-type]

    assert run.prediction.shape == arrays["y_test"].shape
    assert np.all(np.isfinite(run.prediction))
    assert 0 <= run.rounds_retained <= run.rounds_evaluated <= config.n_estimators
    assert run.trees_per_round == trees_per_round
    assert run.effective_trees_evaluated == run.rounds_evaluated * trees_per_round
    assert len(run.training_loss) == run.rounds_evaluated
    assert len(run.validation_loss) == run.rounds_evaluated
