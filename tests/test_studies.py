from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from studies.adapters import IndependentLabelsClassifier, IndependentOutputsRegressor, make_estimator
from studies.aggregate import aggregate_documents
from studies.catalog import find_published_scores, load_published_scores, load_suites, validate_catalog
from studies.metrics import (
    average_relative_rmse,
    binary_cross_entropy,
    classification_metrics,
    global_rmse,
    mean_fold_average_relative_rmse,
    mean_target_r2,
    target_relative_rmse,
)
from studies.run import _restore_target_scale, _validate_task, main


def small_parameters() -> dict[str, object]:
    return {
        "n_estimators": 3,
        "learning_rate": 0.2,
        "max_bins": 8,
        "min_samples_leaf": 3,
        "max_depth": 2,
        "random_state": 7,
        "engine": "numpy",
    }


def test_catalog_cross_references_and_published_scores_are_valid() -> None:
    validate_catalog()
    scores = load_published_scores()

    assert len(scores) == 65
    assert sum(score.comparison_tier == "direct" for score in scores) == 20
    assert {score.dataset_id for score in scores} >= {"rf1", "scm20d", "mediamill", "delicious", "moa"}
    assert {suite["id"] for suite in load_suites()} >= {
        "sketchboost_2022",
        "uco_kdis_multilabel_2024",
        "mocr_icml2025",
        "openml_cc18",
        "talent_tiny_2025",
    }
    assert len(find_published_scores("rf1", "sketchboost_random_holdout_cv5_2022", {"global_rmse"})) == 7

    test_selected_methods = {
        "SketchBoost Top Outputs",
        "SketchBoost Random Sampling",
        "SketchBoost Random Projection",
    }
    test_selected_scores = [score for score in scores if score.method in test_selected_methods]
    assert len(test_selected_scores) == 15
    assert {score.comparison_tier for score in test_selected_scores} == {"context_only"}
    assert all("test-selected" in score.notes for score in test_selected_scores)
    assert all("excluded from clean comparisons and rankings" in score.notes for score in test_selected_scores)

    clean_sketchboost_scores = [
        score for score in scores if score.protocol_id.startswith("sketchboost_") and score.comparison_tier == "direct"
    ]
    assert len(clean_sketchboost_scores) == 20
    assert not ({score.method for score in clean_sketchboost_scores} & test_selected_methods)

    openml_manifest = json.loads(
        (Path(__file__).parents[1] / "studies" / "openml" / "manifest.json").read_text(encoding="utf-8")
    )
    assert {suite["id"] for suite in openml_manifest["suites"]} == {99, 269, 271, 353}
    assert openml_manifest["suites"][1]["repeated_cv_task_ids"] == [361617, 361618, 361619, 361621, 361622]

    talent_manifest = json.loads(
        (Path(__file__).parents[1] / "studies" / "talent" / "manifest.json").read_text(encoding="utf-8")
    )
    assert talent_manifest["suites"][0]["supported_now"] == 33
    assert talent_manifest["suites"][1]["supported_now"] == 220
    assert talent_manifest["source_revision"] is None


def test_regression_metric_definitions() -> None:
    y_train = np.array([[0.0, 0.0], [2.0, 4.0]])
    y_true = np.array([[0.0, 0.0], [2.0, 4.0]])
    y_pred = np.array([[0.0, 1.0], [1.0, 4.0]])

    assert global_rmse(y_true, y_pred) == pytest.approx(np.sqrt(0.5))
    np.testing.assert_allclose(target_relative_rmse(y_train, y_true, y_pred), [np.sqrt(0.5), np.sqrt(0.125)])
    assert average_relative_rmse(y_train, y_true, y_pred) == pytest.approx((np.sqrt(0.5) + np.sqrt(0.125)) / 2.0)
    assert mean_target_r2(y_true, y_pred) == pytest.approx(0.6875)

    folds = [(y_train, y_true, y_pred), (y_train, y_true, y_true)]
    assert mean_fold_average_relative_rmse(folds) == pytest.approx(average_relative_rmse(y_train, y_true, y_pred) / 2.0)
    np.testing.assert_allclose(
        _restore_target_scale(
            {"target_offset": np.array([10.0, -2.0]), "target_scale": np.array([2.0, 4.0])},
            y_true,
        ),
        y_true * [2.0, 4.0] + [10.0, -2.0],
    )


def test_classification_metrics_use_probabilities_and_exact_label_semantics() -> None:
    target = np.array([[1, 0], [0, 1], [1, 1]])
    probability = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6]])
    metrics = classification_metrics(target, probability)

    expected_bce = -np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability))
    assert binary_cross_entropy(target, probability) == pytest.approx(expected_bce)
    assert metrics["binary_cross_entropy"] == pytest.approx(expected_bce)
    assert metrics["hamming_loss"] == 0.0
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["example_jaccard"] == 1.0
    assert metrics["example_f1"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0

    with pytest.raises(ValueError, match="between zero and one"):
        binary_cross_entropy(target, probability + 1.0)


def test_independent_adapters_fit_one_model_per_output() -> None:
    rng = np.random.default_rng(12)
    x = rng.normal(size=(80, 4))
    y_regression = np.column_stack((x[:, 0] - x[:, 1], x[:, 2] + 0.5 * x[:, 3]))
    y_labels = np.column_stack((x[:, 0] > 0.0, x[:, 1] + x[:, 2] > 0.0)).astype(int)

    regressor = make_estimator("regression", "independent", small_parameters()).fit(x, y_regression)
    classifier = make_estimator("multilabel", "independent", small_parameters()).fit(x, y_labels)

    assert isinstance(regressor, IndependentOutputsRegressor)
    assert isinstance(classifier, IndependentLabelsClassifier)
    assert len(regressor.estimators_) == y_regression.shape[1]
    assert len(classifier.estimators_) == y_labels.shape[1]
    assert regressor.predict(x).shape == y_regression.shape
    assert classifier.predict_proba(x).shape == y_labels.shape

    threshold_classifier = make_estimator(
        "multilabel",
        "independent",
        {**small_parameters(), "decision_threshold": 0.8},
    ).fit(x, y_labels)
    np.testing.assert_array_equal(
        threshold_classifier.predict(x),
        threshold_classifier.predict_proba(x) >= 0.8,
    )


def test_runner_rejects_binary_and_multilabel_shape_confusion() -> None:
    common = {
        "x_train": np.ones((4, 2)),
        "x_test": np.ones((2, 2)),
        "y_train": np.ones((4, 2)),
        "y_test": np.ones((2, 2)),
    }
    with pytest.raises(ValueError, match="exactly one"):
        _validate_task(common, "binary")

    common["y_train"] = np.ones(4)
    common["y_test"] = np.ones(2)
    with pytest.raises(ValueError, match="at least two"):
        _validate_task(common, "multilabel")


def test_study_runner_records_reproducibility_metadata(tmp_path: Path) -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(60, 3))
    y = np.column_stack((x[:, 0] + x[:, 1], x[:, 0] - x[:, 2]))
    split = tmp_path / "split.npz"
    output = tmp_path / "result.json"
    np.savez(
        split,
        x_train=x[:45],
        y_train=y[:45],
        x_test=x[45:],
        y_test=y[45:],
    )

    exit_code = main(
        [
            str(split),
            "--dataset-id",
            "synthetic-test",
            "--protocol-id",
            "offline-test",
            "--task",
            "regression",
            "--variant",
            "both",
            "--n-estimators",
            "2",
            "--max-bins",
            "8",
            "--min-samples-leaf",
            "3",
            "--max-depth",
            "2",
            "--engine",
            "numpy",
            "--output",
            str(output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert len(result["split_sha256"]) == 64
    assert result["dimensions"] == {
        "features": 3,
        "outputs": 2,
        "test_rows": 15,
        "train_rows": 45,
        "validation_rows": 0,
    }
    assert [entry["variant"] for entry in result["results"]] == ["shared", "independent"]
    assert result["published_references"] == []
    assert result["results"][0]["model_count"] == 1
    assert result["results"][1]["model_count"] == 2
    assert result["results"][0]["resolved_engines"] == ["numpy"]
    assert result["code"]["revision"]
    assert result["metric_config"]["binary_cross_entropy_clip"] == 1e-7
    assert set(result["results"][0]["metrics"]) == {
        "average_relative_rmse",
        "global_rmse",
        "mae",
        "mean_target_r2",
    }


def test_aggregate_requires_every_catalog_run_before_exposing_references() -> None:
    def document(run_id: int) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_id": "rf1",
            "protocol_id": "sketchboost_random_holdout_cv5_2022",
            "run_id": str(run_id),
            "task": "regression",
            "split_sha256": str(run_id) * 64,
            "parameters": {"n_estimators": 10, "random_state": run_id},
            "metric_config": {"metric": "global_rmse"},
            "target_transform": None,
            "results": [
                {
                    "variant": "shared",
                    "metrics": {"global_rmse": float(run_id)},
                    "fit_seconds": 1.0,
                    "predict_seconds": 0.1,
                    "metric_seconds": 0.01,
                    "serialized_model_bytes": 100,
                    "resolved_engines": ["numpy"],
                }
            ],
            "published_references": [{"method": "reference", "comparison_tier": "direct"}],
            "created_at_utc": "2026-10-28T00:00:00+00:00",
            "code": {"revision": "abc", "dirty": False},
            "environment": {"python": "test"},
        }

    documents = [document(run_id) for run_id in range(5)]
    with pytest.raises(ValueError, match="missing run_ids"):
        aggregate_documents(documents[:-1])

    incomplete = aggregate_documents(documents[:-1], allow_incomplete=True)
    assert not incomplete["complete"]
    assert incomplete["published_references"] == []

    complete = aggregate_documents(documents)
    assert complete["complete"]
    assert complete["comparison_ready"]
    assert complete["comparison_eligible_reference_count"] == 1
    assert complete["context_only_reference_count"] == 0
    assert complete["results"][0]["metrics"]["global_rmse"]["mean"] == 2.0
    assert complete["results"][0]["metrics"]["global_rmse"]["std"] == pytest.approx(np.sqrt(2.5))


def test_aggregate_archives_context_only_references_without_enabling_comparison() -> None:
    document = {
        "schema_version": 1,
        "dataset_id": "rf1",
        "protocol_id": "sketchboost_random_holdout_cv5_2022",
        "run_id": "0",
        "task": "regression",
        "split_sha256": "0" * 64,
        "parameters": {"n_estimators": 10, "random_state": 0},
        "metric_config": {"metric": "global_rmse"},
        "target_transform": None,
        "results": [
            {
                "variant": "shared",
                "metrics": {"global_rmse": 1.0},
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "metric_seconds": 0.01,
                "serialized_model_bytes": 100,
                "resolved_engines": ["numpy"],
            }
        ],
        "published_references": [{"method": "test-selected sketch", "comparison_tier": "context_only"}],
        "created_at_utc": "2026-10-28T00:00:00+00:00",
        "code": {"revision": "abc", "dirty": False},
        "environment": {"python": "test"},
    }
    documents = [{**document, "run_id": str(run_id), "split_sha256": str(run_id) * 64} for run_id in range(5)]

    aggregate = aggregate_documents(documents)

    assert aggregate["complete"]
    assert aggregate["published_references"] == document["published_references"]
    assert aggregate["comparison_eligible_reference_count"] == 0
    assert aggregate["context_only_reference_count"] == 1
    assert not aggregate["comparison_ready"]
