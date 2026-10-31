from __future__ import annotations

import numpy as np
import pytest

from studies.aggregate import aggregate_documents
from studies.metrics import classification_metrics, regression_metrics


def _document(run_id: str, *, transform: dict[str, list[float]], include_auc: bool = True) -> dict[str, object]:
    metrics = {"global_rmse": float(int(run_id[-1]) + 1)}
    if include_auc:
        metrics["roc_auc"] = 0.75
    return {
        "schema_version": 1,
        "dataset_id": "openml-d1-v1",
        "protocol_id": "openml-task-1",
        "run_id": run_id,
        "task": "regression",
        "split_sha256": run_id * 16,
        "parameters": {"n_estimators": 2, "random_state": 42},
        "metric_config": {"regression_target_scale": "raw_inverse_transformed"},
        "target_transform": transform,
        "results": [
            {
                "variant": "shared",
                "metrics": metrics,
                "fit_seconds": 1.0,
                "predict_seconds": 0.1,
                "metric_seconds": 0.01,
                "serialized_model_bytes": 100,
                "resolved_engines": ["numpy"],
            }
        ],
        "published_references": [],
        "code": {"revision": "abc", "dirty": False},
    }


def test_scalar_binary_metrics_include_accuracy_balanced_accuracy_and_tie_aware_auc() -> None:
    target = np.array([0, 0, 1, 1])
    probability = np.array([0.1, 0.8, 0.8, 0.9])

    metrics = classification_metrics(target, probability)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["roc_auc"] == pytest.approx(0.875)


def test_undefined_optional_metrics_are_omitted() -> None:
    classification = classification_metrics(np.ones(3), np.array([0.7, 0.8, 0.9]))
    regression = regression_metrics(np.ones(4), np.ones(2), np.zeros(2))

    assert "balanced_accuracy" not in classification
    assert "roc_auc" not in classification
    assert "average_relative_rmse" not in regression
    assert regression["global_rmse"] == 1.0


def test_dynamic_aggregation_tracks_missing_runs_transforms_and_metric_coverage() -> None:
    documents = [
        _document("f0", transform={"offset": [1.0], "scale": [2.0]}),
        _document("f1", transform={"offset": [3.0], "scale": [4.0]}, include_auc=False),
    ]

    with pytest.raises(ValueError, match="missing run_ids"):
        aggregate_documents(documents, expected_run_ids=["f0", "f1", "f2"], run_axis="official_fold")

    aggregate = aggregate_documents(
        documents,
        allow_incomplete=True,
        expected_run_ids=["f0", "f1", "f2"],
        run_axis="official_fold",
    )

    assert aggregate["run_axis"] == "official_fold"
    assert aggregate["missing_run_ids"] == ["f2"]
    assert aggregate["target_transform_policy"] == "per_run_train_fitted"
    assert aggregate["target_transforms_by_run"][1]["transform"]["offset"] == [3.0]
    auc = aggregate["results"][0]["metrics"]["roc_auc"]
    assert auc["observed_run_ids"] == ["f0"]
    assert auc["missing_run_ids"] == ["f1"]
