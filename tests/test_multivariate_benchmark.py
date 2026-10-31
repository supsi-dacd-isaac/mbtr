from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from studies.multivariate.benchmark import (
    BenchmarkConfig,
    _headline_reference,
    _log_published_references,
    _log_stage_result,
    aggregate_benchmark,
    benchmark_plan,
    build_parser,
    load_benchmark_manifest,
    run_benchmark,
    write_report,
)
from studies.multivariate.models import (
    SelectedLinearResponseClassifier,
    SelectedLinearResponseRegressor,
)
from studies.multivariate.selection import (
    SelectionPolicy,
    estimate_linear_response_cost,
    fit_response_design,
    importance_subset,
    rank_feature_importances,
    select_response_features,
)


def _selection_kwargs(**overrides: object) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "policy": SelectionPolicy(importance_coverage=1.0, max_features=3),
        "n_samples": 200,
        "n_outputs": 4,
        "n_estimators": 10,
        "max_bins": 16,
        "max_depth": 3,
        "work_budget": 1.0e20,
        "time_budget_seconds": 1.0e12,
        "memory_budget_bytes": 2**40,
    }
    parameters.update(overrides)
    return parameters


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_prepared_split(
    item: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    feature_names: list[str],
) -> None:
    split_path = Path(item["split_path"])
    sidecar_path = Path(item["sidecar_path"])
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, **arrays)  # type: ignore[arg-type]
    sidecar_path.write_text(
        json.dumps(
            {
                "dataset_id": item["dataset_id"],
                "protocol_id": item["protocol_id"],
                "run_id": item["run_id"],
                "split_sha256": _sha256(split_path),
                "feature_names": feature_names,
                "provenance": "deterministic offline test fixture",
            }
        ),
        encoding="utf-8",
    )


def _tiny_config() -> BenchmarkConfig:
    return BenchmarkConfig(
        n_estimators=2,
        learning_rate=0.2,
        max_bins=8,
        min_samples_leaf=4,
        max_depth=2,
        l2_regularization=1.0e-4,
        linear_l2_regularization=1.0e-4,
        random_state=11,
        engine="numpy",
        importance_coverage=0.8,
        linear_max_features=1,
        linear_min_features=1,
        linear_work_budget=1.0e20,
        linear_time_budget_seconds=1.0e6,
        linear_memory_budget_bytes=2**30,
    )


def _assert_validation_stopping_artifact(document: dict[str, Any], *, validation_rows: int) -> None:
    assert document["parameters"]["early_stopping_rounds"] == 50
    assert document["best_iteration"] >= -1
    assert np.isfinite(document["best_score"])
    assert document["n_estimators_evaluated"] == len(document["training_loss"])
    assert document["n_estimators_evaluated"] == len(document["validation_loss"])
    assert document["n_estimators_retained"] == document["n_estimators_fitted"]
    assert document["n_estimators_retained"] == document["best_iteration"] + 1
    assert document["n_estimators_retained"] <= document["n_estimators_evaluated"]
    assert document["stopping"] == {
        "enabled": True,
        "monitor": "validation_objective",
        "patience_rounds": 50,
        "validation_rows": validation_rows,
        "requested_max_trees": 2,
        "evaluated_rounds": document["n_estimators_evaluated"],
        "retained_trees": document["n_estimators_retained"],
        "best_iteration": document["best_iteration"],
        "best_iteration_is_zero_based": True,
        "best_score": document["best_score"],
        "patience_exhausted": False,
        "termination_reason": document["stopping"]["termination_reason"],
    }
    assert document["stopping"]["termination_reason"] in {
        "requested_max_evaluated",
        "training_objective_could_not_improve",
    }


def test_validation_selected_protocol_defaults_and_cli_opt_out(tmp_path: Path) -> None:
    config = BenchmarkConfig()
    assert config.n_estimators == 1000
    assert config.learning_rate == pytest.approx(0.05)
    assert config.early_stopping_rounds == 50

    common = [
        "run",
        "--data-directory",
        str(tmp_path / "prepared"),
        "--output-directory",
        str(tmp_path / "results"),
    ]
    arguments = build_parser().parse_args(common)
    assert arguments.n_estimators == 1000
    assert arguments.learning_rate == pytest.approx(0.05)
    assert arguments.early_stopping_rounds == 50
    assert arguments.jobs == 1
    assert arguments.models is None
    assert arguments.models_all is False
    assert arguments.native_threads == 1

    selected = build_parser().parse_args(
        [
            *common,
            "--model",
            "standard",
            "--model",
            "xgboost",
            "--model",
            "catboost",
            "--native-threads",
            "2",
        ]
    )
    assert selected.models == ["standard", "xgboost", "catboost"]
    assert selected.models_all is False
    assert selected.native_threads == 2

    all_models = build_parser().parse_args([*common, "--models-all"])
    assert all_models.models is None
    assert all_models.models_all is True
    with pytest.raises(SystemExit):
        build_parser().parse_args([*common, "--models-all", "--model", "standard"])
    with pytest.raises(SystemExit):
        build_parser().parse_args([*common, "--native-threads", "0"])

    parallel = build_parser().parse_args([*common, "--jobs", "3"])
    assert parallel.jobs == 3
    with pytest.raises(SystemExit):
        build_parser().parse_args([*common, "--jobs", "0"])

    disabled = build_parser().parse_args([*common, "--disable-early-stopping"])
    assert disabled.early_stopping_rounds is None
    with pytest.raises(SystemExit):
        build_parser().parse_args([*common, "--early-stopping-rounds", "25", "--disable-early-stopping"])
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        replace(config, early_stopping_rounds=0).validate()


def test_live_logs_distinguish_trained_stages_from_published_references(
    caplog: pytest.LogCaptureFixture,
) -> None:
    item = {"dataset_id": "rf1", "run_id": "0", "task": "regression"}
    document = {
        "status": "completed",
        "execution_status": "executed",
        "metrics": {"global_rmse": 0.01},
        "n_estimators_retained": 23,
        "n_estimators_evaluated": 73,
        "fit_seconds": 1.25,
    }
    aggregate = {
        "datasets": [
            {
                "dataset_id": "rf1",
                "task": "regression",
                "published_references": [
                    {
                        "method": "XGBoost",
                        "metric": "global_rmse",
                        "value": 0.02,
                        "direction": "minimize",
                        "comparison_tier": "direct",
                    },
                    {
                        "method": "CatBoost",
                        "metric": "global_rmse",
                        "value": 0.03,
                        "direction": "minimize",
                        "comparison_tier": "protocol_reference",
                    },
                    {
                        "method": "test-selected sketch",
                        "metric": "global_rmse",
                        "value": 0.001,
                        "direction": "minimize",
                        "comparison_tier": "context_only",
                    },
                ],
            }
        ]
    }

    with caplog.at_level(logging.INFO, logger="studies.multivariate.benchmark"):
        _log_stage_result(item, "linear_response", document)
        _log_published_references(aggregate)

    messages = [record.getMessage() for record in caplog.records]
    assert any("MBTR linear: test RMSE=0.01" in message for message in messages)
    assert any("trees retained=23 evaluated=73" in message for message in messages)
    assert any("published XGBoost reference rf1: test RMSE=0.02" in message for message in messages)
    assert any("published CatBoost reference rf1: test RMSE=0.03" in message for message in messages)
    assert not any("test-selected sketch" in message for message in messages)
    assert not any("XGBoost: test" in message for message in messages)


def test_primary_manifest_covers_16_datasets_and_120_protocol_runs(tmp_path: Path) -> None:
    manifest = load_benchmark_manifest()
    entries = manifest["datasets"]
    plan = benchmark_plan(tmp_path)

    assert len(entries) == 16
    assert len({entry["dataset_id"] for entry in entries}) == 16
    assert Counter(entry["task"] for entry in entries) == {
        "regression": 10,
        "multilabel": 6,
    }
    assert Counter(entry["linear_response"] for entry in entries) == {
        "eligible": 16,
    }
    assert len(plan) == 120
    assert Counter(item["task"] for item in plan) == {
        "regression": 90,
        "multilabel": 30,
    }
    assert all(not item["split_exists"] and not item["sidecar_exists"] for item in plan)
    assert len({(item["dataset_id"], item["run_id"]) for item in plan}) == 120


def test_feature_importance_ranking_is_stable_and_zero_gain_columns_are_excluded() -> None:
    ranking, ranked_values = rank_feature_importances([0.2, 0.5, 0.5, 0.0, 0.2])
    selected, _, covered = importance_subset(
        [0.2, 0.5, 0.5, 0.0, 0.2],
        SelectionPolicy(importance_coverage=0.6, max_features=4),
    )

    np.testing.assert_array_equal(ranking, [1, 2, 0, 4])
    np.testing.assert_allclose(ranked_values, [0.5, 0.5, 0.2, 0.2])
    np.testing.assert_array_equal(selected, [1, 2])
    assert covered == pytest.approx(1.0 / 1.4)

    no_signal = select_response_features(
        [0.0, 0.0, 0.0],
        **_selection_kwargs(),
    )
    assert no_signal.status == "skip_no_importance"
    assert no_signal.selected_indices == ()
    assert no_signal.ranked_indices == ()

    insufficient = select_response_features(
        [1.0, 0.0, 0.0],
        **_selection_kwargs(policy=SelectionPolicy(importance_coverage=1.0, max_features=3, min_features=2)),
    )
    assert insufficient.status == "skip_insufficient_importance"
    assert insufficient.selected_indices == ()


def test_linear_guard_rejects_unbounded_depth_and_shrinks_top_k_to_budget() -> None:
    unbounded = select_response_features(
        [0.6, 0.3, 0.1],
        **_selection_kwargs(max_depth=None),
    )
    assert unbounded.status == "skip_unbounded_depth"
    assert unbounded.selected_indices == ()

    common = {
        "n_samples": 200,
        "n_split_features": 3,
        "n_outputs": 4,
        "n_estimators": 10,
        "max_bins": 16,
        "max_depth": 3,
    }
    cost_two = estimate_linear_response_cost(n_response_features=2, **common)
    cost_three = estimate_linear_response_cost(n_response_features=3, **common)
    assert cost_two.work_units is not None and cost_three.work_units is not None
    assert cost_two.work_units < cost_three.work_units

    shrunk = select_response_features(
        [0.6, 0.3, 0.1],
        **_selection_kwargs(
            work_budget=(cost_two.work_units + cost_three.work_units) / 2.0,
        ),
    )
    assert shrunk.status == "run"
    assert shrunk.importance_selected_indices == (0, 1, 2)
    assert shrunk.selected_indices == (0, 1)
    assert shrunk.selected_importance == pytest.approx(0.9)
    assert shrunk.cost.response_features == 2


def test_linear_classifier_cost_guard_accounts_for_per_label_exact_curvature() -> None:
    common = {
        "n_samples": 200,
        "n_split_features": 3,
        "n_response_features": 2,
        "n_estimators": 10,
        "max_bins": 16,
        "max_depth": 3,
    }
    regression = estimate_linear_response_cost(
        n_outputs=4,
        leaf_kind="regression",
        curvature_mode="upper_bound",
        **common,
    )
    upper_bound = estimate_linear_response_cost(
        n_outputs=4,
        leaf_kind="classification",
        curvature_mode="upper_bound",
        **common,
    )
    exact = estimate_linear_response_cost(
        n_outputs=4,
        leaf_kind="classification",
        curvature_mode="exact",
        **common,
    )
    one_label_upper = estimate_linear_response_cost(
        n_outputs=1,
        leaf_kind="classification",
        curvature_mode="upper_bound",
        **common,
    )
    one_label_exact = estimate_linear_response_cost(
        n_outputs=1,
        leaf_kind="classification",
        curvature_mode="exact",
        **common,
    )

    assert upper_bound.work_units == regression.work_units
    assert upper_bound.temporary_bytes == regression.temporary_bytes
    assert exact.work_units is not None and upper_bound.work_units is not None
    assert exact.work_units > upper_bound.work_units
    assert exact.predicted_seconds is not None and upper_bound.predicted_seconds is not None
    assert exact.predicted_seconds > upper_bound.predicted_seconds
    assert exact.temporary_bytes > upper_bound.temporary_bytes
    assert one_label_exact.work_units == one_label_upper.work_units
    assert one_label_exact.temporary_bytes == one_label_upper.temporary_bytes
    assert exact.leaf_kind == "classification"
    assert exact.curvature_mode == "exact"


def test_response_design_statistics_are_fit_only_on_training_rows() -> None:
    x_train = np.asarray(
        [
            [1.0, 50.0, 10.0],
            [2.0, 60.0, 20.0],
            [3.0, 70.0, 30.0],
        ]
    )
    x_test = np.asarray([[1000.0, -5.0, -1000.0]])

    transformed, offset, scale = fit_response_design(
        x_train,
        {"test": x_test},
        (0, 2),
    )
    changed, changed_offset, changed_scale = fit_response_design(
        x_train,
        {"test": x_test * 1.0e9},
        (0, 2),
    )

    np.testing.assert_allclose(offset, [2.0, 20.0])
    np.testing.assert_allclose(scale, np.std(x_train[:, [0, 2]], axis=0))
    np.testing.assert_allclose(np.mean(transformed["train"], axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.std(transformed["train"], axis=0), 1.0)
    np.testing.assert_allclose(offset, changed_offset)
    np.testing.assert_allclose(scale, changed_scale)
    np.testing.assert_allclose(transformed["train"], changed["train"])


def test_selected_linear_response_model_uses_train_scaling_validation_and_pickle() -> None:
    rng = np.random.default_rng(13)
    x_train = rng.normal(size=(48, 4))
    x_validation = rng.normal(loc=50.0, size=(12, 4))
    x_test = rng.normal(size=(9, 4))

    def response(x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                1.7 * x[:, 0] - 0.4 * x[:, 2],
                -0.8 * x[:, 0] + 1.2 * x[:, 2],
            )
        )

    model = SelectedLinearResponseRegressor(
        (0, 2),
        {
            "n_estimators": 2,
            "learning_rate": 0.2,
            "max_bins": 8,
            "min_samples_leaf": 4,
            "max_depth": 2,
            "l2_regularization": 1.0e-4,
            "early_stopping_rounds": None,
            "random_state": 3,
            "engine": "numpy",
        },
    ).fit(
        x_train,
        response(x_train),
        eval_set=(x_validation, response(x_validation)),
    )

    np.testing.assert_allclose(model.design_mean_, np.mean(x_train[:, [0, 2]], axis=0))
    np.testing.assert_allclose(model.design_scale_, np.std(x_train[:, [0, 2]], axis=0))
    assert len(model.linear_model_.evals_result_["validation"]) == model.linear_model_.n_estimators_
    prediction = model.predict(x_test)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(x_test), prediction)
    assert prediction.shape == (len(x_test), 2)
    assert restored.selected_indices_ == (0, 2)


def test_selected_linear_response_classifier_uses_train_scaling_validation_and_pickle() -> None:
    rng = np.random.default_rng(17)
    x_train = rng.normal(size=(64, 4))
    x_validation = rng.normal(loc=30.0, size=(16, 4))
    x_test = rng.normal(size=(12, 4))

    def labels(x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            (
                x[:, 0] - 0.5 * x[:, 2] > 0.0,
                -0.7 * x[:, 0] + x[:, 2] > 0.0,
            )
        ).astype(int)

    model = SelectedLinearResponseClassifier(
        (0, 2),
        {
            "n_estimators": 3,
            "learning_rate": 0.3,
            "max_bins": 8,
            "min_samples_leaf": 4,
            "max_depth": 2,
            "l2_regularization": 1.0e-4,
            "early_stopping_rounds": None,
            "random_state": 5,
            "engine": "numpy",
            "decision_threshold": 0.4,
        },
        linear_response_curvature="upper_bound",
    ).fit(
        x_train,
        labels(x_train),
        eval_set=(x_validation, labels(x_validation)),
    )

    np.testing.assert_allclose(model.design_mean_, np.mean(x_train[:, [0, 2]], axis=0))
    np.testing.assert_allclose(model.design_scale_, np.std(x_train[:, [0, 2]], axis=0))
    assert len(model.linear_model_.evals_result_["validation"]) == model.linear_model_.n_estimators_
    assert model.linear_model_.linear_response_ is True
    probability = model.predict_proba(x_test)
    prediction = model.predict(x_test)
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict_proba(x_test), probability)
    np.testing.assert_array_equal(restored.predict(x_test), prediction)
    assert probability.shape == (len(x_test), 2)
    assert prediction.shape == (len(x_test), 2)
    assert restored.selected_indices_ == (0, 2)


def test_parallel_runner_executes_independent_work_units(tmp_path: Path) -> None:
    data_directory = tmp_path / "prepared"
    output_directory = tmp_path / "parallel-results"
    items = benchmark_plan(data_directory, ["mocr_households"])[:2]
    for run_index, item in enumerate(items):
        rng = np.random.default_rng(100 + run_index)
        x_train = rng.normal(size=(48, 3))
        x_validation = rng.normal(size=(16, 3))
        x_test = rng.normal(size=(12, 3))

        def response(x: np.ndarray) -> np.ndarray:
            return np.column_stack((1.5 * x[:, 0] - 0.2 * x[:, 1], -0.7 * x[:, 0] + x[:, 2]))

        _write_prepared_split(
            item,
            {
                "x_train": x_train,
                "y_train": response(x_train),
                "x_validation": x_validation,
                "y_validation": response(x_validation),
                "x_test": x_test,
                "y_test": response(x_test),
            },
            feature_names=["signal", "secondary", "third"],
        )

    summary = run_benchmark(
        data_directory,
        output_directory,
        config=_tiny_config(),
        dataset_ids=["mocr_households"],
        allow_missing=True,
        jobs=2,
    )

    assert summary["jobs_requested"] == 2
    assert summary["parallel_workers"] == 2
    assert summary["executed_work_units"] == 2
    assert summary["counts"]["standard:completed"] == 2
    assert summary["counts"]["linear_response:completed"] == 2
    assert summary["failures"] == []
    for item in items:
        run_directory = output_directory / "results" / "mocr_households" / item["run_id"]
        assert (run_directory / "standard.json").is_file()
        assert (run_directory / "linear_response.json").is_file()

    resumed = run_benchmark(
        data_directory,
        output_directory,
        config=_tiny_config(),
        dataset_ids=["mocr_households"],
        allow_missing=True,
        jobs=1,
    )
    assert resumed["jobs_requested"] == 1
    assert resumed["parallel_workers"] == 1
    assert resumed["counts"]["execution:reused"] == 4

    with pytest.raises(ValueError, match="jobs must be a positive integer"):
        run_benchmark(data_directory, tmp_path / "invalid", config=_tiny_config(), jobs=0)


def test_regression_runner_writes_paired_artifacts_report_and_resumes(tmp_path: Path) -> None:
    data_directory = tmp_path / "prepared"
    output_directory = tmp_path / "results"
    item = benchmark_plan(data_directory, ["mocr_households"])[0]
    x_train = np.column_stack(
        (
            np.linspace(-3.0, 3.0, 48),
            np.tile([-1.0, 1.0, 0.0], 16),
            np.linspace(1.0, -1.0, 48) ** 2,
        )
    )
    x_validation = np.column_stack(
        (
            np.linspace(-2.5, 2.5, 12),
            np.full(12, 1.0e6),
            np.linspace(0.0, 1.0, 12),
        )
    )
    x_test = np.column_stack(
        (
            np.linspace(-2.75, 2.75, 16),
            np.full(16, -1.0e6),
            np.linspace(1.0, 0.0, 16),
        )
    )

    def response(x: np.ndarray) -> np.ndarray:
        return np.column_stack((1.5 * x[:, 0] + 0.2, -0.8 * x[:, 0] - 0.4))

    target_offset = np.asarray([3.0, -2.0])
    target_scale = np.asarray([2.0, 4.0])

    def stored_response(x: np.ndarray) -> np.ndarray:
        return (response(x) - target_offset) / target_scale

    _write_prepared_split(
        item,
        {
            "x_train": x_train,
            "y_train": stored_response(x_train),
            "x_validation": x_validation,
            "y_validation": stored_response(x_validation),
            "x_test": x_test,
            "y_test": stored_response(x_test),
            "target_offset": target_offset,
            "target_scale": target_scale,
        },
        feature_names=["signal", "validation_only_outlier", "nuisance"],
    )

    first = run_benchmark(
        data_directory,
        output_directory,
        config=_tiny_config(),
        dataset_ids=["mocr_households"],
        allow_missing=True,
    )
    second = run_benchmark(
        data_directory,
        output_directory,
        config=_tiny_config(),
        dataset_ids=["mocr_households"],
        allow_missing=True,
    )
    changed_linear = run_benchmark(
        data_directory,
        output_directory,
        config=replace(_tiny_config(), linear_l2_regularization=0.25),
        dataset_ids=["mocr_households"],
        allow_missing=True,
        replace=True,
    )

    assert first["counts"]["standard:completed"] == 1
    assert first["counts"]["linear_response:completed"] == 1
    assert first["counts"]["execution:executed"] == 2
    assert second["counts"]["execution:reused"] == 2
    assert changed_linear["counts"]["execution:reused"] == 1
    assert changed_linear["counts"]["execution:executed"] == 1
    run_directory = output_directory / "results" / "mocr_households" / item["run_id"]
    standard = json.loads((run_directory / "standard.json").read_text(encoding="utf-8"))
    linear = json.loads((run_directory / "linear_response.json").read_text(encoding="utf-8"))
    assert standard["status"] == "completed"
    assert linear["status"] == "completed"
    _assert_validation_stopping_artifact(standard, validation_rows=len(x_validation))
    _assert_validation_stopping_artifact(linear, validation_rows=len(x_validation))
    assert linear["standard_fingerprint"] == standard["fingerprint"]
    assert linear["selection"]["uses_test_data"] is False
    assert linear["selection"]["selected_indices"] == [0]
    assert linear["selection"]["selected_names"] == ["signal"]
    assert linear["selection"]["importance_selected_names"] == ["signal"]
    assert linear["design_mean"] == pytest.approx([float(np.mean(x_train[:, 0]))])
    assert linear["design_scale"] == pytest.approx([float(np.std(x_train[:, 0]))])
    assert "global_rmse" in standard["metrics"]
    assert "global_rmse" in linear["metrics"]
    assert "global_rmse" in standard["stored_scale_metrics"]
    assert standard["metric_target_scale"] == "raw_inverse_transformed"

    aggregate = aggregate_benchmark(output_directory, dataset_ids=["mocr_households"])
    dataset = aggregate["datasets"][0]
    assert dataset["paired_comparison"]["paired_run_ids"] == [item["run_id"]]
    comparison = dataset["paired_comparison"]["metrics"]["global_rmse"]
    assert comparison["standard"]["count"] == 1
    assert comparison["linear_response"]["count"] == 1
    assert comparison["wins_ties_losses"]["wins"] in {0, 1}
    assert dataset["paired_stored_scale_comparison"]["metrics"]["global_rmse"]["standard"]["count"] == 1
    assert "equal_dataset_global_rmse_ratio" not in aggregate["suite"]
    paths = write_report(output_directory, aggregate)
    assert all(path.is_file() for path in paths.values())
    report = paths["markdown"].read_text(encoding="utf-8")
    assert "mocr_households" in report
    assert "global RMSE" in report

    lock = json.loads((output_directory / "benchmark-lock.json").read_text(encoding="utf-8"))
    assert lock["environment"]["python"]
    assert lock["environment"]["numpy"]
    assert "VECLIB_MAXIMUM_THREADS" in lock["environment"]["threads"]
    assert lock["standard_configuration_fingerprint"]
    assert lock["linear_configuration_fingerprint"]

    linear["standard_fingerprint"] = "corrupted-dependency"
    (run_directory / "linear_response.json").write_text(json.dumps(linear), encoding="utf-8")
    corrupted = aggregate_benchmark(output_directory, dataset_ids=["mocr_households"])
    assert corrupted["datasets"][0]["linear_response_statuses"]["stale"] == 1
    assert corrupted["datasets"][0]["paired_comparison"]["paired_run_ids"] == []


def test_multilabel_runner_writes_paired_linear_logistic_artifacts(tmp_path: Path) -> None:
    data_directory = tmp_path / "prepared"
    output_directory = tmp_path / "results"
    item = benchmark_plan(data_directory, ["emotions"])[0]
    rng = np.random.default_rng(29)
    x_train = rng.normal(size=(64, 3))
    x_validation = rng.normal(size=(16, 3))
    x_test = rng.normal(size=(20, 3))

    def labels(x: np.ndarray) -> np.ndarray:
        return np.column_stack((x[:, 0] > 0.0, x[:, 1] + 0.3 * x[:, 0] > 0.0)).astype(int)

    _write_prepared_split(
        item,
        {
            "x_train": x_train,
            "y_train": labels(x_train),
            "x_validation": x_validation,
            "y_validation": labels(x_validation),
            "x_test": x_test,
            "y_test": labels(x_test),
        },
        feature_names=["label_0_signal", "label_1_signal", "noise"],
    )

    summary = run_benchmark(
        data_directory,
        output_directory,
        config=_tiny_config(),
        dataset_ids=["emotions"],
        allow_missing=True,
    )

    assert summary["counts"]["standard:completed"] == 1
    assert summary["counts"]["linear_response:completed"] == 1
    run_directory = output_directory / "results" / "emotions" / item["run_id"]
    standard = json.loads((run_directory / "standard.json").read_text(encoding="utf-8"))
    linear = json.loads((run_directory / "linear_response.json").read_text(encoding="utf-8"))
    assert standard["status"] == "completed"
    _assert_validation_stopping_artifact(standard, validation_rows=len(x_validation))
    assert "binary_cross_entropy" in standard["metrics"]
    assert standard["parameters"]["decision_threshold"] == pytest.approx(0.5)
    assert linear["status"] == "completed"
    _assert_validation_stopping_artifact(linear, validation_rows=len(x_validation))
    assert linear["standard_fingerprint"] == standard["fingerprint"]
    assert "binary_cross_entropy" in linear["metrics"]
    assert linear["parameters"]["decision_threshold"] == pytest.approx(0.5)
    assert linear["parameters"]["linear_response_curvature"] == "upper_bound"
    assert linear["parameters"]["regularize_intercept"] is False
    assert linear["selection"]["uses_test_data"] is False
    assert len(linear["selection"]["selected_indices"]) == 1
    assert linear["selection"]["selected_names"][0] in {
        "label_0_signal",
        "label_1_signal",
    }
    assert linear["selection"]["cost"]["leaf_kind"] == "classification"
    assert linear["selection"]["cost"]["curvature_mode"] == "upper_bound"
    assert len(linear["design_mean"]) == 1
    assert len(linear["design_scale"]) == 1

    aggregate = aggregate_benchmark(output_directory, dataset_ids=["emotions"])
    dataset = aggregate["datasets"][0]
    assert dataset["paired_comparison"]["paired_run_ids"] == [item["run_id"]]
    comparison = dataset["paired_comparison"]["metrics"]["binary_cross_entropy"]
    assert comparison["standard"]["count"] == 1
    assert comparison["linear_response"]["count"] == 1
    assert comparison["wins_ties_losses"]["wins"] in {0, 1}
    assert dataset["standard_complete"] is False
    assert dataset["linear_response_complete"] is False
    paths = write_report(output_directory, aggregate)
    csv_text = paths["csv"].read_text(encoding="utf-8")
    report = paths["markdown"].read_text(encoding="utf-8")
    assert "linear_binary_cross_entropy" in csv_text
    assert "emotions" in report
    assert "binary cross-entropy" in report

    exact_summary = run_benchmark(
        data_directory,
        output_directory,
        config=replace(_tiny_config(), linear_response_curvature="exact"),
        dataset_ids=["emotions"],
        allow_missing=True,
        replace=True,
    )
    assert exact_summary["counts"]["execution:reused"] == 1
    assert exact_summary["counts"]["execution:executed"] == 1
    exact = json.loads((run_directory / "linear_response.json").read_text(encoding="utf-8"))
    assert exact["status"] == "completed"
    assert exact["parameters"]["linear_response_curvature"] == "exact"
    assert exact["selection"]["cost"]["curvature_mode"] == "exact"


def test_benchmark_rejects_unknown_linear_classifier_curvature() -> None:
    with pytest.raises(ValueError, match="linear_response_curvature"):
        replace(_tiny_config(), linear_response_curvature="diagonal").validate()


@pytest.mark.parametrize("value", [0, True])
def test_benchmark_rejects_invalid_output_sketch_size(value: object) -> None:
    with pytest.raises(ValueError, match="output_sketch_size"):
        replace(_tiny_config(), output_sketch_size=value).validate()  # type: ignore[arg-type]


def test_validation_early_stopping_requires_a_prepared_validation_fold(tmp_path: Path) -> None:
    data_directory = tmp_path / "prepared"
    item = benchmark_plan(data_directory, ["emotions"])[0]
    rng = np.random.default_rng(41)
    x_train = rng.normal(size=(48, 3))
    x_test = rng.normal(size=(16, 3))

    def labels(x: np.ndarray) -> np.ndarray:
        return np.column_stack((x[:, 0] > 0.0, x[:, 1] - x[:, 2] > 0.0)).astype(int)

    _write_prepared_split(
        item,
        {
            "x_train": x_train,
            "y_train": labels(x_train),
            "x_test": x_test,
            "y_test": labels(x_test),
        },
        feature_names=["first", "second", "third"],
    )

    with pytest.raises(ValueError, match="nested validation fold"):
        run_benchmark(
            data_directory,
            tmp_path / "strict-results",
            config=_tiny_config(),
            dataset_ids=["emotions"],
            allow_missing=True,
            fail_fast=True,
        )

    fixed_output = tmp_path / "fixed-results"
    fixed = run_benchmark(
        data_directory,
        fixed_output,
        config=replace(_tiny_config(), early_stopping_rounds=None),
        dataset_ids=["emotions"],
        allow_missing=True,
    )
    assert fixed["counts"]["standard:completed"] == 1
    standard = json.loads(
        (fixed_output / "results" / "emotions" / item["run_id"] / "standard.json").read_text(encoding="utf-8")
    )
    assert standard["parameters"]["early_stopping_rounds"] is None
    assert standard["validation_loss"] == []
    assert standard["stopping"]["enabled"] is False
    assert standard["stopping"]["monitor"] is None
    assert standard["stopping"]["validation_rows"] == 0


def test_stage_artifact_records_a_validation_selected_base_model(tmp_path: Path) -> None:
    data_directory = tmp_path / "prepared"
    output_directory = tmp_path / "results"
    item = benchmark_plan(data_directory, ["mocr_households"])[0]
    x_train = np.linspace(-2.0, 2.0, 48)[:, None]
    x_validation = np.linspace(-1.8, 1.8, 16)[:, None]
    x_test = np.linspace(-1.9, 1.9, 12)[:, None]

    def aligned(x: np.ndarray) -> np.ndarray:
        return np.column_stack((x[:, 0], 2.0 * x[:, 0]))

    _write_prepared_split(
        item,
        {
            "x_train": x_train,
            "y_train": aligned(x_train),
            "x_validation": x_validation,
            "y_validation": -aligned(x_validation),
            "x_test": x_test,
            "y_test": -aligned(x_test),
        },
        feature_names=["opposed_validation_signal"],
    )

    summary = run_benchmark(
        data_directory,
        output_directory,
        config=replace(_tiny_config(), n_estimators=1, early_stopping_rounds=1),
        dataset_ids=["mocr_households"],
        allow_missing=True,
    )

    assert summary["counts"]["standard:completed"] == 1
    standard = json.loads(
        (output_directory / "results" / "mocr_households" / item["run_id"] / "standard.json").read_text(
            encoding="utf-8"
        )
    )
    assert standard["best_iteration"] == -1
    assert standard["n_estimators_evaluated"] == 1
    assert standard["n_estimators_retained"] == 0
    assert standard["n_estimators_fitted"] == 0
    assert standard["feature_importances"] == [0.0]
    assert standard["stopping"]["retained_trees"] == 0
    assert standard["stopping"]["requested_max_trees"] == 1
    assert standard["stopping"]["patience_exhausted"] is True
    assert standard["stopping"]["termination_reason"] == "validation_patience_exhausted"
    assert standard["stopping"]["best_score"] == pytest.approx(standard["best_score"])


def test_classification_headline_reference_falls_back_to_hamming_loss() -> None:
    dataset = {
        "task": "multilabel",
        "published_references": [
            {
                "method": "baseline",
                "metric": "hamming_loss",
                "value": 0.2,
                "direction": "minimize",
                "comparison_tier": "protocol_reference",
            },
            {
                "method": "better",
                "metric": "hamming_loss",
                "value": 0.1,
                "direction": "minimize",
                "comparison_tier": "protocol_reference",
            },
        ],
    }

    reference = _headline_reference(dataset)

    assert reference is not None
    assert reference["method"] == "better"


def test_headline_reference_never_selects_a_context_only_result() -> None:
    dataset = {
        "task": "regression",
        "published_references": [
            {
                "method": "test-selected sketch",
                "metric": "global_rmse",
                "value": 0.01,
                "direction": "minimize",
                "comparison_tier": "context_only",
            },
            {
                "method": "direct baseline",
                "metric": "global_rmse",
                "value": 0.4,
                "direction": "minimize",
                "comparison_tier": "direct",
            },
        ],
    }

    reference = _headline_reference(dataset)

    assert reference is not None
    assert reference["method"] == "direct baseline"


def test_headline_reference_returns_none_for_context_only_results() -> None:
    reference = _headline_reference(
        {
            "task": "regression",
            "published_references": [
                {
                    "method": "test-selected sketch",
                    "metric": "global_rmse",
                    "value": 0.01,
                    "direction": "minimize",
                    "comparison_tier": "context_only",
                }
            ],
        }
    )

    assert reference is None


def test_context_only_preferred_metric_does_not_mask_an_eligible_fallback() -> None:
    dataset = {
        "task": "multilabel",
        "published_references": [
            {
                "method": "context BCE",
                "metric": "binary_cross_entropy",
                "value": 0.01,
                "direction": "minimize",
                "comparison_tier": "context_only",
            },
            {
                "method": "eligible Hamming",
                "metric": "hamming_loss",
                "value": 0.2,
                "direction": "minimize",
                "comparison_tier": "protocol_reference",
            },
        ],
    }

    reference = _headline_reference(dataset)

    assert reference is not None
    assert reference["method"] == "eligible Hamming"
