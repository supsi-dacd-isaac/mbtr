from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from studies.openml.pipeline import (
    SUITES,
    aggregate_suite,
    prepare_suite,
    run_suite,
    write_report,
)
from studies.openml.prepare import PreprocessingConfig, prepare_fold
from studies.openml.provider import LoadedOpenMLTask, ResolvedOpenMLSuite
from studies.openml.report import build_report


class FakeSourceTask:
    def __init__(self, splits: dict[tuple[int, int, int], tuple[list[int], list[int]]]) -> None:
        self.splits = splits

    def get_split_dimensions(self) -> tuple[int, int, int]:
        repeats = max(key[0] for key in self.splits) + 1
        folds = max(key[1] for key in self.splits) + 1
        samples = max(key[2] for key in self.splits) + 1
        return repeats, folds, samples

    def get_train_test_split_indices(
        self,
        *,
        repeat: int,
        fold: int,
        sample: int,
    ) -> tuple[list[int], list[int]]:
        return self.splits[(repeat, fold, sample)]


def loaded_task(
    *,
    task_id: int,
    x: np.ndarray,
    y: np.ndarray,
    task_type_id: int,
    class_labels: tuple[str, ...] | None,
    categorical: tuple[bool, ...],
    splits: dict[tuple[int, int, int], tuple[list[int], list[int]]],
) -> LoadedOpenMLTask:
    return LoadedOpenMLTask(
        task_id=task_id,
        dataset_id=task_id + 100,
        task_type_id=task_type_id,
        task_type="classification" if task_type_id == 1 else "regression",
        target_name="target",
        class_labels=class_labels,
        estimation_procedure={"type": "crossvalidation"},
        dataset_name=f"dataset-{task_id}",
        dataset_version=1,
        dataset_format="ARFF",
        dataset_licence="CC-BY-4.0",
        dataset_citation="Synthetic offline fixture",
        dataset_md5_checksum="fixture",
        dataset_url=f"https://www.openml.org/d/{task_id + 100}",
        original_data_url=None,
        feature_types={},
        x=x,
        y=y,
        categorical_indicator=categorical,
        attribute_names=tuple(f"feature_{index}" for index in range(x.shape[1])),
        source_task=FakeSourceTask(splits),
    )


class FakeProvider:
    client_version = "offline-fixture"

    def __init__(self, suite_id: int, tasks: list[LoadedOpenMLTask]) -> None:
        self.suite_id = suite_id
        self.tasks = {task.task_id: task for task in tasks}

    def resolve_suite(self, suite_id: int) -> ResolvedOpenMLSuite:
        assert suite_id == self.suite_id
        return ResolvedOpenMLSuite(
            suite_id=suite_id,
            alias="offline-suite",
            name="Offline suite",
            status="active",
            task_ids=tuple(self.tasks),
        )

    def load_task(self, task_id: int) -> LoadedOpenMLTask:
        return self.tasks[task_id]


def test_fold_preprocessing_is_fit_on_training_rows_only_and_preserves_indices(tmp_path: Path) -> None:
    x = np.asarray(
        [
            [1.0, "red"],
            [3.0, "red"],
            [np.nan, "blue"],
            [5.0, None],
            [1000.0, "green"],
            [np.nan, "unseen"],
        ],
        dtype=object,
    )
    y = np.asarray(["no", "yes", "no", "yes", "no", "yes"])
    task = loaded_task(
        task_id=17,
        x=x,
        y=y,
        task_type_id=1,
        class_labels=("no", "yes"),
        categorical=(False, True),
        splits={(0, 0, 0): ([0, 1, 2, 3], [4, 5])},
    )
    split = tmp_path / "fold.npz"

    artifact = prepare_fold(
        task,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        task_kind="binary",
        split_path=split,
        run_id="r0-f0-s0",
        repeat=0,
        fold=0,
        sample=0,
        config=PreprocessingConfig(max_categories=8),
    )

    with np.load(split, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["source_train_indices"], [0, 1, 2, 3])
        np.testing.assert_array_equal(archive["source_test_indices"], [4, 5])
        # The second test row is missing numerically and gets the training mean
        # (1 + 3 + 5) / 3, not a statistic involving the test value 1000.
        assert archive["x_test"][1, 0] == pytest.approx(3.0)
        # Neither test-only category has a fitted one-hot column. Unknowns are
        # represented by all zeroes when no training category was truncated.
        np.testing.assert_array_equal(archive["x_test"][:, 1:], 0.0)
        np.testing.assert_array_equal(archive["y_train"], [0, 1, 0, 1])
    metadata = json.loads(artifact.metadata_path.read_text(encoding="utf-8"))
    assert metadata["preprocessing"]["fit_partition"] == "outer_training_fold_only"
    assert metadata["target"]["positive_class"].endswith(":yes")

    reused = prepare_fold(
        task,
        train_indices=[0, 1, 2, 3],
        test_indices=[4, 5],
        task_kind="binary",
        split_path=split,
        run_id="r0-f0-s0",
        repeat=0,
        fold=0,
        sample=0,
        config=PreprocessingConfig(max_categories=8),
    )
    assert reused.reused


def test_offline_pipeline_runs_resumes_and_keeps_partial_official_coverage_visible(tmp_path: Path) -> None:
    x = np.column_stack((np.linspace(-2.0, 2.0, 20), np.arange(20) % 3))
    y = 2.0 * x[:, 0] - 0.25 * x[:, 1]
    splits = {
        (0, 0, 0): (list(range(10, 20)), list(range(0, 10))),
        (0, 1, 0): (list(range(0, 10)), list(range(10, 20))),
    }
    task = loaded_task(
        task_id=9001,
        x=x,
        y=y,
        task_type_id=2,
        class_labels=None,
        categorical=(False, False),
        splits=splits,
    )
    provider = FakeProvider(353, [task])
    workspace = tmp_path / "study"

    lock = prepare_suite(
        provider,
        SUITES["ctr23"],
        workspace,
        max_runs_per_task=1,
        preprocessing=PreprocessingConfig(max_categories=8),
    )
    assert lock["tasks"][0]["official_expected_runs"] == 2
    assert lock["tasks"][0]["selected_runs"] == 1
    assert not lock["tasks"][0]["official_selection_complete"]
    assert len(list((workspace / "indices" / "task-9001").glob("*.npz"))) == 2

    first = run_suite(
        workspace,
        n_estimators=2,
        max_bins=4,
        min_samples_leaf=2,
        max_depth=2,
        engine="numpy",
    )
    second = run_suite(
        workspace,
        n_estimators=2,
        max_bins=4,
        min_samples_leaf=2,
        max_depth=2,
        engine="numpy",
    )
    assert first == {"completed": 1, "reused": 0, "failed": 0}
    assert second == {"completed": 0, "reused": 1, "failed": 0}
    with pytest.raises(ValueError, match="different model or source tree"):
        run_suite(
            workspace,
            n_estimators=3,
            max_bins=4,
            min_samples_leaf=2,
            max_depth=2,
            engine="numpy",
        )

    aggregates = aggregate_suite(workspace)
    assert len(aggregates) == 1
    assert aggregates[0]["complete"]
    assert not aggregates[0]["openml"]["official_complete"]
    assert aggregates[0]["hierarchy"]["variants"]["shared"]["metrics"]["global_rmse"][
        "task_mean"
    ] >= 0.0
    report = write_report(workspace)
    assert report["tasks_with_complete_selection"] == 1
    assert report["tasks_with_complete_official_resampling"] == 0


def test_report_joins_references_by_numeric_task_id_and_never_averages_raw_rmse() -> None:
    lock: dict[str, Any] = {
        "suite": {"id": 353, "alias": "OpenML-CTR23"},
        "selection": {},
        "tasks": [
            {
                "task_id": 1,
                "dataset_id": 101,
                "dataset_name": "same-name-is-not-used-for-join",
                "task_kind": "regression",
                "status": "prepared",
                "official_expected_runs": 10,
            },
            {
                "task_id": 2,
                "dataset_id": 102,
                "dataset_name": "another",
                "task_kind": "regression",
                "status": "prepared",
                "official_expected_runs": 100,
            },
        ],
    }

    def aggregate(task_id: int, rmse: float) -> dict[str, Any]:
        return {
            "openml": {"task_id": task_id, "official_complete": True},
            "complete": True,
            "observed_run_ids": ["fixture"],
            "hierarchy": {
                "variants": {
                    "shared": {"metrics": {"global_rmse": {"task_mean": rmse}}},
                }
            },
        }

    references = [
        {"openml_task_id": 1, "method": "Baseline", "value": 1.0},
        {"openml_task_id": 2, "method": "Baseline", "value": 100.0},
    ]
    report = build_report(lock, [aggregate(1, 2.0), aggregate(2, 50.0)], reference_rows=references)

    comparison = report["suite_summary"]["reference_comparisons"]["Baseline"]
    assert comparison["tasks"] == 2
    assert comparison["mbtr_wins"] == 1
    assert comparison["mbtr_losses"] == 1
    assert comparison["geometric_mean_rmse_ratio"] == pytest.approx(1.0)
    assert "equal_task_mean_metrics" not in report["suite_summary"]
