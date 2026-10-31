"""A small, lazy OpenML adapter used by the study pipeline.

The optional :mod:`openml` dependency is imported only when a real provider is
constructed.  Pure preparation/reporting code and offline tests therefore do
not need the OpenML client installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ResolvedOpenMLSuite:
    """The immutable part of a suite resolution needed by this pipeline."""

    suite_id: int
    alias: str
    name: str
    status: str | None
    task_ids: tuple[int, ...]


@dataclass
class LoadedOpenMLTask:
    """One OpenML task together with its unmodified row-ordered data."""

    task_id: int
    dataset_id: int
    task_type_id: int
    task_type: str
    target_name: str
    class_labels: tuple[Any, ...] | None
    estimation_procedure: dict[str, Any]
    dataset_name: str
    dataset_version: int | None
    dataset_format: str | None
    dataset_licence: str | None
    dataset_citation: str | None
    dataset_md5_checksum: str | None
    dataset_url: str | None
    original_data_url: str | None
    feature_types: dict[str, str]
    x: Any
    y: Any
    categorical_indicator: tuple[bool, ...]
    attribute_names: tuple[str, ...]
    source_task: Any

    def get_split_dimensions(self) -> tuple[int, int, int]:
        """Return ``(repeats, folds, samples)`` from the task definition."""

        values = tuple(int(value) for value in self.source_task.get_split_dimensions())
        if len(values) != 3 or any(value < 1 for value in values):
            raise ValueError(f"Task {self.task_id} returned invalid split dimensions: {values}.")
        return values[0], values[1], values[2]

    def get_split_indices(
        self,
        *,
        repeat: int,
        fold: int,
        sample: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return official positional train/test indices without reordering."""

        train, test = self.source_task.get_train_test_split_indices(
            repeat=repeat,
            fold=fold,
            sample=sample,
        )
        return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)

    def metadata(self) -> dict[str, Any]:
        """Return selected, JSON-safe task metadata used in suite locks."""

        return {
            "task_id": self.task_id,
            "task_type_id": self.task_type_id,
            "task_type": self.task_type,
            "target_name": self.target_name,
            "class_labels": None
            if self.class_labels is None
            else [str(label) for label in self.class_labels],
            "estimation_procedure": _json_safe(self.estimation_procedure),
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "dataset_format": self.dataset_format,
            "dataset_licence": self.dataset_licence,
            "dataset_citation": self.dataset_citation,
            "dataset_md5_checksum": self.dataset_md5_checksum,
            "dataset_url": self.dataset_url,
            "original_data_url": self.original_data_url,
            "attribute_names": list(self.attribute_names),
            "categorical_indicator": list(self.categorical_indicator),
            "feature_types": dict(sorted(self.feature_types.items())),
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


class RealOpenMLProvider:
    """Load suites and tasks through the optional OpenML Python client."""

    def __init__(self, cache_directory: Path) -> None:
        cache_directory.mkdir(parents=True, exist_ok=True)
        # OpenML configures logging while importing. Point that initialization
        # at the study workspace before import instead of touching ~/.openml.
        os.environ.setdefault("OPENML_CACHE_DIR", str(cache_directory))
        try:
            import openml
        except ImportError as exc:  # pragma: no cover - depends on optional installation
            raise RuntimeError(
                "The OpenML pipeline requires the 'studies' extra: "
                "python -m pip install -e '.[studies]'"
            ) from exc
        openml.config.set_root_cache_directory(str(cache_directory))
        self._openml = openml
        self.client_version = str(openml.__version__)

    def resolve_suite(self, suite_id: int) -> ResolvedOpenMLSuite:
        suite = self._openml.study.get_suite(int(suite_id))
        if suite.tasks is None:
            raise ValueError(f"OpenML suite {suite_id} has no task list.")
        task_ids = tuple(int(task_id) for task_id in suite.tasks)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise ValueError(f"OpenML suite {suite_id} returned an empty or duplicate task list.")
        if suite.id is None:
            raise ValueError(f"OpenML suite {suite_id} has no numeric id.")
        return ResolvedOpenMLSuite(
            suite_id=int(suite.id),
            alias=str(suite.alias or ""),
            name=str(suite.name or suite.alias or suite.id),
            status=None if suite.status is None else str(suite.status),
            task_ids=task_ids,
        )

    def load_task(self, task_id: int) -> LoadedOpenMLTask:
        task: Any = self._openml.tasks.get_task(
            int(task_id),
            download_splits=True,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        dataset: Any = task.get_dataset(
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
            cache_format="pickle",
        )
        target_name = str(task.target_name)
        x, y, categorical, names = dataset.get_data(
            target=target_name,
            include_row_id=False,
            include_ignore_attribute=False,
        )
        if y is None:
            raise ValueError(f"Task {task_id} did not return target {target_name!r}.")
        attribute_names = tuple(str(name) for name in names)
        categorical_indicator = tuple(bool(value) for value in categorical)
        if len(attribute_names) != len(categorical_indicator) or len(attribute_names) != int(x.shape[1]):
            raise ValueError(f"Task {task_id} returned inconsistent feature metadata.")

        feature_types: dict[str, str] = {}
        features = getattr(dataset, "features", None) or {}
        values = features.values() if isinstance(features, dict) else features
        for feature in values:
            name = getattr(feature, "name", None)
            data_type = getattr(feature, "data_type", None)
            if name is not None and data_type is not None:
                feature_types[str(name)] = str(data_type)

        raw_labels = getattr(task, "class_labels", None)
        class_labels = None if raw_labels is None else tuple(raw_labels)
        raw_task_type_id = getattr(task, "task_type_id", 0)
        try:
            task_type_id = int(raw_task_type_id)
        except (TypeError, ValueError):
            task_type_id = int(getattr(raw_task_type_id, "value", 0))
        return LoadedOpenMLTask(
            task_id=int(getattr(task, "task_id", getattr(task, "id", task_id))),
            dataset_id=int(task.dataset_id),
            task_type_id=task_type_id,
            task_type=str(getattr(task, "task_type", raw_task_type_id)),
            target_name=target_name,
            class_labels=class_labels,
            estimation_procedure=dict(getattr(task, "estimation_procedure", {}) or {}),
            dataset_name=str(dataset.name),
            dataset_version=None if dataset.version is None else int(dataset.version),
            dataset_format=None if dataset.format is None else str(dataset.format),
            dataset_licence=None if dataset.licence is None else str(dataset.licence),
            dataset_citation=None if dataset.citation is None else str(dataset.citation),
            dataset_md5_checksum=None
            if dataset.md5_checksum is None
            else str(dataset.md5_checksum),
            dataset_url=None if dataset.url is None else str(dataset.url),
            original_data_url=None
            if dataset.original_data_url is None
            else str(dataset.original_data_url),
            feature_types=feature_types,
            x=x,
            y=y,
            categorical_indicator=categorical_indicator,
            attribute_names=attribute_names,
            source_task=task,
        )


__all__ = ["LoadedOpenMLTask", "RealOpenMLProvider", "ResolvedOpenMLSuite"]
