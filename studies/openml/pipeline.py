"""Prepare, run, aggregate, and report reproducible OpenML MBTR studies."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from studies.aggregate import aggregate_documents
from studies.fetch_scores import fetch_score_source
from studies.run import main as run_study

from .prepare import (
    PreprocessingConfig,
    canonical_sha256,
    file_sha256,
    prepare_fold,
    split_indices_sha256,
    validate_split_indices,
)
from .provider import LoadedOpenMLTask, RealOpenMLProvider, ResolvedOpenMLSuite
from .report import build_report, report_csv, report_markdown


@dataclass(frozen=True)
class SuiteSpec:
    key: str
    suite_id: int
    alias: str
    task_kind: str
    reference_source: str | None = None


SUITES = {
    "cc18": SuiteSpec("cc18", 99, "OpenML-CC18", "binary"),
    "ctr23": SuiteSpec("ctr23", 353, "OpenML-CTR23", "regression", "openml_ctr23_results"),
}

REPOSITORY = Path(__file__).resolve().parents[2]


class OpenMLProvider(Protocol):
    client_version: str

    def resolve_suite(self, suite_id: int) -> ResolvedOpenMLSuite: ...

    def load_task(self, task_id: int) -> LoadedOpenMLTask: ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative(path: Path, workspace: Path) -> str:
    return str(path.relative_to(workspace))


def _workspace_path(workspace: Path, relative: str) -> Path:
    path = workspace / relative
    resolved_workspace = workspace.resolve()
    resolved_path = path.resolve()
    if resolved_path != resolved_workspace and resolved_workspace not in resolved_path.parents:
        raise ValueError(f"Artifact path escapes the workspace: {relative!r}.")
    return path


def _source_fingerprint(paths: list[Path]) -> str:
    records = {
        str(path.relative_to(REPOSITORY)): file_sha256(path)
        for path in sorted(paths)
        if path.is_file()
    }
    if not records:
        raise ValueError("Could not find source files for the study fingerprint.")
    return canonical_sha256(records)


def _preparation_code_fingerprint() -> str:
    return _source_fingerprint(
        [
            Path(__file__),
            Path(__file__).with_name("prepare.py"),
            Path(__file__).with_name("provider.py"),
        ]
    )


def _model_code_fingerprint() -> str:
    paths = list((REPOSITORY / "mbtr").glob("*.py"))
    paths.extend(
        [
            REPOSITORY / "studies" / "adapters.py",
            REPOSITORY / "studies" / "metrics.py",
            REPOSITORY / "studies" / "run.py",
        ]
    )
    return _source_fingerprint(paths)


def _run_id(repeat: int, fold: int, sample: int) -> str:
    return f"r{repeat}-f{fold}-s{sample}"


def _enumerate_splits(dimensions: tuple[int, int, int]) -> list[dict[str, Any]]:
    repeats, folds, samples = dimensions
    return [
        {
            "run_id": _run_id(repeat, fold, sample),
            "repeat": repeat,
            "fold": fold,
            "sample": sample,
        }
        for repeat in range(repeats)
        for fold in range(folds)
        for sample in range(samples)
    ]


def _write_index_artifact(
    path: Path,
    train: np.ndarray,
    test: np.ndarray,
    *,
    force: bool,
) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            matches = (
                set(archive.files) == {"train_indices", "test_indices"}
                and np.array_equal(archive["train_indices"], train)
                and np.array_equal(archive["test_indices"], test)
            )
        if matches:
            return
        if not force:
            raise ValueError(f"Official index artifact changed at {path}; pass --force to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, train_indices=train, test_indices=test)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _select_task_ids(
    suite_task_ids: tuple[int, ...],
    requested: list[int] | None,
    max_tasks: int | None,
) -> list[int]:
    if requested:
        unknown = sorted(set(requested) - set(suite_task_ids))
        if unknown:
            raise ValueError(f"Requested task ids are not in the resolved suite: {unknown}.")
        selected = [task_id for task_id in suite_task_ids if task_id in set(requested)]
    else:
        selected = list(suite_task_ids)
    if max_tasks is not None:
        if max_tasks < 1:
            raise ValueError("max_tasks must be positive.")
        selected = selected[:max_tasks]
    return selected


def _task_supported(task: LoadedOpenMLTask, spec: SuiteSpec) -> tuple[bool, str | None]:
    if spec.task_kind == "regression":
        if task.task_type_id != 2:
            return False, f"task type id {task.task_type_id} is not regression"
        return True, None
    if task.task_type_id != 1:
        return False, f"task type id {task.task_type_id} is not classification"
    if task.class_labels is not None and len(task.class_labels) != 2:
        return False, f"declares {len(task.class_labels)} classes; MBTR currently supports binary tasks here"
    return True, None


def prepare_suite(
    provider: OpenMLProvider,
    spec: SuiteSpec,
    workspace: Path,
    *,
    task_ids: list[int] | None = None,
    max_tasks: int | None = None,
    max_runs_per_task: int | None = None,
    preprocessing: PreprocessingConfig | None = None,
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Resolve a suite, persist official indices, and prepare selected folds."""

    workspace.mkdir(parents=True, exist_ok=True)
    settings = preprocessing or PreprocessingConfig()
    settings.validate()
    if max_runs_per_task is not None and max_runs_per_task < 1:
        raise ValueError("max_runs_per_task must be positive.")
    resolved = provider.resolve_suite(spec.suite_id)
    if resolved.suite_id != spec.suite_id:
        raise ValueError(f"Requested suite {spec.suite_id}, but OpenML returned {resolved.suite_id}.")
    selected_ids = _select_task_ids(resolved.task_ids, task_ids, max_tasks)
    preparation_code_fingerprint = _preparation_code_fingerprint()
    preparation_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "openml": provider.client_version,
        "pandas": _package_version("pandas"),
    }
    resolution_fingerprint = canonical_sha256(
        {
            "suite_id": spec.suite_id,
            "suite_task_ids": list(resolved.task_ids),
            "selected_task_ids": selected_ids,
            "max_runs_per_task": max_runs_per_task,
            "preprocessing": asdict(settings),
            "preparation_code_fingerprint": preparation_code_fingerprint,
            "preparation_environment": preparation_environment,
        }
    )
    lock_path = workspace / "suite-lock.json"
    if lock_path.exists() and not force:
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        if previous.get("resolution_fingerprint") != resolution_fingerprint:
            raise ValueError(
                f"{lock_path} describes a different selection or preprocessing policy; "
                "use another workspace or pass --force."
            )

    task_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task_id in selected_ids:
        try:
            task = provider.load_task(task_id)
            supported, reason = _task_supported(task, spec)
            metadata = task.metadata()
            base_record = {
                "task_id": task.task_id,
                "dataset_id": task.dataset_id,
                "dataset_name": task.dataset_name,
                "dataset_version": task.dataset_version,
                "target_name": task.target_name,
                "task_kind": spec.task_kind,
                "metadata_sha256": canonical_sha256(metadata),
                "metadata": metadata,
            }
            if not supported:
                task_records.append({**base_record, "status": "skipped", "reason": reason})
                continue
            dimensions = task.get_split_dimensions()
            units = _enumerate_splits(dimensions)
            selected_count = len(units) if max_runs_per_task is None else min(max_runs_per_task, len(units))
            work_units: list[dict[str, Any]] = []
            for position, unit in enumerate(units):
                train, test = task.get_split_indices(
                    repeat=unit["repeat"],
                    fold=unit["fold"],
                    sample=unit["sample"],
                )
                train, test = validate_split_indices(train, test, n_rows=len(task.x))
                index_hash = split_indices_sha256(train, test)
                index_path = workspace / "indices" / f"task-{task_id}" / f"{unit['run_id']}.npz"
                _write_index_artifact(index_path, train, test, force=force)
                selected = position < selected_count
                record = {
                    **unit,
                    "selected": selected,
                    "indices_path": _relative(index_path, workspace),
                    "indices_sha256": index_hash,
                    "train_rows": len(train),
                    "test_rows": len(test),
                }
                if selected:
                    split_path = workspace / "prepared" / f"task-{task_id}" / f"{unit['run_id']}.npz"
                    artifact = prepare_fold(
                        task,
                        train_indices=train,
                        test_indices=test,
                        task_kind=spec.task_kind,
                        split_path=split_path,
                        run_id=unit["run_id"],
                        repeat=unit["repeat"],
                        fold=unit["fold"],
                        sample=unit["sample"],
                        config=settings,
                        force=force,
                    )
                    record.update(
                        {
                            "prepared_path": _relative(artifact.split_path, workspace),
                            "prepared_metadata_path": _relative(artifact.metadata_path, workspace),
                            "prepared_sha256": artifact.split_sha256,
                            "prepared_fingerprint": artifact.fingerprint,
                            "reused": artifact.reused,
                        }
                    )
                work_units.append(record)
            task_records.append(
                {
                    **base_record,
                    "status": "prepared",
                    "split_dimensions": {
                        "repeats": dimensions[0],
                        "folds": dimensions[1],
                        "samples": dimensions[2],
                    },
                    "official_expected_runs": len(units),
                    "selected_runs": selected_count,
                    "official_selection_complete": selected_count == len(units),
                    "work_units": work_units,
                }
            )
        except Exception as exc:
            failure = {
                "stage": "prepare",
                "task_id": task_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            task_records.append({"task_id": task_id, "status": "failed", "reason": str(exc)})
            if fail_fast:
                raise

    lock = {
        "schema_version": 1,
        "artifact_type": "openml_suite_lock",
        "created_at_utc": _utc_now(),
        "resolution_fingerprint": resolution_fingerprint,
        "suite": {
            "key": spec.key,
            "id": spec.suite_id,
            "alias": spec.alias,
            "openml_alias": resolved.alias,
            "name": resolved.name,
            "status": resolved.status,
            "resolved_task_ids": list(resolved.task_ids),
        },
        "selection": {
            "task_ids": selected_ids,
            "max_tasks": max_tasks,
            "max_runs_per_task": max_runs_per_task,
            "is_full_suite": selected_ids == list(resolved.task_ids),
        },
        "preprocessing": asdict(settings),
        "preparation_code_fingerprint": preparation_code_fingerprint,
        "provider": {
            "name": type(provider).__name__,
            **preparation_environment,
        },
        "tasks": task_records,
        "failures": failures,
    }
    _atomic_json(lock_path, lock)
    return lock


def load_lock(workspace: Path) -> dict[str, Any]:
    path = workspace / "suite-lock.json"
    if not path.is_file():
        raise FileNotFoundError(f"No suite lock at {path}; run the prepare stage first.")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("artifact_type") != "openml_suite_lock":
        raise ValueError(f"Unsupported suite lock at {path}.")
    return lock


def _result_matches(
    result: dict[str, Any],
    *,
    run_fingerprint: str,
    split_sha256: str,
) -> bool:
    return (
        result.get("pipeline", {}).get("run_fingerprint") == run_fingerprint
        and result.get("split_sha256") == split_sha256
    )


def _unit_run_fingerprint(
    *,
    run_plan_fingerprint: str,
    prepared_fingerprint: str,
    task_id: int,
    run_id: str,
) -> str:
    return canonical_sha256(
        {
            "run_plan_fingerprint": run_plan_fingerprint,
            "prepared_fingerprint": prepared_fingerprint,
            "task_id": task_id,
            "run_id": run_id,
        }
    )


def run_suite(
    workspace: Path,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_bins: int = 64,
    min_samples_leaf: int = 20,
    max_depth: int | None = None,
    l2_regularization: float = 0.1,
    random_state: int = 42,
    engine: str = "auto",
    force: bool = False,
    fail_fast: bool = False,
) -> dict[str, int]:
    """Run scalar MBTR on every selected prepared fold, resuming exact matches."""

    lock = load_lock(workspace)
    model_config = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_bins": max_bins,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": max_depth,
        "l2_regularization": l2_regularization,
        "random_state": random_state,
        "engine": engine,
        "variant": "shared",
    }
    model_code_fingerprint = _model_code_fingerprint()
    model_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "numba": _package_version("numba"),
        "mbtr": _package_version("mbtr"),
    }
    run_plan_fingerprint = canonical_sha256(
        {
            "suite_resolution_fingerprint": lock["resolution_fingerprint"],
            "model": model_config,
            "model_code_fingerprint": model_code_fingerprint,
            "model_environment": model_environment,
        }
    )
    run_lock_path = workspace / "run-lock.json"
    if run_lock_path.exists() and not force:
        previous_run_lock = json.loads(run_lock_path.read_text(encoding="utf-8"))
        if previous_run_lock.get("run_plan_fingerprint") != run_plan_fingerprint:
            raise ValueError(
                f"{run_lock_path} describes a different model or source tree; use another workspace or pass --force."
            )
    _atomic_json(
        run_lock_path,
        {
            "schema_version": 1,
            "artifact_type": "openml_run_lock",
            "created_at_utc": _utc_now(),
            "suite_resolution_fingerprint": lock["resolution_fingerprint"],
            "model_code_fingerprint": model_code_fingerprint,
            "model_environment": model_environment,
            "model": model_config,
            "run_plan_fingerprint": run_plan_fingerprint,
        },
    )
    counts = {"completed": 0, "reused": 0, "failed": 0}
    failures: list[dict[str, Any]] = []
    suite_id = int(lock["suite"]["id"])
    for task in lock["tasks"]:
        if task.get("status") != "prepared":
            continue
        task_id = int(task["task_id"])
        dataset_id = f"openml-d{task['dataset_id']}-v{task.get('dataset_version') or 'unknown'}"
        protocol_id = f"openml-suite-{suite_id}-task-{task_id}-official-resampling"
        for unit in task["work_units"]:
            if not unit.get("selected"):
                continue
            split_path = _workspace_path(workspace, unit["prepared_path"])
            actual_split_hash = file_sha256(split_path)
            if actual_split_hash != unit["prepared_sha256"]:
                raise ValueError(f"Prepared split checksum changed: {split_path}.")
            run_fingerprint = _unit_run_fingerprint(
                run_plan_fingerprint=run_plan_fingerprint,
                prepared_fingerprint=unit["prepared_fingerprint"],
                task_id=task_id,
                run_id=unit["run_id"],
            )
            result_path = workspace / "results" / f"task-{task_id}" / f"{unit['run_id']}.json"
            if result_path.exists():
                previous = json.loads(result_path.read_text(encoding="utf-8"))
                if _result_matches(
                    previous,
                    run_fingerprint=run_fingerprint,
                    split_sha256=actual_split_hash,
                ):
                    counts["reused"] += 1
                    continue
                if not force:
                    raise ValueError(f"Result {result_path} has a different run fingerprint; pass --force.")
            temporary = result_path.with_name(f".{result_path.name}.{os.getpid()}.tmp")
            arguments = [
                str(split_path),
                "--dataset-id",
                dataset_id,
                "--protocol-id",
                protocol_id,
                "--task",
                task["task_kind"],
                "--variant",
                "shared",
                "--run-id",
                unit["run_id"],
                "--n-estimators",
                str(n_estimators),
                "--learning-rate",
                str(learning_rate),
                "--max-bins",
                str(max_bins),
                "--min-samples-leaf",
                str(min_samples_leaf),
                "--l2-regularization",
                str(l2_regularization),
                "--random-state",
                str(random_state),
                "--engine",
                engine,
                "--output",
                str(temporary),
            ]
            if max_depth is not None:
                arguments.extend(["--max-depth", str(max_depth)])
            try:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                run_study(arguments)
                result = json.loads(temporary.read_text(encoding="utf-8"))
                result["pipeline"] = {
                    "run_fingerprint": run_fingerprint,
                    "openml_task_id": task_id,
                    "repeat": unit["repeat"],
                    "fold": unit["fold"],
                    "sample": unit["sample"],
                    "prepared_fingerprint": unit["prepared_fingerprint"],
                }
                temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                os.replace(temporary, result_path)
                counts["completed"] += 1
            except Exception as exc:
                counts["failed"] += 1
                failures.append(
                    {
                        "stage": "run",
                        "task_id": task_id,
                        "run_id": unit["run_id"],
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                if fail_fast:
                    raise
            finally:
                if temporary.exists():
                    temporary.unlink()
    _atomic_json(workspace / "run-failures.json", {"created_at_utc": _utc_now(), "failures": failures})
    return counts


def _hierarchy(documents: list[dict[str, Any]]) -> dict[str, Any]:
    by_repeat: dict[int, list[dict[str, Any]]] = {}
    for document in documents:
        repeat = int(document.get("pipeline", {}).get("repeat", 0))
        by_repeat.setdefault(repeat, []).append(document)
    variants = sorted({result["variant"] for document in documents for result in document["results"]})
    output: dict[str, Any] = {"policy": "fold_mean_within_repeat_then_equal_repeat_mean", "variants": {}}
    for variant in variants:
        metric_names = sorted(
            {
                name
                for document in documents
                for result in document["results"]
                if result["variant"] == variant
                for name in result["metrics"]
            }
        )
        metrics: dict[str, Any] = {}
        for name in metric_names:
            repeat_values: list[dict[str, Any]] = []
            for repeat, repeat_documents in sorted(by_repeat.items()):
                values = [
                    float(result["metrics"][name])
                    for document in repeat_documents
                    for result in document["results"]
                    if result["variant"] == variant and name in result["metrics"]
                ]
                if values:
                    repeat_values.append(
                        {
                            "repeat": repeat,
                            "fold_values": values,
                            "mean": float(np.mean(values)),
                            "observed_folds": len(values),
                        }
                    )
            means = [record["mean"] for record in repeat_values]
            metrics[name] = {
                "task_mean": None if not means else float(np.mean(means)),
                "repeat_means": repeat_values,
                "observed_repeats": len(means),
            }
        output["variants"][variant] = {"metrics": metrics}
    return output


def aggregate_suite(workspace: Path, *, allow_incomplete: bool = False) -> list[dict[str, Any]]:
    """Aggregate fold results per task while retaining the repeat hierarchy."""

    lock = load_lock(workspace)
    run_lock_path = workspace / "run-lock.json"
    if not run_lock_path.is_file():
        raise FileNotFoundError(f"No run lock at {run_lock_path}; run the run stage first.")
    run_lock = json.loads(run_lock_path.read_text(encoding="utf-8"))
    if run_lock.get("suite_resolution_fingerprint") != lock["resolution_fingerprint"]:
        raise ValueError("The run lock belongs to a different suite preparation; rerun the run stage.")
    run_plan_fingerprint = str(run_lock["run_plan_fingerprint"])
    aggregates: list[dict[str, Any]] = []
    for task in lock["tasks"]:
        if task.get("status") != "prepared":
            continue
        selected = [unit for unit in task["work_units"] if unit.get("selected")]
        documents: list[dict[str, Any]] = []
        for unit in selected:
            result_path = workspace / "results" / f"task-{task['task_id']}" / f"{unit['run_id']}.json"
            if result_path.is_file():
                document = json.loads(result_path.read_text(encoding="utf-8"))
                expected_fingerprint = _unit_run_fingerprint(
                    run_plan_fingerprint=run_plan_fingerprint,
                    prepared_fingerprint=unit["prepared_fingerprint"],
                    task_id=int(task["task_id"]),
                    run_id=str(unit["run_id"]),
                )
                if document.get("pipeline", {}).get("run_fingerprint") == expected_fingerprint:
                    documents.append(document)
        if not documents:
            if allow_incomplete:
                continue
            raise ValueError(f"Task {task['task_id']} has no completed fold results.")
        expected_ids = [str(unit["run_id"]) for unit in selected]
        aggregate = aggregate_documents(
            documents,
            allow_incomplete=allow_incomplete,
            expected_run_ids=expected_ids,
            run_axis="official_openml_resample",
        )
        official_ids = [str(unit["run_id"]) for unit in task["work_units"]]
        aggregate["hierarchy"] = _hierarchy(documents)
        aggregate["openml"] = {
            "task_id": task["task_id"],
            "dataset_id": task["dataset_id"],
            "dataset_name": task["dataset_name"],
            "dataset_version": task.get("dataset_version"),
            "split_dimensions": task["split_dimensions"],
            "official_expected_run_ids": official_ids,
            "official_complete": set(aggregate["observed_run_ids"]) == set(official_ids),
            "selection_complete": aggregate["complete"],
            "suite_resolution_fingerprint": lock["resolution_fingerprint"],
            "run_plan_fingerprint": run_plan_fingerprint,
        }
        output = workspace / "aggregates" / f"task-{task['task_id']}.json"
        _atomic_json(output, aggregate)
        aggregates.append(aggregate)
    return aggregates


def _load_aggregates(workspace: Path, lock: dict[str, Any]) -> list[dict[str, Any]]:
    run_lock_path = workspace / "run-lock.json"
    run_plan_fingerprint = None
    if run_lock_path.is_file():
        run_plan_fingerprint = json.loads(run_lock_path.read_text(encoding="utf-8")).get(
            "run_plan_fingerprint"
        )
    aggregates = []
    for path in sorted((workspace / "aggregates").glob("task-*.json")):
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        metadata = aggregate.get("openml", {})
        if (
            metadata.get("suite_resolution_fingerprint") == lock["resolution_fingerprint"]
            and metadata.get("run_plan_fingerprint") == run_plan_fingerprint
        ):
            aggregates.append(aggregate)
    return aggregates


def fetch_references(
    workspace: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Fetch the pinned score table for the locked suite, if one exists."""

    lock = load_lock(workspace)
    spec = next((value for value in SUITES.values() if value.suite_id == int(lock["suite"]["id"])), None)
    if spec is None or spec.reference_source is None:
        return []
    cache_directory = workspace / "references"
    source_directory, rows = fetch_score_source(
        spec.reference_source,
        cache_directory,
        offline=offline,
        refresh=refresh,
    )
    _atomic_json(source_directory / "normalized.json", rows)
    return [dict(row) for row in rows]


def _cached_references(workspace: Path) -> list[dict[str, Any]]:
    lock = load_lock(workspace)
    spec = next((value for value in SUITES.values() if value.suite_id == int(lock["suite"]["id"])), None)
    if spec is None or spec.reference_source is None:
        return []
    normalized = workspace / "references" / spec.reference_source / "normalized.json"
    if not normalized.is_file():
        return []
    value = json.loads(normalized.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Invalid normalized reference cache at {normalized}.")
    return value


def write_report(workspace: Path) -> dict[str, Any]:
    """Write JSON, CSV, and Markdown views of available task aggregates."""

    lock = load_lock(workspace)
    report = build_report(lock, _load_aggregates(workspace, lock), reference_rows=_cached_references(workspace))
    _atomic_json(workspace / "summary.json", report)
    _atomic_text(workspace / "summary.csv", report_csv(report))
    _atomic_text(workspace / "summary.md", report_markdown(report))
    return report


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--task-id", type=int, action="append", dest="task_ids")
    parser.add_argument("--max-tasks", type=int)
    parser.add_argument("--max-runs-per-task", type=int)
    parser.add_argument("--max-categories", type=int, default=256)
    parser.add_argument("--max-dense-mib", type=float, default=2048.0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")


def _add_run_arguments(parser: argparse.ArgumentParser, *, include_force: bool = True) -> None:
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-bins", type=int, default=64)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--engine", choices=("auto", "numpy", "numba"), default="auto")
    if include_force:
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--fail-fast", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Resolve/download tasks and prepare official folds.")
    _add_prepare_arguments(prepare)
    run = commands.add_parser("run", help="Run unfinished prepared folds.")
    run.add_argument("--workspace", type=Path, required=True)
    _add_run_arguments(run)
    aggregate = commands.add_parser("aggregate", help="Aggregate fold results per task.")
    aggregate.add_argument("--workspace", type=Path, required=True)
    aggregate.add_argument("--allow-incomplete", action="store_true")
    scores = commands.add_parser("fetch-scores", help="Fetch a pinned public score table.")
    scores.add_argument("--workspace", type=Path, required=True)
    scores.add_argument("--offline", action="store_true")
    scores.add_argument("--refresh", action="store_true")
    report = commands.add_parser("report", help="Write task-balanced JSON/CSV/Markdown reports.")
    report.add_argument("--workspace", type=Path, required=True)
    all_stages = commands.add_parser("all", help="Prepare, run, aggregate, fetch scores, and report.")
    _add_prepare_arguments(all_stages)
    _add_run_arguments(all_stages, include_force=False)
    all_stages.add_argument("--offline-scores", action="store_true")
    all_stages.add_argument("--refresh-scores", action="store_true")
    all_stages.add_argument("--allow-incomplete", action="store_true")
    return parser


def _preprocessing(arguments: argparse.Namespace) -> PreprocessingConfig:
    max_dense_bytes = int(arguments.max_dense_mib * 1024**2)
    return PreprocessingConfig(max_categories=arguments.max_categories, max_dense_bytes=max_dense_bytes)


def _run_arguments(arguments: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": arguments.n_estimators,
        "learning_rate": arguments.learning_rate,
        "max_bins": arguments.max_bins,
        "min_samples_leaf": arguments.min_samples_leaf,
        "max_depth": arguments.max_depth,
        "l2_regularization": arguments.l2_regularization,
        "random_state": arguments.random_state,
        "engine": arguments.engine,
        "force": arguments.force,
        "fail_fast": arguments.fail_fast,
    }


def main(argv: list[str] | None = None, *, provider: OpenMLProvider | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    workspace: Path = arguments.workspace
    if arguments.command in {"prepare", "all"}:
        spec = SUITES[arguments.suite]
        active_provider = provider or RealOpenMLProvider(workspace / "openml-cache")
        prepare_suite(
            active_provider,
            spec,
            workspace,
            task_ids=arguments.task_ids,
            max_tasks=arguments.max_tasks,
            max_runs_per_task=arguments.max_runs_per_task,
            preprocessing=_preprocessing(arguments),
            force=arguments.force,
            fail_fast=arguments.fail_fast,
        )
    if arguments.command in {"run", "all"}:
        counts = run_suite(workspace, **_run_arguments(arguments))
        sys.stdout.write(json.dumps({"run": counts}, sort_keys=True) + "\n")
    if arguments.command in {"aggregate", "all"}:
        aggregate_suite(workspace, allow_incomplete=arguments.allow_incomplete)
    if arguments.command == "fetch-scores":
        rows = fetch_references(workspace, offline=arguments.offline, refresh=arguments.refresh)
        sys.stdout.write(json.dumps({"reference_rows": len(rows)}, sort_keys=True) + "\n")
    elif arguments.command == "all":
        rows = fetch_references(
            workspace,
            offline=arguments.offline_scores,
            refresh=arguments.refresh_scores,
        )
        sys.stdout.write(json.dumps({"reference_rows": len(rows)}, sort_keys=True) + "\n")
    if arguments.command in {"report", "all"}:
        report = write_report(workspace)
        sys.stdout.write(json.dumps({"tasks_reported": report["tasks_reported"]}, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "SUITES",
    "SuiteSpec",
    "aggregate_suite",
    "build_parser",
    "fetch_references",
    "load_lock",
    "main",
    "prepare_suite",
    "run_suite",
    "write_report",
]
