"""Benchmark MBTR variants against external multivariate boosted-tree baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import logging
import multiprocessing
import os
import pickle
import platform
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Literal

import numpy as np

import mbtr
from mbtr import MBTClassifier, MBTRegressor
from studies.catalog import find_published_scores, load_datasets, load_protocols
from studies.metrics import classification_metrics, regression_metrics
from studies.run import _load_split, _restore_target_scale, _validate_task

from .models import SelectedLinearResponseClassifier, SelectedLinearResponseRegressor
from .selection import SelectionPolicy, rank_feature_importances, select_response_features

MANIFEST_FILE = Path(__file__).with_name("manifest.json")
REPOSITORY = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger(__name__)
LOG_FORMAT = "%(asctime)s %(levelname)s [%(processName)s] %(message)s"

DEFAULT_MODELS = ("standard", "linear_response")
EXTERNAL_MODELS = ("xgboost", "lightgbm", "catboost", "xgboost_multi_output")
ALL_MODELS = (*DEFAULT_MODELS, *EXTERNAL_MODELS)
MODEL_CLI_CHOICES = (
    "standard",
    "linear-response",
    "xgboost",
    "lightgbm",
    "catboost",
    "xgboost-multi-output",
)
MODEL_DISPLAY_NAMES = {
    "standard": "MBTR standard",
    "linear_response": "MBTR linear",
    "xgboost": "XGBoost one-output",
    "lightgbm": "LightGBM synchronized",
    "catboost": "CatBoost native",
    "xgboost_multi_output": "XGBoost multi-output",
}
MODEL_PACKAGES = {
    "xgboost": "xgboost",
    "xgboost_multi_output": "xgboost",
    "lightgbm": "lightgbm",
    "catboost": "catboost",
}


@dataclass(frozen=True)
class BenchmarkConfig:
    n_estimators: int = 1000
    learning_rate: float = 0.05
    max_bins: int = 64
    min_samples_leaf: int = 20
    max_depth: int | None = 4
    l2_regularization: float = 0.1
    linear_l2_regularization: float = 0.1
    early_stopping_rounds: int | None = 50
    random_state: int = 42
    engine: str = "auto"
    output_sketch_size: int | None = None
    decision_threshold: float = 0.5
    importance_coverage: float = 0.8
    linear_max_features: int = 8
    linear_min_features: int = 1
    linear_work_budget: float = 2.0e11
    linear_time_budget_seconds: float = 300.0
    linear_memory_budget_bytes: int = 1024**3
    linear_response_curvature: str = "upper_bound"
    force_linear: bool = False

    def validate(self) -> None:
        if self.n_estimators < 1 or self.max_bins < 2 or self.min_samples_leaf < 1:
            raise ValueError("Invalid positive estimator, bin, or minimum-leaf setting.")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be None or positive.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if self.l2_regularization < 0.0 or self.linear_l2_regularization < 0.0:
            raise ValueError("Regularization values must be non-negative.")
        if self.early_stopping_rounds is not None and (
            not isinstance(self.early_stopping_rounds, int) or self.early_stopping_rounds < 1
        ):
            raise ValueError("early_stopping_rounds must be None or a positive integer.")
        if self.engine not in {"auto", "numpy", "numba"}:
            raise ValueError("engine must be auto, numpy, or numba.")
        if self.output_sketch_size is not None and (
            not isinstance(self.output_sketch_size, int)
            or isinstance(self.output_sketch_size, bool)
            or self.output_sketch_size < 1
        ):
            raise ValueError("output_sketch_size must be None or a positive integer.")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be in (0, 1).")
        SelectionPolicy(
            self.importance_coverage,
            self.linear_max_features,
            self.linear_min_features,
        ).validate()
        if not np.isfinite(self.linear_work_budget) or self.linear_work_budget <= 0.0:
            raise ValueError("linear_work_budget must be finite and positive.")
        if not np.isfinite(self.linear_time_budget_seconds) or self.linear_time_budget_seconds <= 0.0:
            raise ValueError("linear_time_budget_seconds must be finite and positive.")
        if self.linear_memory_budget_bytes < 1:
            raise ValueError("linear_memory_budget_bytes must be positive.")
        if self.linear_response_curvature not in {"upper_bound", "exact"}:
            raise ValueError("linear_response_curvature must be 'upper_bound' or 'exact'.")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _configure_logging(level: int | str, *, force: bool = False) -> None:
    """Configure concise stderr progress logs for CLI and worker processes."""

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stderr,
        force=force,
    )


def _configure_worker_logging(level: int) -> None:
    _configure_logging(level, force=True)


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _resolve_models(
    models: list[str] | tuple[str, ...] | None = None,
    *,
    models_all: bool = False,
) -> tuple[str, ...]:
    """Normalize CLI model names and include the linear stage's selector dependency."""

    if models_all:
        if models:
            raise ValueError("models_all and an explicit model list are mutually exclusive.")
        return ALL_MODELS
    if not models:
        return DEFAULT_MODELS
    normalized: list[str] = []
    for value in models:
        name = str(value).replace("-", "_")
        if name not in ALL_MODELS:
            raise ValueError(f"Unknown benchmark model {value!r}.")
        if name not in normalized:
            normalized.append(name)
    if "linear_response" in normalized and "standard" not in normalized:
        normalized.insert(0, "standard")
    return tuple(normalized)


def _preflight_external_dependencies(models: tuple[str, ...]) -> None:
    missing = sorted(
        {
            package
            for model in models
            if (package := MODEL_PACKAGES.get(model)) is not None and _package_version(package) is None
        }
    )
    if not missing:
        return
    packages = ", ".join(missing)
    raise ModuleNotFoundError(
        f"Requested external benchmark models require missing packages: {packages}. "
        "Install them with `uv pip install --python .venv/bin/python "
        "-e '.[studies,external-boosters]'`, "
        "then rerun the same command."
    )


def _metric_display_name(metric: str) -> str:
    return {
        "global_rmse": "RMSE",
        "binary_cross_entropy": "BCE",
        "hamming_loss": "Hamming loss",
        "average_relative_rmse": "aRRMSE",
    }.get(metric, metric)


def _log_stage_result(item: dict[str, Any], stage: str, document: dict[str, Any]) -> None:
    """Emit one unambiguous line as soon as a trained stage becomes terminal."""

    variant = MODEL_DISPLAY_NAMES.get(stage, stage)
    dataset_run = f"{item['dataset_id']}/{item['run_id']}"
    status = str(document.get("status", "unknown"))
    action = "reused" if document.get("execution_status") == "reused" else status
    metrics = document.get("metrics") or {}
    metric = "global_rmse" if item["task"] == "regression" else "binary_cross_entropy"
    value = metrics.get(metric)
    if status == "completed" and value is not None:
        auxiliary = ""
        if item["task"] == "multilabel" and "hamming_loss" in metrics:
            auxiliary = f", Hamming loss={float(metrics['hamming_loss']):.6g}"
        tree_details = ""
        if document.get("effective_trees_retained") is not None:
            tree_details = (
                f"; effective trees retained={document['effective_trees_retained']} "
                f"evaluated={document.get('effective_trees_evaluated', '?')}"
            )
        LOGGER.info(
            "%s %s %s: test %s=%.6g%s; trees retained=%s evaluated=%s%s; fit=%.3fs; predict=%.3fs",
            action,
            dataset_run,
            variant,
            _metric_display_name(metric),
            float(value),
            auxiliary,
            document.get("n_estimators_retained", "?"),
            document.get("n_estimators_evaluated", "?"),
            tree_details,
            float(document.get("fit_seconds", 0.0)),
            float(document.get("predict_seconds", 0.0)),
        )
        return
    reason = document.get("reason") or document.get("error") or "no result details"
    LOGGER.info("%s %s %s: %s", action, dataset_run, variant, reason)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _source_fingerprint() -> str:
    paths = list((REPOSITORY / "mbtr").glob("*.py"))
    paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("models.py"),
            Path(__file__).with_name("selection.py"),
            MANIFEST_FILE,
            REPOSITORY / "studies" / "metrics.py",
            REPOSITORY / "studies" / "run.py",
            REPOSITORY / "studies" / "catalog.py",
        ]
    )
    records = {str(path.relative_to(REPOSITORY)): _file_sha256(path) for path in sorted(paths) if path.is_file()}
    return _canonical_sha256(records)


def load_benchmark_manifest() -> dict[str, Any]:
    """Load and cross-check the 16-dataset primary multivariate manifest."""

    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("datasets"), list):
        raise ValueError("Unsupported multivariate benchmark manifest.")
    datasets = {record["id"]: record for record in load_datasets()}
    protocols = {record["id"]: record for record in load_protocols()}
    expected = {
        identifier
        for identifier, record in datasets.items()
        if record["task"] in {"multioutput_regression", "multilabel_classification"}
    }
    observed: set[str] = set()
    for entry in manifest["datasets"]:
        dataset_id = str(entry.get("dataset_id", ""))
        protocol_id = str(entry.get("protocol_id", ""))
        if dataset_id in observed or dataset_id not in datasets:
            raise ValueError(f"Invalid or duplicate benchmark dataset {dataset_id!r}.")
        if protocol_id not in protocols or dataset_id not in protocols[protocol_id].get("datasets", []):
            raise ValueError(f"Protocol {protocol_id!r} does not cover dataset {dataset_id!r}.")
        catalog_task = datasets[dataset_id]["task"]
        expected_task = "regression" if catalog_task == "multioutput_regression" else "multilabel"
        if entry.get("task") != expected_task:
            raise ValueError(f"Task mismatch for benchmark dataset {dataset_id!r}.")
        if entry.get("linear_response") != "eligible":
            raise ValueError(f"Linear-response status mismatch for {dataset_id!r}.")
        observed.add(dataset_id)
    if observed != expected:
        raise ValueError(
            "Manifest dataset coverage mismatch: "
            f"missing={sorted(expected - observed)} extra={sorted(observed - expected)}."
        )
    return manifest


def _selected_entries(manifest: dict[str, Any], dataset_ids: list[str] | None) -> list[dict[str, Any]]:
    entries = list(manifest["datasets"])
    if dataset_ids:
        requested = set(dataset_ids)
        known = {str(entry["dataset_id"]) for entry in entries}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown benchmark datasets: {unknown}.")
        entries = [entry for entry in entries if entry["dataset_id"] in requested]
    return entries


def benchmark_plan(data_directory: Path, dataset_ids: list[str] | None = None) -> list[dict[str, Any]]:
    """Resolve every expected prepared split and its optional provenance sidecar."""

    manifest = load_benchmark_manifest()
    protocols = {record["id"]: record for record in load_protocols()}
    template = str(manifest["prepared_path_template"])
    plan: list[dict[str, Any]] = []
    for entry in _selected_entries(manifest, dataset_ids):
        protocol = protocols[entry["protocol_id"]]
        for run_id in protocol["expected_run_ids"]:
            relative = template.format(
                protocol_id=entry["protocol_id"],
                dataset_id=entry["dataset_id"],
                run_id=run_id,
            )
            split_path = data_directory / relative
            sidecar_path = split_path.with_suffix(".json")
            plan.append(
                {
                    **entry,
                    "run_id": str(run_id),
                    "split_path": str(split_path),
                    "sidecar_path": str(sidecar_path),
                    "split_exists": split_path.is_file(),
                    "sidecar_exists": sidecar_path.is_file(),
                }
            )
    return plan


def _validate_sidecar(
    path: Path,
    *,
    dataset_id: str,
    protocol_id: str,
    run_id: str,
    split_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {"dataset_id": dataset_id, "protocol_id": protocol_id, "run_id": run_id}
    for key, expected_value in expected.items():
        if str(value.get(key)) != expected_value:
            raise ValueError(f"Prepared sidecar {path} has {key}={value.get(key)!r}; expected {expected_value!r}.")
    if value.get("split_sha256") != split_sha256:
        raise ValueError(f"Prepared sidecar {path} does not match the split checksum.")
    return value


def _feature_names(sidecar: dict[str, Any] | None, n_features: int) -> list[str]:
    candidates: Any = None
    if sidecar is not None:
        candidates = sidecar.get("feature_names")
        if candidates is None:
            candidates = sidecar.get("preprocessing", {}).get("output_feature_names")
    if isinstance(candidates, list) and len(candidates) == n_features:
        return [str(name) for name in candidates]
    return [f"x{index}" for index in range(n_features)]


def _model_parameters(config: BenchmarkConfig, *, linear: bool, random_state: int) -> dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_bins": config.max_bins,
        "min_samples_leaf": config.min_samples_leaf,
        "max_depth": config.max_depth,
        "l2_regularization": (config.linear_l2_regularization if linear else config.l2_regularization),
        "early_stopping_rounds": config.early_stopping_rounds,
        "random_state": random_state,
        "engine": config.engine,
        "output_sketch_size": config.output_sketch_size,
    }


def _has_validation_fold(arrays: dict[str, np.ndarray]) -> bool:
    has_features = "x_validation" in arrays
    has_targets = "y_validation" in arrays
    if has_features != has_targets:
        raise ValueError("Prepared splits must contain both x_validation and y_validation, or neither.")
    return has_features


def _stopping_measurements(
    model: Any,
    *,
    parameters: dict[str, Any],
    validation_rows: int,
) -> dict[str, Any]:
    """Describe validation selection separately from rounds evaluated."""

    training_history = list(model.evals_result_["training"])
    evaluated_rounds = len(training_history)
    retained_trees = int(model.n_estimators_)
    best_iteration = int(model.best_iteration_)
    configured_patience = parameters.get("early_stopping_rounds")
    patience: int | None = None if configured_patience is None else int(configured_patience)
    enabled = patience is not None
    patience_exhausted = bool(patience is not None and evaluated_rounds - (best_iteration + 1) >= patience)
    if patience_exhausted:
        termination_reason = "validation_patience_exhausted"
    elif evaluated_rounds == int(parameters["n_estimators"]):
        termination_reason = "requested_max_evaluated"
    else:
        termination_reason = "training_objective_could_not_improve"
    monitor: str | None = None
    if enabled:
        monitor = "validation_objective"
    elif validation_rows:
        monitor = "validation_objective_observed_only"
    return {
        # Keep the historical field while making its retained-prefix meaning explicit.
        "n_estimators_fitted": retained_trees,
        "n_estimators_evaluated": evaluated_rounds,
        "n_estimators_retained": retained_trees,
        "best_iteration": best_iteration,
        "best_score": float(model.best_score_),
        "stopping": {
            "enabled": enabled,
            "monitor": monitor,
            "patience_rounds": patience,
            "validation_rows": validation_rows,
            "requested_max_trees": int(parameters["n_estimators"]),
            "evaluated_rounds": evaluated_rounds,
            "retained_trees": retained_trees,
            "best_iteration": best_iteration,
            "best_iteration_is_zero_based": True,
            "best_score": float(model.best_score_),
            "patience_exhausted": patience_exhausted,
            "termination_reason": termination_reason,
        },
    }


def _metrics(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    prediction: np.ndarray,
    decision_threshold: float,
) -> dict[str, float]:
    if task == "regression":
        return regression_metrics(
            _restore_target_scale(arrays, arrays["y_train"]),
            _restore_target_scale(arrays, arrays["y_test"]),
            _restore_target_scale(arrays, prediction),
        )
    return classification_metrics(arrays["y_test"], prediction, threshold=decision_threshold)


def _stored_scale_metrics(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    prediction: np.ndarray,
) -> dict[str, float] | None:
    if task != "regression" or "target_offset" not in arrays:
        return None
    return regression_metrics(arrays["y_train"], arrays["y_test"], prediction)


def _standard_stage(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    parameters: dict[str, Any],
    decision_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    recorded_parameters = dict(parameters)
    if task == "regression":
        model: Any = MBTRegressor(**parameters)
    else:
        classifier_parameters = dict(parameters)
        classifier_parameters["decision_threshold"] = decision_threshold
        recorded_parameters["decision_threshold"] = decision_threshold
        model = MBTClassifier(**classifier_parameters)
    has_validation = _has_validation_fold(arrays)
    fit_kwargs: dict[str, Any] = {}
    if has_validation:
        fit_kwargs["eval_set"] = (arrays["x_validation"], arrays["y_validation"])
    started = perf_counter()
    model.fit(arrays["x_train"], arrays["y_train"], **fit_kwargs)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    if task == "regression":
        prediction = np.asarray(model.predict(arrays["x_test"]), dtype=np.float64)
    else:
        prediction = np.asarray(model.predict_proba(arrays["x_test"]), dtype=np.float64)
    predict_seconds = perf_counter() - started
    started = perf_counter()
    metrics = _metrics(
        arrays,
        task=task,
        prediction=prediction,
        decision_threshold=decision_threshold,
    )
    stored_metrics = _stored_scale_metrics(
        arrays,
        task=task,
        prediction=prediction,
    )
    metric_seconds = perf_counter() - started
    importance = np.asarray(model.feature_importances_, dtype=np.float64)
    ranking, _ = rank_feature_importances(importance)
    result = {
        "status": "completed",
        "variant": "standard",
        "parameters": recorded_parameters,
        "metrics": metrics,
        "stored_scale_metrics": stored_metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "metric_seconds": metric_seconds,
        "serialized_model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "resolved_engine": str(model.engine_),
        **_stopping_measurements(
            model,
            parameters=parameters,
            validation_rows=int(len(arrays["x_validation"])) if has_validation else 0,
        ),
        "feature_importances": importance.tolist(),
        "feature_ranking": ranking.tolist(),
        "training_loss": list(model.evals_result_["training"]),
        "validation_loss": list(model.evals_result_["validation"]),
    }
    return model, result


def _linear_stage(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    parameters: dict[str, Any],
    selected_indices: tuple[int, ...],
    decision_threshold: float,
    linear_response_curvature: str,
) -> tuple[Any, dict[str, Any]]:
    recorded_parameters = dict(parameters)
    if task == "regression":
        model: Any = SelectedLinearResponseRegressor(selected_indices, parameters)
    else:
        classifier_parameters = dict(parameters)
        classifier_parameters["decision_threshold"] = decision_threshold
        model = SelectedLinearResponseClassifier(
            selected_indices,
            classifier_parameters,
            linear_response_curvature=linear_response_curvature,
        )
        recorded_parameters.update(
            {
                "decision_threshold": decision_threshold,
                "linear_response_curvature": linear_response_curvature,
                "regularize_intercept": False,
            }
        )
    has_validation = _has_validation_fold(arrays)
    fit_kwargs: dict[str, Any] = {}
    if has_validation:
        fit_kwargs["eval_set"] = (arrays["x_validation"], arrays["y_validation"])
    started = perf_counter()
    model.fit(arrays["x_train"], arrays["y_train"], **fit_kwargs)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    if task == "regression":
        prediction = np.asarray(model.predict(arrays["x_test"]), dtype=np.float64)
    else:
        prediction = np.asarray(model.predict_proba(arrays["x_test"]), dtype=np.float64)
    predict_seconds = perf_counter() - started
    started = perf_counter()
    metrics = _metrics(
        arrays,
        task=task,
        prediction=prediction,
        decision_threshold=decision_threshold,
    )
    stored_metrics = _stored_scale_metrics(
        arrays,
        task=task,
        prediction=prediction,
    )
    metric_seconds = perf_counter() - started
    result = {
        "status": "completed",
        "variant": "linear_response",
        "parameters": recorded_parameters,
        "metrics": metrics,
        "stored_scale_metrics": stored_metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "metric_seconds": metric_seconds,
        "serialized_model_bytes": len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)),
        "resolved_engine": model.engine_,
        **_stopping_measurements(
            model.linear_model_,
            parameters=parameters,
            validation_rows=int(len(arrays["x_validation"])) if has_validation else 0,
        ),
        "design_mean": model.design_mean_.tolist(),
        "design_scale": model.design_scale_.tolist(),
        "training_loss": list(model.linear_model_.evals_result_["training"]),
        "validation_loss": list(model.linear_model_.evals_result_["validation"]),
    }
    return model, result


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "scipy": _package_version("scipy"),
        "numba": _package_version("numba"),
        "mbtr": mbtr.__version__,
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            )
        },
    }


def _dimensions(arrays: dict[str, np.ndarray]) -> dict[str, int]:
    target = arrays["y_train"]
    return {
        "train_rows": int(len(arrays["x_train"])),
        "validation_rows": int(len(arrays.get("x_validation", []))),
        "test_rows": int(len(arrays["x_test"])),
        "features": int(arrays["x_train"].shape[1]),
        "outputs": int(1 if target.ndim == 1 else target.shape[1]),
    }


def _target_transform(arrays: dict[str, np.ndarray]) -> dict[str, Any] | None:
    if "target_offset" not in arrays:
        return None
    return {
        "offset": arrays["target_offset"].tolist(),
        "scale": arrays["target_scale"].tolist(),
    }


def _metric_target_scale(arrays: dict[str, np.ndarray], task: str) -> str | None:
    if task != "regression":
        return None
    return "raw_inverse_transformed" if "target_offset" in arrays else "stored"


def _error_result(error: BaseException) -> dict[str, Any]:
    return {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
    }


def _existing_stage(path: Path, fingerprint: str, *, replace: bool) -> dict[str, Any] | None:
    """Return a reusable terminal stage or reject stale output."""

    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("fingerprint") != fingerprint:
        if not replace:
            raise ValueError(
                f"Existing result {path} has a different fingerprint. "
                "Use --replace only after reviewing the changed data or configuration."
            )
        return None
    if document.get("status") in {"failed", "skipped_dependency"}:
        return None
    document["execution_status"] = "reused"
    return document


def _stage_identity(
    item: dict[str, Any],
    *,
    split_path: Path,
    split_sha256: str,
    sidecar_path: Path,
    sidecar_sha256: str | None,
    source_fingerprint: str,
) -> dict[str, Any]:
    return {
        "dataset_id": str(item["dataset_id"]),
        "protocol_id": str(item["protocol_id"]),
        "run_id": str(item["run_id"]),
        "task": str(item["task"]),
        "split_path": str(split_path),
        "split_sha256": split_sha256,
        "sidecar_path": str(sidecar_path) if sidecar_sha256 is not None else None,
        "sidecar_sha256": sidecar_sha256,
        "source_fingerprint": source_fingerprint,
    }


def _result_header(
    identity: dict[str, Any],
    *,
    variant: str,
    fingerprint: str,
    stage_configuration_fingerprint: str,
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "result_type": "multivariate_benchmark_stage",
        **identity,
        "variant": variant,
        "fingerprint": fingerprint,
        "stage_configuration_fingerprint": stage_configuration_fingerprint,
        "created_at_utc": _utc_now(),
        "dimensions": _dimensions(arrays),
        "metric_target_scale": _metric_target_scale(arrays, str(identity["task"])),
        "target_transform": _target_transform(arrays),
        "environment": _environment(),
        "execution_status": "executed",
    }


def _random_state(
    *,
    protocol: dict[str, Any],
    run_id: str,
    configured: int,
) -> int:
    return int(run_id) if protocol.get("run_axis") == "seed" else configured


def _write_dependency_skip(
    path: Path,
    *,
    header: dict[str, Any],
    standard_fingerprint: str,
) -> dict[str, Any]:
    result = {
        **header,
        "status": "skipped_dependency",
        "reason": "The standard MBTR stage did not complete, so feature selection is unavailable.",
        "standard_fingerprint": standard_fingerprint,
    }
    _atomic_json(path, result)
    return result


def _external_configuration(
    config: BenchmarkConfig,
    *,
    model_name: str,
    random_state: int,
    native_threads: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_bins": config.max_bins,
        "min_samples_leaf": config.min_samples_leaf,
        "max_depth": config.max_depth,
        "l2_regularization": config.l2_regularization,
        "early_stopping_rounds": config.early_stopping_rounds,
        "random_state": random_state,
        "native_threads": native_threads,
    }


def _external_stage_measurements(
    run: Any,
    *,
    arrays: dict[str, np.ndarray],
    task: str,
    decision_threshold: float,
    requested_max_rounds: int,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    started = perf_counter()
    metrics = _metrics(
        arrays,
        task=task,
        prediction=np.asarray(run.prediction, dtype=np.float64),
        decision_threshold=decision_threshold,
    )
    stored_metrics = _stored_scale_metrics(
        arrays,
        task=task,
        prediction=np.asarray(run.prediction, dtype=np.float64),
    )
    metric_seconds = perf_counter() - started
    retained = int(run.rounds_retained)
    evaluated = int(run.rounds_evaluated)
    best_iteration = int(run.best_iteration)
    validation_loss = [float(value) for value in run.validation_loss]
    has_validation = "x_validation" in arrays
    if run.base_validation_loss is None:
        best_score = None
    elif retained == 0:
        best_score = float(run.base_validation_loss)
    elif retained <= len(validation_loss):
        best_score = float(validation_loss[retained - 1])
    else:
        best_score = None
    patience_exhausted = bool(early_stopping_rounds is not None and evaluated - retained >= early_stopping_rounds)
    if patience_exhausted:
        termination_reason = "validation_patience_exhausted"
    elif evaluated == requested_max_rounds:
        termination_reason = "requested_max_evaluated"
    else:
        termination_reason = "training_objective_could_not_improve"
    return {
        "status": "completed",
        "variant": str(run.name).replace("-", "_"),
        "parameters": dict(run.parameters),
        "framework_version": str(run.framework_version),
        "output_strategy": str(run.output_strategy),
        "shared_partition": bool(run.shared_partition),
        "metrics": metrics,
        "stored_scale_metrics": stored_metrics,
        "fit_seconds": float(run.fit_seconds),
        "predict_seconds": float(run.predict_seconds),
        "metric_seconds": metric_seconds,
        "serialized_model_bytes": int(run.serialized_model_bytes),
        "resolved_engine": str(run.name),
        "n_estimators_fitted": retained,
        "n_estimators_evaluated": evaluated,
        "n_estimators_retained": retained,
        "best_iteration": best_iteration,
        "best_score": best_score,
        "trees_per_round": int(run.trees_per_round),
        "effective_trees_evaluated": int(run.effective_trees_evaluated),
        "effective_trees_retained": int(run.effective_trees_retained),
        "base_validation_loss": (None if run.base_validation_loss is None else float(run.base_validation_loss)),
        "training_loss": [float(value) for value in run.training_loss],
        "validation_loss": validation_loss,
        "stopping": {
            "enabled": early_stopping_rounds is not None and has_validation,
            "monitor": (
                "validation_objective"
                if early_stopping_rounds is not None and has_validation
                else "validation_objective_observed_only"
                if has_validation
                else None
            ),
            "patience_rounds": early_stopping_rounds,
            "validation_rows": int(len(arrays.get("x_validation", []))),
            "requested_max_trees": requested_max_rounds,
            "requested_max_rounds": requested_max_rounds,
            "evaluated_rounds": evaluated,
            "retained_trees": retained,
            "retained_rounds": retained,
            "best_iteration": best_iteration,
            "best_iteration_is_zero_based": True,
            "best_score": best_score,
            "patience_exhausted": patience_exhausted,
            "termination_reason": termination_reason,
        },
    }


def _run_external_stages(
    item: dict[str, Any],
    *,
    arrays: dict[str, np.ndarray],
    identity: dict[str, Any],
    output_directory: Path,
    config: BenchmarkConfig,
    random_state: int,
    models: tuple[str, ...],
    native_threads: int,
    replace: bool,
) -> dict[str, dict[str, Any]]:
    selected = [name for name in models if name in EXTERNAL_MODELS]
    if not selected:
        return {}
    from .baselines import ExternalBaselineConfig, fit_external_baseline

    adapter_path = Path(__file__).with_name("baselines.py")
    adapter_fingerprint = _file_sha256(adapter_path)
    result_directory = output_directory / "results" / str(item["dataset_id"]) / str(item["run_id"])
    stages: dict[str, dict[str, Any]] = {}
    for model_name in selected:
        configuration = _external_configuration(
            config,
            model_name=model_name,
            random_state=random_state,
            native_threads=native_threads,
        )
        package = MODEL_PACKAGES[model_name]
        stage_configuration_fingerprint = _canonical_sha256(
            {
                "adapter_source_sha256": adapter_fingerprint,
                "configuration": configuration,
                "framework_package": package,
                "framework_version": _package_version(package),
            }
        )
        fingerprint = _canonical_sha256(
            {
                "stage": model_name,
                "identity": identity,
                "stage_configuration_fingerprint": stage_configuration_fingerprint,
                "decision_threshold": config.decision_threshold,
            }
        )
        path = result_directory / f"{model_name}.json"
        document = _existing_stage(path, fingerprint, replace=replace)
        if document is None:
            header = _result_header(
                identity,
                variant=model_name,
                fingerprint=fingerprint,
                stage_configuration_fingerprint=stage_configuration_fingerprint,
                arrays=arrays,
            )
            try:
                external_task: Literal["regression", "classification"] = (
                    "regression" if item["task"] == "regression" else "classification"
                )
                external_config = ExternalBaselineConfig(
                    n_estimators=config.n_estimators,
                    learning_rate=config.learning_rate,
                    max_bins=config.max_bins,
                    min_samples_leaf=config.min_samples_leaf,
                    max_depth=config.max_depth,
                    l2_regularization=config.l2_regularization,
                    early_stopping_rounds=config.early_stopping_rounds,
                    random_state=random_state,
                    native_threads=native_threads,
                )
                fitted = fit_external_baseline(
                    model_name.replace("_", "-"),
                    arrays,
                    task=external_task,
                    config=external_config,
                )
                measurements = _external_stage_measurements(
                    fitted,
                    arrays=arrays,
                    task=str(item["task"]),
                    decision_threshold=config.decision_threshold,
                    requested_max_rounds=config.n_estimators,
                    early_stopping_rounds=config.early_stopping_rounds,
                )
                document = {**header, **measurements}
            except Exception as error:
                document = {**header, **_error_result(error), "parameters": configuration}
            _atomic_json(path, document)
        _log_stage_result(item, model_name, document)
        stages[model_name] = document
    return stages


def run_work_unit(
    item: dict[str, Any],
    *,
    output_directory: Path,
    config: BenchmarkConfig,
    models: tuple[str, ...] = DEFAULT_MODELS,
    native_threads: int = 1,
    require_sidecar: bool = True,
    replace: bool = False,
) -> dict[str, Any]:
    """Run or resume one manifest-defined dataset/protocol/run combination."""

    config.validate()
    selected_models = _resolve_models(list(models))
    if isinstance(native_threads, bool) or not isinstance(native_threads, int) or native_threads < 1:
        raise ValueError("native_threads must be a positive integer.")
    split_path = Path(item["split_path"])
    sidecar_path = Path(item["sidecar_path"])
    if not split_path.is_file():
        raise FileNotFoundError(f"Prepared split does not exist: {split_path}")
    if require_sidecar and not sidecar_path.is_file():
        raise FileNotFoundError(f"Prepared provenance sidecar does not exist: {sidecar_path}")

    split_sha256 = _file_sha256(split_path)
    sidecar: dict[str, Any] | None = None
    sidecar_sha256: str | None = None
    if sidecar_path.is_file():
        sidecar_sha256 = _file_sha256(sidecar_path)
        sidecar = _validate_sidecar(
            sidecar_path,
            dataset_id=str(item["dataset_id"]),
            protocol_id=str(item["protocol_id"]),
            run_id=str(item["run_id"]),
            split_sha256=split_sha256,
        )

    arrays = _load_split(split_path)
    _validate_task(arrays, str(item["task"]))
    has_validation = _has_validation_fold(arrays)
    if config.early_stopping_rounds is not None and not has_validation:
        raise ValueError(
            "Validation-based early stopping requires x_validation and y_validation in every prepared split. "
            "Prepare a nested validation fold (including for KDIS data), or explicitly use "
            "--disable-early-stopping for a fixed-tree run outside this protocol."
        )
    protocols = {record["id"]: record for record in load_protocols()}
    protocol = protocols[str(item["protocol_id"])]
    random_state = _random_state(
        protocol=protocol,
        run_id=str(item["run_id"]),
        configured=config.random_state,
    )
    source_fingerprint = _source_fingerprint()
    benchmark_lock = _lock_document(config)
    standard_configuration_fingerprint = str(benchmark_lock["standard_configuration_fingerprint"])
    linear_configuration_fingerprint = str(benchmark_lock["linear_configuration_fingerprint"])
    identity = _stage_identity(
        item,
        split_path=split_path,
        split_sha256=split_sha256,
        sidecar_path=sidecar_path,
        sidecar_sha256=sidecar_sha256,
        source_fingerprint=source_fingerprint,
    )

    def finish_with_external(
        completed_stages: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        external_stages = _run_external_stages(
            item,
            arrays=arrays,
            identity=identity,
            output_directory=output_directory,
            config=config,
            random_state=random_state,
            models=selected_models,
            native_threads=native_threads,
            replace=replace,
        )
        return {**completed_stages, **external_stages}

    if not any(name in selected_models for name in DEFAULT_MODELS):
        return finish_with_external({})

    standard_parameters = _model_parameters(config, linear=False, random_state=random_state)
    standard_fingerprint = _canonical_sha256(
        {
            "stage": "standard",
            "identity": identity,
            "stage_configuration_fingerprint": standard_configuration_fingerprint,
            "parameters": standard_parameters,
            "decision_threshold": config.decision_threshold,
        }
    )
    result_directory = output_directory / "results" / str(item["dataset_id"]) / str(item["run_id"])
    standard_path = result_directory / "standard.json"
    linear_path = result_directory / "linear_response.json"
    standard = _existing_stage(standard_path, standard_fingerprint, replace=replace)
    if standard is None:
        header = _result_header(
            identity,
            variant="standard",
            fingerprint=standard_fingerprint,
            stage_configuration_fingerprint=standard_configuration_fingerprint,
            arrays=arrays,
        )
        try:
            _, measurements = _standard_stage(
                arrays,
                task=str(item["task"]),
                parameters=standard_parameters,
                decision_threshold=config.decision_threshold,
            )
            names = _feature_names(sidecar, arrays["x_train"].shape[1])
            standard = {
                **header,
                **measurements,
                "feature_names": names,
                "feature_ranking_names": [names[index] for index in measurements["feature_ranking"]],
            }
        except Exception as error:  # Keep long suites resumable after one failure.
            standard = {**header, **_error_result(error)}
        _atomic_json(standard_path, standard)
    _log_stage_result(item, "standard", standard)

    completed_stages = {"standard": standard}
    if "linear_response" not in selected_models:
        return finish_with_external(completed_stages)

    linear_parameters = _model_parameters(config, linear=True, random_state=random_state)
    selection_policy = SelectionPolicy(
        importance_coverage=config.importance_coverage,
        max_features=config.linear_max_features,
        min_features=config.linear_min_features,
    )
    linear_fingerprint = _canonical_sha256(
        {
            "stage": "linear_response",
            "identity": identity,
            "stage_configuration_fingerprint": linear_configuration_fingerprint,
            "standard_fingerprint": standard_fingerprint,
            "parameters": linear_parameters,
            "selection_policy": asdict(selection_policy),
            "linear_response_curvature": config.linear_response_curvature,
            "budgets": {
                "work": config.linear_work_budget,
                "seconds": config.linear_time_budget_seconds,
                "memory_bytes": config.linear_memory_budget_bytes,
                "force": config.force_linear,
            },
        }
    )
    linear = _existing_stage(linear_path, linear_fingerprint, replace=replace)
    if linear is not None:
        _log_stage_result(item, "linear_response", linear)
        return finish_with_external({**completed_stages, "linear_response": linear})
    linear_header = _result_header(
        identity,
        variant="linear_response",
        fingerprint=linear_fingerprint,
        stage_configuration_fingerprint=linear_configuration_fingerprint,
        arrays=arrays,
    )
    if standard.get("status") != "completed":
        linear = _write_dependency_skip(
            linear_path,
            header=linear_header,
            standard_fingerprint=standard_fingerprint,
        )
        _log_stage_result(item, "linear_response", linear)
        return finish_with_external({**completed_stages, "linear_response": linear})

    started = perf_counter()
    selection = select_response_features(
        standard["feature_importances"],
        policy=selection_policy,
        n_samples=len(arrays["x_train"]),
        n_outputs=_dimensions(arrays)["outputs"],
        n_estimators=config.n_estimators,
        max_bins=config.max_bins,
        max_depth=config.max_depth,
        work_budget=config.linear_work_budget,
        time_budget_seconds=config.linear_time_budget_seconds,
        memory_budget_bytes=config.linear_memory_budget_bytes,
        force=config.force_linear,
        leaf_kind="classification" if item["task"] == "multilabel" else "regression",
        curvature_mode=config.linear_response_curvature,
    )
    selection_seconds = perf_counter() - started
    feature_names = list(standard["feature_names"])
    selection_document = selection.as_dict()
    selection_document.update(
        {
            "policy": asdict(selection_policy),
            "selected_names": [feature_names[index] for index in selection.selected_indices],
            "importance_selected_names": [feature_names[index] for index in selection.importance_selected_indices],
            "selection_rule_seconds": selection_seconds,
            "selector_model_fit_seconds": float(standard["fit_seconds"]),
            "uses_test_data": False,
            "tie_break": "descending split gain, then ascending original column index",
        }
    )
    if selection.status != "run":
        linear = {
            **linear_header,
            "status": "skipped_guard",
            "reason": selection.reason,
            "guard_status": selection.status,
            "standard_fingerprint": standard_fingerprint,
            "selection": selection_document,
        }
        _atomic_json(linear_path, linear)
        _log_stage_result(item, "linear_response", linear)
        return finish_with_external({**completed_stages, "linear_response": linear})

    try:
        _, measurements = _linear_stage(
            arrays,
            task=str(item["task"]),
            parameters=linear_parameters,
            selected_indices=selection.selected_indices,
            decision_threshold=config.decision_threshold,
            linear_response_curvature=config.linear_response_curvature,
        )
        measurements["two_stage_fit_seconds"] = (
            float(standard["fit_seconds"]) + selection_seconds + float(measurements["fit_seconds"])
        )
        linear = {
            **linear_header,
            **measurements,
            "standard_fingerprint": standard_fingerprint,
            "selection": selection_document,
        }
    except Exception as error:  # Preserve the successful screening result.
        linear = {
            **linear_header,
            **_error_result(error),
            "standard_fingerprint": standard_fingerprint,
            "selection": selection_document,
        }
    _atomic_json(linear_path, linear)
    _log_stage_result(item, "linear_response", linear)
    return finish_with_external({**completed_stages, "linear_response": linear})


def _standard_configuration(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "max_bins": config.max_bins,
        "min_samples_leaf": config.min_samples_leaf,
        "max_depth": config.max_depth,
        "l2_regularization": config.l2_regularization,
        "early_stopping_rounds": config.early_stopping_rounds,
        "random_state": config.random_state,
        "engine": config.engine,
        "output_sketch_size": config.output_sketch_size,
        "decision_threshold": config.decision_threshold,
    }


def _linear_configuration(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "standard": _standard_configuration(config),
        "linear_l2_regularization": config.linear_l2_regularization,
        "importance_coverage": config.importance_coverage,
        "linear_max_features": config.linear_max_features,
        "linear_min_features": config.linear_min_features,
        "linear_work_budget": config.linear_work_budget,
        "linear_time_budget_seconds": config.linear_time_budget_seconds,
        "linear_memory_budget_bytes": config.linear_memory_budget_bytes,
        "linear_response_curvature": config.linear_response_curvature,
        "force_linear": config.force_linear,
    }


def _lock_document(config: BenchmarkConfig) -> dict[str, Any]:
    provenance = {
        "manifest_sha256": _file_sha256(MANIFEST_FILE),
        "source_fingerprint": _source_fingerprint(),
        "environment": _environment(),
    }
    standard_configuration_fingerprint = _canonical_sha256(
        {"provenance": provenance, "configuration": _standard_configuration(config)}
    )
    linear_configuration_fingerprint = _canonical_sha256(
        {"provenance": provenance, "configuration": _linear_configuration(config)}
    )
    content = {
        "schema_version": 1,
        **provenance,
        "config": asdict(config),
        "standard_configuration_fingerprint": standard_configuration_fingerprint,
        "linear_configuration_fingerprint": linear_configuration_fingerprint,
    }
    return {**content, "fingerprint": _canonical_sha256(content)}


def _prepare_output_lock(
    output_directory: Path,
    config: BenchmarkConfig,
    *,
    replace: bool,
) -> None:
    path = output_directory / "benchmark-lock.json"
    requested = _lock_document(config)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != requested and not replace:
            raise ValueError(
                f"Benchmark lock {path} differs from the requested source or configuration. "
                "Choose a new output directory or pass --replace after review."
            )
    _atomic_json(path, requested)


def run_benchmark(
    data_directory: Path,
    output_directory: Path,
    *,
    config: BenchmarkConfig | None = None,
    dataset_ids: list[str] | None = None,
    allow_missing: bool = False,
    require_sidecars: bool = True,
    replace: bool = False,
    fail_fast: bool = False,
    jobs: int = 1,
    models: list[str] | tuple[str, ...] | None = None,
    native_threads: int = 1,
) -> dict[str, Any]:
    """Run every selected work unit and write independently resumable stages."""

    resolved_config = BenchmarkConfig() if config is None else config
    resolved_config.validate()
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
        raise ValueError("jobs must be a positive integer.")
    if isinstance(native_threads, bool) or not isinstance(native_threads, int) or native_threads < 1:
        raise ValueError("native_threads must be a positive integer.")
    selected_models = _resolve_models(models)
    _preflight_external_dependencies(selected_models)
    plan = benchmark_plan(data_directory, dataset_ids)
    missing_splits = [item["split_path"] for item in plan if not item["split_exists"]]
    missing_sidecars = [item["sidecar_path"] for item in plan if not item["sidecar_exists"]]
    problems: list[str] = []
    if missing_splits and not allow_missing:
        problems.append("missing prepared splits:\n  " + "\n  ".join(missing_splits))
    if missing_sidecars and require_sidecars and not allow_missing:
        problems.append("missing provenance sidecars:\n  " + "\n  ".join(missing_sidecars))
    if problems:
        raise FileNotFoundError(
            "The benchmark inputs are incomplete. Run the plan command for the exact contract, "
            "or pass --allow-missing for a partial smoke run.\n" + "\n".join(problems)
        )
    _prepare_output_lock(output_directory, resolved_config, replace=replace)

    counts: Counter[str] = Counter()
    failures: list[dict[str, str]] = []
    executed = 0
    available_items: list[dict[str, Any]] = []
    for item in plan:
        if not item["split_exists"] or require_sidecars and not item["sidecar_exists"]:
            counts["missing_input"] += 1
            LOGGER.warning("missing input %s/%s; work unit skipped", item["dataset_id"], item["run_id"])
            continue
        available_items.append(item)

    active_workers = min(jobs, len(available_items))
    LOGGER.info(
        "benchmark start: planned=%d available=%d jobs=%d native_threads=%d models=%s",
        len(plan),
        len(available_items),
        active_workers,
        native_threads,
        ",".join(selected_models),
    )
    if active_workers > 1:
        LOGGER.info("parallel workers execute independent dataset/run units; stages within each unit remain ordered")
        LOGGER.warning(
            "process parallelism can oversubscribe native threads; set OMP_NUM_THREADS, MKL_NUM_THREADS, "
            "OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS, and NUMBA_NUM_THREADS to 1 unless measured otherwise"
        )

    def record_success(item: dict[str, Any], stages: dict[str, dict[str, Any]]) -> None:
        nonlocal executed
        executed += 1
        for name, document in stages.items():
            counts[f"{name}:{document['status']}"] += 1
            counts[f"execution:{document.get('execution_status', 'executed')}"] += 1
            if document["status"] == "failed":
                failures.append(
                    {
                        "dataset_id": str(item["dataset_id"]),
                        "run_id": str(item["run_id"]),
                        "stage": name,
                        "error": str(document.get("error", "unknown failure")),
                    }
                )

    def record_error(item: dict[str, Any], error: BaseException) -> None:
        failures.append(
            {
                "dataset_id": str(item["dataset_id"]),
                "run_id": str(item["run_id"]),
                "stage": "orchestration",
                "error": f"{type(error).__name__}: {error}",
            }
        )
        counts["orchestration:failed"] += 1
        LOGGER.error(
            "failed %s/%s orchestration: %s: %s",
            item["dataset_id"],
            item["run_id"],
            type(error).__name__,
            error,
        )

    if active_workers <= 1:
        for item in available_items:
            try:
                stages = run_work_unit(
                    item,
                    output_directory=output_directory,
                    config=resolved_config,
                    models=selected_models,
                    native_threads=native_threads,
                    require_sidecar=require_sidecars,
                    replace=replace,
                )
                record_success(item, stages)
            except Exception as error:
                record_error(item, error)
                if fail_fast:
                    raise
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=active_workers,
            mp_context=context,
            initializer=_configure_worker_logging,
            initargs=(LOGGER.getEffectiveLevel(),),
        ) as executor:
            futures = {
                executor.submit(
                    run_work_unit,
                    item,
                    output_directory=output_directory,
                    config=resolved_config,
                    models=selected_models,
                    native_threads=native_threads,
                    require_sidecar=require_sidecars,
                    replace=replace,
                ): item
                for item in available_items
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record_success(item, future.result())
                except Exception as error:
                    record_error(item, error)
                    if fail_fast:
                        for pending in futures:
                            pending.cancel()
                        raise
    failures.sort(key=lambda value: (value["dataset_id"], value["run_id"], value["stage"], value["error"]))
    summary = {
        "schema_version": 1,
        "created_at_utc": _utc_now(),
        "planned_work_units": len(plan),
        "executed_work_units": executed,
        "missing_split_count": len(missing_splits),
        "missing_sidecar_count": len(missing_sidecars),
        "jobs_requested": jobs,
        "parallel_workers": active_workers,
        "models": list(selected_models),
        "native_threads": native_threads,
        "counts": dict(sorted(counts.items())),
        "failures": failures,
        "complete_without_failures": executed == len(plan) and not failures,
    }
    _atomic_json(output_directory / "run-summary.json", summary)
    return summary


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in values],
        "count": len(values),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "median": float(median(values)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _summarize_metrics(
    documents: list[dict[str, Any]],
    *,
    field: str = "metrics",
) -> dict[str, Any]:
    names = sorted({name for document in documents for name in (document.get(field) or {})})
    return {
        name: _summary([float(document[field][name]) for document in documents if name in (document.get(field) or {})])
        for name in names
    }


def _metric_direction(name: str) -> str:
    if name in {"global_rmse", "mae", "average_relative_rmse", "binary_cross_entropy", "hamming_loss"}:
        return "minimize"
    return "maximize"


def _paired_comparison(
    standard_documents: list[dict[str, Any]],
    comparison_documents: list[dict[str, Any]],
    *,
    field: str = "metrics",
    comparison_name: str = "linear_response",
) -> dict[str, Any]:
    standard_by_run = {str(document["run_id"]): document for document in standard_documents}
    comparison_by_run = {str(document["run_id"]): document for document in comparison_documents}
    paired_ids = sorted(
        run_id
        for run_id in set(standard_by_run) & set(comparison_by_run)
        if standard_by_run[run_id].get(field) and comparison_by_run[run_id].get(field)
    )
    metric_names = (
        sorted(
            set.intersection(
                *(set(standard_by_run[run_id][field]) & set(comparison_by_run[run_id][field]) for run_id in paired_ids)
            )
        )
        if paired_ids
        else []
    )
    comparisons: dict[str, Any] = {}
    key_prefix = "linear" if comparison_name == "linear_response" else comparison_name
    for name in metric_names:
        direction = _metric_direction(name)
        standard_values = [float(standard_by_run[run_id][field][name]) for run_id in paired_ids]
        comparison_values = [float(comparison_by_run[run_id][field][name]) for run_id in paired_ids]
        pairs = list(zip(standard_values, comparison_values, strict=True))
        deltas = [comparison - standard for standard, comparison in pairs]
        improvements = [
            standard - comparison if direction == "minimize" else comparison - standard
            for standard, comparison in pairs
        ]
        tolerance = 1e-12
        wins = sum(value > tolerance for value in improvements)
        losses = sum(value < -tolerance for value in improvements)
        metric_result: dict[str, Any] = {
            "direction": direction,
            "standard": _summary(standard_values),
            comparison_name: _summary(comparison_values),
            f"{key_prefix}_minus_standard": _summary(deltas),
            "improvement_in_preferred_direction": _summary(improvements),
            "wins_ties_losses": {
                "wins": wins,
                "ties": len(improvements) - wins - losses,
                "losses": losses,
            },
        }
        if all(value > 0.0 for value in standard_values):
            metric_result[f"{key_prefix}_over_standard_ratio"] = _summary(
                [comparison / standard for standard, comparison in pairs]
            )
        comparisons[name] = metric_result
    return {"paired_run_ids": paired_ids, "metrics": comparisons}


def _load_stage(
    path: Path,
    *,
    stage_configuration_fingerprint: str | None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        stage_configuration_fingerprint is not None
        and document.get("stage_configuration_fingerprint") != stage_configuration_fingerprint
    ):
        return {
            "status": "stale",
            "reason": "The stage does not match the current benchmark lock.",
            "dataset_id": document.get("dataset_id"),
            "run_id": document.get("run_id"),
            "variant": document.get("variant"),
        }
    return document


def _stage_has_expected_identity(
    document: dict[str, Any],
    *,
    entry: dict[str, Any],
    run_id: str,
    variant: str,
) -> bool:
    expected = {
        "dataset_id": str(entry["dataset_id"]),
        "protocol_id": str(entry["protocol_id"]),
        "run_id": run_id,
        "task": str(entry["task"]),
        "variant": variant,
    }
    return all(str(document.get(name)) == value for name, value in expected.items())


def _stale_stage(document: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "status": "stale",
        "reason": reason,
        "dataset_id": document.get("dataset_id"),
        "run_id": document.get("run_id"),
        "variant": document.get("variant"),
    }


def aggregate_benchmark(
    output_directory: Path,
    *,
    dataset_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate only exact paired completed stages, preserving omissions."""

    manifest = load_benchmark_manifest()
    protocols = {record["id"]: record for record in load_protocols()}
    lock_path = output_directory / "benchmark-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else None
    if lock is not None:
        if not isinstance(lock, dict) or not isinstance(lock.get("fingerprint"), str):
            raise ValueError(f"Benchmark lock {lock_path} is malformed.")
        lock_content = {name: value for name, value in lock.items() if name != "fingerprint"}
        if _canonical_sha256(lock_content) != lock["fingerprint"]:
            raise ValueError(f"Benchmark lock {lock_path} failed its integrity check.")
    lock_fingerprint = None if lock is None else lock.get("fingerprint")
    benchmark_fingerprint = lock_fingerprint if isinstance(lock_fingerprint, str) else None
    standard_lock_fingerprint = None if lock is None else lock.get("standard_configuration_fingerprint")
    standard_configuration_fingerprint = (
        standard_lock_fingerprint if isinstance(standard_lock_fingerprint, str) else None
    )
    linear_lock_fingerprint = None if lock is None else lock.get("linear_configuration_fingerprint")
    linear_configuration_fingerprint = linear_lock_fingerprint if isinstance(linear_lock_fingerprint, str) else None
    results_root = output_directory / "results"
    observed_external_models = tuple(
        model_name
        for model_name in EXTERNAL_MODELS
        if results_root.is_dir() and any(results_root.glob(f"*/*/{model_name}.json"))
    )
    datasets: list[dict[str, Any]] = []
    regression_ratios: list[float] = []
    classification_ratios: list[float] = []
    headline_ratios: list[float] = []
    for entry in _selected_entries(manifest, dataset_ids):
        protocol = protocols[entry["protocol_id"]]
        expected_ids = [str(value) for value in protocol["expected_run_ids"]]
        standard_documents: list[dict[str, Any]] = []
        linear_documents: list[dict[str, Any]] = []
        linear_statuses: Counter[str] = Counter()
        standard_statuses: Counter[str] = Counter()
        external_documents: dict[str, list[dict[str, Any]]] = {name: [] for name in observed_external_models}
        external_statuses: dict[str, Counter[str]] = {name: Counter() for name in observed_external_models}
        selected_counts: Counter[str] = Counter()
        for run_id in expected_ids:
            base = output_directory / "results" / entry["dataset_id"] / run_id
            standard = _load_stage(
                base / "standard.json",
                stage_configuration_fingerprint=standard_configuration_fingerprint,
            )
            linear = _load_stage(
                base / "linear_response.json",
                stage_configuration_fingerprint=linear_configuration_fingerprint,
            )
            external = {
                name: _load_stage(
                    base / f"{name}.json",
                    stage_configuration_fingerprint=None,
                )
                for name in observed_external_models
            }
            if standard is not None and not _stage_has_expected_identity(
                standard,
                entry=entry,
                run_id=run_id,
                variant="standard",
            ):
                standard = _stale_stage(
                    standard,
                    "The standard artifact identity does not match its manifest path.",
                )
            if linear is not None and not _stage_has_expected_identity(
                linear,
                entry=entry,
                run_id=run_id,
                variant="linear_response",
            ):
                linear = _stale_stage(
                    linear,
                    "The linear artifact identity does not match its manifest path.",
                )
            if linear is not None and linear.get("status") != "stale":
                if standard is None or standard.get("status") == "stale":
                    linear = _stale_stage(
                        linear,
                        "The linear artifact has no matching current standard artifact.",
                    )
                elif (
                    linear.get("standard_fingerprint") != standard.get("fingerprint")
                    or linear.get("split_sha256") != standard.get("split_sha256")
                    or linear.get("sidecar_sha256") != standard.get("sidecar_sha256")
                    or linear.get("dimensions") != standard.get("dimensions")
                ):
                    linear = _stale_stage(
                        linear,
                        "The linear artifact does not match its standard-stage dependency.",
                    )
            for name, document in external.items():
                if document is not None and not _stage_has_expected_identity(
                    document,
                    entry=entry,
                    run_id=run_id,
                    variant=name,
                ):
                    external[name] = _stale_stage(
                        document,
                        f"The {name} artifact identity does not match its manifest path.",
                    )
            standard_statuses["missing" if standard is None else str(standard.get("status"))] += 1
            linear_statuses["missing" if linear is None else str(linear.get("status"))] += 1
            if standard is not None and standard.get("status") == "completed":
                standard_documents.append(standard)
            if linear is not None and linear.get("status") == "completed":
                linear_documents.append(linear)
                for name in linear.get("selection", {}).get("selected_names", []):
                    selected_counts[str(name)] += 1
            for name, document in external.items():
                external_statuses[name]["missing" if document is None else str(document.get("status"))] += 1
                if document is not None and document.get("status") == "completed":
                    external_documents[name].append(document)
        paired = _paired_comparison(standard_documents, linear_documents)
        paired_stored_scale = _paired_comparison(
            standard_documents,
            linear_documents,
            field="stored_scale_metrics",
        )
        headline_metric = "global_rmse" if entry["task"] == "regression" else "binary_cross_entropy"
        ratio = paired.get("metrics", {}).get(headline_metric, {}).get("linear_over_standard_ratio", {}).get("mean")
        complete_standard = len(standard_documents) == len(expected_ids)
        complete_linear = len(linear_documents) == len(expected_ids)
        if (
            complete_standard
            and complete_linear
            and len(paired["paired_run_ids"]) == len(expected_ids)
            and ratio is not None
            and ratio > 0.0
        ):
            target = regression_ratios if entry["task"] == "regression" else classification_ratios
            target.append(float(ratio))
            headline_ratios.append(float(ratio))
        observed_metric_names = set(_summarize_metrics(standard_documents)) if standard_documents else None
        references = find_published_scores(
            entry["dataset_id"],
            entry["protocol_id"],
            observed_metric_names,
        )
        timing: dict[str, Any] = {}
        if standard_documents:
            timing["standard_fit_seconds"] = _summary(
                [float(document["fit_seconds"]) for document in standard_documents]
            )
        if linear_documents:
            timing.update(
                {
                    "linear_incremental_fit_seconds": _summary(
                        [float(document["fit_seconds"]) for document in linear_documents]
                    ),
                    "two_stage_fit_seconds": _summary(
                        [float(document["two_stage_fit_seconds"]) for document in linear_documents]
                    ),
                }
            )
        external_variants: dict[str, Any] = {}
        for name in observed_external_models:
            documents = external_documents[name]
            comparison = _paired_comparison(
                standard_documents,
                documents,
                comparison_name=name,
            )
            headline_comparison = comparison.get("metrics", {}).get(headline_metric, {})
            variant_timing: dict[str, Any] = {}
            if documents:
                variant_timing = {
                    "fit_seconds": _summary([float(document["fit_seconds"]) for document in documents]),
                    "predict_seconds": _summary([float(document["predict_seconds"]) for document in documents]),
                    "rounds_evaluated": _summary([float(document["n_estimators_evaluated"]) for document in documents]),
                    "rounds_retained": _summary([float(document["n_estimators_retained"]) for document in documents]),
                    "effective_trees_evaluated": _summary(
                        [float(document["effective_trees_evaluated"]) for document in documents]
                    ),
                    "effective_trees_retained": _summary(
                        [float(document["effective_trees_retained"]) for document in documents]
                    ),
                }
            external_variants[name] = {
                "display_name": MODEL_DISPLAY_NAMES[name],
                "statuses": dict(sorted(external_statuses[name].items())),
                "complete": len(documents) == len(expected_ids),
                "completed_runs": len(documents),
                "metrics": _summarize_metrics(documents),
                "stored_scale_metrics": _summarize_metrics(documents, field="stored_scale_metrics"),
                "paired_comparison": comparison,
                "headline_metric": headline_metric,
                "headline_mean": (
                    None if not documents else _summarize_metrics(documents).get(headline_metric, {}).get("mean")
                ),
                "over_standard_ratio": headline_comparison.get(f"{name}_over_standard_ratio", {}).get("mean"),
                "timing": variant_timing,
            }
        datasets.append(
            {
                **entry,
                "expected_run_ids": expected_ids,
                "standard_statuses": dict(sorted(standard_statuses.items())),
                "linear_response_statuses": dict(sorted(linear_statuses.items())),
                "standard_complete": complete_standard,
                "linear_response_complete": complete_linear,
                "comparison_ready": complete_standard and complete_linear,
                "standard_metrics": _summarize_metrics(standard_documents),
                "linear_response_metrics": _summarize_metrics(linear_documents),
                "paired_comparison": paired,
                "standard_stored_scale_metrics": _summarize_metrics(
                    standard_documents,
                    field="stored_scale_metrics",
                ),
                "linear_response_stored_scale_metrics": _summarize_metrics(
                    linear_documents,
                    field="stored_scale_metrics",
                ),
                "paired_stored_scale_comparison": paired_stored_scale,
                "selection_frequency": dict(sorted(selected_counts.items())),
                "timing": timing,
                "external_variants": external_variants,
                "published_references": [asdict(score) for score in references] if complete_standard else [],
                "published_references_withheld": 0 if complete_standard else len(references),
            }
        )
    suite_summary: dict[str, Any] = {
        "datasets": len(datasets),
        "comparison_ready_datasets": sum(bool(dataset["comparison_ready"]) for dataset in datasets),
        "external_variants": {
            name: {
                "complete_datasets": sum(
                    bool(dataset.get("external_variants", {}).get(name, {}).get("complete")) for dataset in datasets
                )
            }
            for name in observed_external_models
        },
    }
    if regression_ratios:
        suite_summary["equal_dataset_global_rmse_ratio"] = {
            **_summary(regression_ratios),
            "geometric_mean": float(np.exp(np.mean(np.log(regression_ratios)))),
            "note": (
                "Each complete regression dataset contributes one paired mean ratio; raw RMSE is never "
                "pooled across datasets."
            ),
        }
    if classification_ratios:
        suite_summary["equal_dataset_binary_cross_entropy_ratio"] = {
            **_summary(classification_ratios),
            "geometric_mean": float(np.exp(np.mean(np.log(classification_ratios)))),
            "note": (
                "Each complete multilabel dataset contributes one paired mean ratio; per-label examples "
                "are never pooled across datasets."
            ),
        }
    if headline_ratios:
        suite_summary["equal_dataset_headline_loss_ratio"] = {
            **_summary(headline_ratios),
            "geometric_mean": float(np.exp(np.mean(np.log(headline_ratios)))),
            "note": (
                "Each complete dataset contributes one ratio: global RMSE for regression and binary "
                "cross-entropy for multilabel classification."
            ),
        }
    return {
        "schema_version": 1,
        "result_type": "multivariate_benchmark_aggregate",
        "created_at_utc": _utc_now(),
        "benchmark_lock_validated": (
            benchmark_fingerprint is not None
            and standard_configuration_fingerprint is not None
            and linear_configuration_fingerprint is not None
        ),
        "benchmark_fingerprint": benchmark_fingerprint,
        "external_model_stages": list(observed_external_models),
        "suite": suite_summary,
        "datasets": datasets,
    }


def _report_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in aggregate["datasets"]:
        metric_name = "global_rmse" if dataset["task"] == "regression" else "binary_cross_entropy"
        comparison = dataset["paired_comparison"].get("metrics", {}).get(metric_name, {})
        reference = _headline_reference(dataset)
        row: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "task": dataset["task"],
            "protocol_id": dataset["protocol_id"],
            "standard_complete": dataset["standard_complete"],
            "linear_response_complete": dataset["linear_response_complete"],
            "paired_runs": len(dataset["paired_comparison"]["paired_run_ids"]),
            "headline_metric": metric_name,
            "standard_headline_metric": comparison.get("standard", {}).get("mean"),
            "linear_headline_metric": comparison.get("linear_response", {}).get("mean"),
            "linear_over_standard_headline": comparison.get("linear_over_standard_ratio", {}).get("mean"),
            "standard_global_rmse": (
                comparison.get("standard", {}).get("mean") if metric_name == "global_rmse" else None
            ),
            "linear_global_rmse": (
                comparison.get("linear_response", {}).get("mean") if metric_name == "global_rmse" else None
            ),
            "linear_over_standard_rmse": (
                comparison.get("linear_over_standard_ratio", {}).get("mean") if metric_name == "global_rmse" else None
            ),
            "standard_binary_cross_entropy": dataset["standard_metrics"].get("binary_cross_entropy", {}).get("mean"),
            "linear_binary_cross_entropy": dataset["linear_response_metrics"]
            .get("binary_cross_entropy", {})
            .get("mean"),
            "linear_over_standard_binary_cross_entropy": (
                comparison.get("linear_over_standard_ratio", {}).get("mean")
                if metric_name == "binary_cross_entropy"
                else None
            ),
            "standard_hamming_loss": dataset["standard_metrics"].get("hamming_loss", {}).get("mean"),
            "standard_example_f1": dataset["standard_metrics"].get("example_f1", {}).get("mean"),
            "published_method": None if reference is None else reference["method"],
            "published_metric": None if reference is None else reference["metric"],
            "published_value": None if reference is None else reference["value"],
            "published_comparison_tier": None if reference is None else reference["comparison_tier"],
            "linear_statuses": json.dumps(dataset["linear_response_statuses"], sort_keys=True),
        }
        for name in aggregate.get("external_model_stages", []):
            variant = dataset.get("external_variants", {}).get(name, {})
            timing = variant.get("timing", {})
            row.update(
                {
                    f"{name}_complete": variant.get("complete", False),
                    f"{name}_completed_runs": variant.get("completed_runs", 0),
                    f"{name}_headline_metric": variant.get("headline_mean"),
                    f"{name}_over_standard": variant.get("over_standard_ratio"),
                    f"{name}_fit_seconds_median": timing.get("fit_seconds", {}).get("median"),
                    f"{name}_predict_seconds_median": timing.get("predict_seconds", {}).get("median"),
                    f"{name}_rounds_retained_median": timing.get("rounds_retained", {}).get("median"),
                    f"{name}_effective_trees_retained_median": timing.get("effective_trees_retained", {}).get("median"),
                    f"{name}_statuses": json.dumps(variant.get("statuses", {}), sort_keys=True),
                }
            )
        rows.append(row)
    return rows


def _headline_reference(dataset: dict[str, Any]) -> dict[str, Any] | None:
    preferred_metrics = (
        ("global_rmse", "average_relative_rmse")
        if dataset["task"] == "regression"
        else ("binary_cross_entropy", "hamming_loss")
    )
    eligible_references = [
        reference
        for reference in dataset.get("published_references", [])
        if reference.get("comparison_tier") in {"direct", "protocol_reference"}
    ]
    candidates: list[dict[str, Any]] = []
    for metric in preferred_metrics:
        candidates = [reference for reference in eligible_references if reference.get("metric") == metric]
        if candidates:
            break
    if not candidates:
        return None
    direct = [reference for reference in candidates if reference.get("comparison_tier") == "direct"]
    pool = direct or candidates
    direction = str(pool[0].get("direction", "minimize"))
    return (min if direction == "minimize" else max)(pool, key=lambda value: float(value["value"]))


def _log_published_references(aggregate: dict[str, Any]) -> None:
    """Log eligible catalog values without implying that this runner fitted them."""

    for dataset in aggregate.get("datasets", []):
        for reference in dataset.get("published_references", []):
            if reference.get("comparison_tier") not in {"direct", "protocol_reference"}:
                continue
            metric = str(reference["metric"])
            LOGGER.info(
                "published %s reference %s: test %s=%.6g; tier=%s",
                reference["method"],
                dataset["dataset_id"],
                _metric_display_name(metric),
                float(reference["value"]),
                reference["comparison_tier"],
            )


def _markdown_report(aggregate: dict[str, Any]) -> str:
    lines = [
        "# MBTR multivariate benchmark",
        "",
        (
            "Linear-response leaves use only features selected from the fitted standard MBTR's "
            "training-time split gains. The test partition is not used for selection."
        ),
        "",
        (
            "Both variants fit on training rows, retain the validation-best tree prefix under a "
            "frozen configuration, and evaluate the common test partition once."
        ),
        "",
        (
            "| Dataset | Task | Standard runs | Linear status | Paired runs | Metric | Standard | "
            "Linear | Linear / standard | Published best |"
        ),
        "|---|---:|---:|---|---:|---|---:|---:|---:|---|",
    ]
    for dataset in aggregate["datasets"]:
        expected = len(dataset["expected_run_ids"])
        completed = dataset["standard_statuses"].get("completed", 0)
        statuses = ", ".join(f"{key}={value}" for key, value in dataset["linear_response_statuses"].items())
        metric_name = "global_rmse" if dataset["task"] == "regression" else "binary_cross_entropy"
        comparison = dataset["paired_comparison"].get("metrics", {}).get(metric_name, {})
        standard_value = comparison.get("standard", {}).get("mean")
        linear_value = comparison.get("linear_response", {}).get("mean")
        ratio = comparison.get("linear_over_standard_ratio", {}).get("mean")
        standard_text = "—" if standard_value is None else f"{float(standard_value):.4f}"
        linear_text = "—" if linear_value is None else f"{float(linear_value):.4f}"
        ratio_text = "—" if ratio is None else f"{float(ratio):.4f}"
        reference = _headline_reference(dataset)
        reference_text = (
            "—"
            if reference is None
            else (
                f"{reference['method']} {reference['metric']}={float(reference['value']):.4f} "
                f"({reference['comparison_tier']})"
            )
        )
        lines.append(
            f"| {dataset['dataset_id']} | {dataset['task']} | {completed}/{expected} | "
            f"{statuses} | {len(dataset['paired_comparison']['paired_run_ids'])} | "
            f"{metric_name} | {standard_text} | {linear_text} | {ratio_text} | {reference_text} |"
        )
    if aggregate.get("external_model_stages"):
        lines.extend(
            [
                "",
                "## Locally fitted external boosters",
                "",
                (
                    "All external rows use the same prepared folds, learning rate, maximum boosting "
                    "rounds, and validation patience as MBTR. Physical tree counts are reported "
                    "separately because independent-output methods build one tree per output per round."
                ),
                "",
                (
                    "| Dataset | Model | Completed | Metric | Score | / standard | Fit s | Predict s | "
                    "Retained rounds | Retained trees |"
                ),
                "|---|---|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in aggregate["datasets"]:
            expected = len(dataset["expected_run_ids"])
            for name in aggregate["external_model_stages"]:
                variant = dataset.get("external_variants", {}).get(name, {})
                timing = variant.get("timing", {})

                def format_value(value: Any, digits: int = 4) -> str:
                    return "—" if value is None else f"{float(value):.{digits}f}"

                lines.append(
                    f"| {dataset['dataset_id']} | {variant.get('display_name', name)} | "
                    f"{variant.get('completed_runs', 0)}/{expected} | {variant.get('headline_metric', '—')} | "
                    f"{format_value(variant.get('headline_mean'))} | "
                    f"{format_value(variant.get('over_standard_ratio'))} | "
                    f"{format_value(timing.get('fit_seconds', {}).get('median'), 3)} | "
                    f"{format_value(timing.get('predict_seconds', {}).get('median'), 3)} | "
                    f"{format_value(timing.get('rounds_retained', {}).get('median'), 1)} | "
                    f"{format_value(timing.get('effective_trees_retained', {}).get('median'), 1)} |"
                )
    lines.extend(
        [
            "",
            "A headline-loss ratio below 1 favors linear-response MBTR. The headline loss is "
            "global RMSE for regression and binary cross-entropy for multilabel classification. "
            "Comparisons include only run IDs for which both variants completed.",
            "Published values are shown only after the matching protocol is complete, and their "
            "catalogued comparison tier is retained. Context-only rows are archival and are never "
            "eligible for the published-best column.",
            "For SketchBoost rows, `direct` means that source rows and targets, split, aggregation, "
            "and metric match (subject to RF1's documented missing-value treatment difference). "
            "It does not mean tuning-budget parity: MBTR uses a frozen configuration and validation "
            "selects only its retained iteration, whereas the published baselines were hyperparameter-"
            "tuned. Sketch-width results selected by test loss are catalogued as context-only and "
            "excluded here.",
            "",
        ]
    )
    return "\n".join(lines)


def write_report(output_directory: Path, aggregate: dict[str, Any]) -> dict[str, Path]:
    """Write machine-readable and human-readable aggregate reports."""

    aggregate_path = output_directory / "aggregate.json"
    csv_path = output_directory / "summary.csv"
    markdown_path = output_directory / "report.md"
    _atomic_json(aggregate_path, aggregate)
    rows = _report_rows(aggregate)
    buffer = io.StringIO()
    fields = list(rows[0]) if rows else ["dataset_id"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(csv_path, buffer.getvalue())
    _atomic_text(markdown_path, _markdown_report(aggregate))
    _log_published_references(aggregate)
    return {"json": aggregate_path, "csv": csv_path, "markdown": markdown_path}


def _add_dataset_filter(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help="Restrict to a manifest dataset; repeat to select several.",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-estimators", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-bins", type=int, default=64)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--unbounded-depth", action="store_true")
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--linear-l2-regularization", type=float, default=0.1)
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=50,
        help="Retain the validation-best prefix after this many non-improving rounds (default: 50).",
    )
    stopping.add_argument(
        "--disable-early-stopping",
        action="store_const",
        dest="early_stopping_rounds",
        const=None,
        help="Run the fixed tree budget; this opts out of the validation-selected benchmark protocol.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--engine", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument(
        "--output-sketch-size",
        type=int,
        help="Use only the top-k gradient outputs for split search; leaves still use all outputs.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--importance-coverage", type=float, default=0.8)
    parser.add_argument("--linear-max-features", type=int, default=8)
    parser.add_argument("--linear-min-features", type=int, default=1)
    parser.add_argument("--linear-work-budget", type=float, default=2.0e11)
    parser.add_argument("--linear-time-budget-seconds", type=float, default=300.0)
    parser.add_argument("--linear-memory-budget-bytes", type=int, default=1024**3)
    parser.add_argument(
        "--linear-response-curvature",
        choices=("upper_bound", "exact"),
        default="upper_bound",
        help=(
            "Bernoulli curvature for linear-logistic leaves. upper_bound reuses one shared "
            "Gram matrix; exact solves a Hessian-weighted system per label."
        ),
    )
    parser.add_argument("--force-linear", action="store_true")


def _config_from_arguments(arguments: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        n_estimators=arguments.n_estimators,
        learning_rate=arguments.learning_rate,
        max_bins=arguments.max_bins,
        min_samples_leaf=arguments.min_samples_leaf,
        max_depth=None if arguments.unbounded_depth else arguments.max_depth,
        l2_regularization=arguments.l2_regularization,
        linear_l2_regularization=arguments.linear_l2_regularization,
        early_stopping_rounds=arguments.early_stopping_rounds,
        random_state=arguments.random_state,
        engine=arguments.engine,
        output_sketch_size=arguments.output_sketch_size,
        decision_threshold=arguments.decision_threshold,
        importance_coverage=arguments.importance_coverage,
        linear_max_features=arguments.linear_max_features,
        linear_min_features=arguments.linear_min_features,
        linear_work_budget=arguments.linear_work_budget,
        linear_time_budget_seconds=arguments.linear_time_budget_seconds,
        linear_memory_budget_bytes=arguments.linear_memory_budget_bytes,
        linear_response_curvature=arguments.linear_response_curvature,
        force_linear=arguments.force_linear,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="List every required prepared artifact.")
    plan_parser.add_argument("--data-directory", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path)
    _add_dataset_filter(plan_parser)

    run_parser = subparsers.add_parser("run", help="Run or resume the benchmark.")
    run_parser.add_argument("--data-directory", type=Path, required=True)
    run_parser.add_argument("--output-directory", type=Path, required=True)
    run_parser.add_argument("--allow-missing", action="store_true")
    run_parser.add_argument("--allow-missing-sidecars", action="store_true")
    run_parser.add_argument("--replace", action="store_true")
    run_parser.add_argument("--fail-fast", action="store_true")
    run_parser.add_argument(
        "--jobs",
        type=_positive_integer,
        default=1,
        help="Independent dataset/run work units to execute concurrently (default: 1).",
    )
    model_selection = run_parser.add_mutually_exclusive_group()
    model_selection.add_argument(
        "--model",
        action="append",
        dest="models",
        choices=MODEL_CLI_CHOICES,
        help=(
            "Run a selected model stage; repeat for several. Selecting linear-response also runs "
            "standard MBTR as its paired reference. The default is standard plus linear-response."
        ),
    )
    model_selection.add_argument(
        "--models-all",
        action="store_true",
        help=("Run MBTR standard/linear and all XGBoost, LightGBM, and CatBoost stages."),
    )
    run_parser.add_argument(
        "--native-threads",
        type=_positive_integer,
        default=1,
        help="Native threads used by each external booster (default: 1).",
    )
    run_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Progress logging level on stderr (default: INFO).",
    )
    _add_dataset_filter(run_parser)
    _add_model_arguments(run_parser)

    report_parser = subparsers.add_parser("report", help="Aggregate completed result stages.")
    report_parser.add_argument("--output-directory", type=Path, required=True)
    report_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Progress logging level on stderr (default: INFO).",
    )
    _add_dataset_filter(report_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.command in {"run", "report"}:
        _configure_logging(arguments.log_level, force=True)
    if arguments.command == "plan":
        plan = benchmark_plan(arguments.data_directory, arguments.dataset_ids)
        value = {
            "schema_version": 1,
            "work_units": len(plan),
            "available_splits": sum(bool(item["split_exists"]) for item in plan),
            "available_sidecars": sum(bool(item["sidecar_exists"]) for item in plan),
            "items": plan,
        }
        serialized = json.dumps(value, indent=2, sort_keys=True) + "\n"
        if arguments.output is None:
            sys.stdout.write(serialized)
        else:
            _atomic_text(arguments.output, serialized)
        return 0
    if arguments.command == "run":
        selected_models = _resolve_models(arguments.models, models_all=arguments.models_all)
        summary = run_benchmark(
            arguments.data_directory,
            arguments.output_directory,
            config=_config_from_arguments(arguments),
            dataset_ids=arguments.dataset_ids,
            allow_missing=arguments.allow_missing,
            require_sidecars=not arguments.allow_missing_sidecars,
            replace=arguments.replace,
            fail_fast=arguments.fail_fast,
            jobs=arguments.jobs,
            models=selected_models,
            native_threads=arguments.native_threads,
        )
        sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 1 if summary["failures"] else 0
    aggregate = aggregate_benchmark(
        arguments.output_directory,
        dataset_ids=arguments.dataset_ids,
    )
    paths = write_report(arguments.output_directory, aggregate)
    sys.stdout.write(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfig",
    "aggregate_benchmark",
    "benchmark_plan",
    "build_parser",
    "load_benchmark_manifest",
    "main",
    "run_benchmark",
    "run_work_unit",
    "write_report",
]
