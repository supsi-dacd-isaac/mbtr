"""Run a shared-tree versus independent-output experiment on a prepared split."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import mbtr

from .adapters import make_estimator
from .catalog import find_protocol, find_published_scores
from .metrics import classification_metrics, regression_metrics


def _load_split(path: Path) -> dict[str, np.ndarray]:
    required = {"x_train", "y_train", "x_test", "y_test"}
    with np.load(path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Prepared split is missing arrays: {sorted(missing)}.")
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name, array in arrays.items():
        if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
            raise ValueError(f"Prepared array {name} must have a numeric or Boolean dtype.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Prepared array {name} must contain only finite values.")
    for partition in ("train", "test"):
        x = arrays[f"x_{partition}"]
        y = arrays[f"y_{partition}"]
        if x.ndim != 2 or y.ndim not in {1, 2} or not len(x) or not x.shape[1] or len(x) != len(y):
            raise ValueError(f"Invalid {partition} feature/target shapes: {x.shape}, {y.shape}.")
    if arrays["x_train"].shape[1] != arrays["x_test"].shape[1]:
        raise ValueError("Training and test data must have the same feature count.")
    training_outputs = 1 if arrays["y_train"].ndim == 1 else arrays["y_train"].shape[1]
    test_outputs = 1 if arrays["y_test"].ndim == 1 else arrays["y_test"].shape[1]
    if training_outputs != test_outputs:
        raise ValueError("Training and test data must have the same target count.")
    transform_keys = {"target_offset", "target_scale"}
    if transform_keys & set(arrays) and not transform_keys <= set(arrays):
        raise ValueError("Prepared splits must provide both target_offset and target_scale.")
    if transform_keys <= set(arrays):
        if arrays["target_offset"].shape != (training_outputs,) or arrays["target_scale"].shape != (training_outputs,):
            raise ValueError("target_offset and target_scale must contain one value per output.")
        if np.any(arrays["target_scale"] <= 0.0):
            raise ValueError("target_scale values must be strictly positive.")
    validation_keys = {"x_validation", "y_validation"}
    if validation_keys & set(arrays) and not validation_keys <= set(arrays):
        raise ValueError("Prepared splits must provide both x_validation and y_validation.")
    if validation_keys <= set(arrays):
        if (
            arrays["x_validation"].ndim != 2
            or arrays["y_validation"].ndim not in {1, 2}
            or not len(arrays["x_validation"])
            or not arrays["x_validation"].shape[1]
            or len(arrays["x_validation"]) != len(arrays["y_validation"])
        ):
            raise ValueError("Invalid validation feature/target shapes.")
        validation_outputs = 1 if arrays["y_validation"].ndim == 1 else arrays["y_validation"].shape[1]
        if arrays["x_validation"].shape[1] != arrays["x_train"].shape[1]:
            raise ValueError("Validation data must have the same feature count as training data.")
        if validation_outputs != training_outputs:
            raise ValueError("Validation data must have the same target count as training data.")
    return arrays


def _validate_task(arrays: dict[str, np.ndarray], task: str) -> None:
    target_names = [name for name in ("y_train", "y_validation", "y_test") if name in arrays]
    outputs = 1 if arrays["y_train"].ndim == 1 else arrays["y_train"].shape[1]
    if task == "binary" and outputs != 1:
        raise ValueError("The binary task requires exactly one target column.")
    if task == "multilabel" and (arrays["y_train"].ndim != 2 or outputs < 2):
        raise ValueError("The multilabel task requires a two-dimensional target with at least two labels.")
    if task in {"binary", "multilabel"}:
        if "target_offset" in arrays or "target_scale" in arrays:
            raise ValueError("target_offset and target_scale are supported only for regression.")
        for name in target_names:
            target = arrays[name]
            if np.any((target != 0) & (target != 1)):
                raise ValueError(f"Prepared array {name} must contain only binary values 0 and 1 for task {task}.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, str | bool | None]:
    repository = Path(__file__).resolve().parents[1]

    def run_git(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=repository,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return None
        return completed.stdout.strip() if completed.returncode == 0 else None

    revision = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {
        "revision": revision,
        "dirty": None if status is None else bool(status),
    }


def _restore_target_scale(arrays: dict[str, np.ndarray], target: np.ndarray) -> np.ndarray:
    if "target_offset" not in arrays:
        return target
    return target * arrays["target_scale"] + arrays["target_offset"]


def _resolved_engines(estimator: Any) -> list[str]:
    components = getattr(estimator, "estimators_", [estimator])
    return sorted({str(getattr(component, "engine_", "unknown")) for component in components})


def _positive_probability(estimator: Any, x: np.ndarray, *, task: str) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(x), dtype=np.float64)
    if task == "binary" and probability.ndim == 2 and probability.shape[1] == 2:
        return probability[:, 1]
    return probability


def run_variant(
    arrays: dict[str, np.ndarray],
    *,
    task: str,
    variant: str,
    parameters: dict[str, Any],
    stored_target_scale: str = "raw",
) -> dict[str, Any]:
    """Fit and evaluate one variant, returning JSON-compatible measurements."""

    _validate_task(arrays, task)
    estimator = make_estimator(task, variant, parameters)
    fit_kwargs: dict[str, Any] = {}
    if "x_validation" in arrays:
        fit_kwargs["eval_set"] = (arrays["x_validation"], arrays["y_validation"])
    started = perf_counter()
    estimator.fit(arrays["x_train"], arrays["y_train"], **fit_kwargs)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    if task == "regression":
        prediction = np.asarray(estimator.predict(arrays["x_test"]), dtype=np.float64)
        predict_seconds = perf_counter() - started
        started = perf_counter()
        metrics = regression_metrics(
            _restore_target_scale(arrays, arrays["y_train"]),
            _restore_target_scale(arrays, arrays["y_test"]),
            _restore_target_scale(arrays, prediction),
        )
    else:
        probability = _positive_probability(estimator, arrays["x_test"], task=task)
        predict_seconds = perf_counter() - started
        started = perf_counter()
        metrics = classification_metrics(
            arrays["y_test"],
            probability,
            threshold=float(parameters.get("decision_threshold", 0.5)),
        )
    metric_seconds = perf_counter() - started
    model_count = len(getattr(estimator, "estimators_", [estimator]))
    return {
        "variant": variant,
        "metrics": metrics,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "metric_seconds": metric_seconds,
        "serialized_model_bytes": len(pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)),
        "model_count": model_count,
        "resolved_engines": _resolved_engines(estimator),
        "metric_target_scale": "raw_inverse_transformed" if "target_offset" in arrays else stored_target_scale,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", type=Path, help="Prepared .npz split.")
    parser.add_argument("--dataset-id", required=True, help="Stable id from studies/catalog/datasets.json.")
    parser.add_argument("--protocol-id", required=True, help="Stable id from studies/catalog/protocols.json.")
    parser.add_argument("--task", choices=("regression", "binary", "multilabel"), required=True)
    parser.add_argument("--variant", choices=("shared", "independent", "both"), default="both")
    parser.add_argument("--run-id", help="Fold, repeat, or seed id required by multi-run catalog protocols.")
    parser.add_argument("--output", type=Path, help="Write JSON here instead of standard output.")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-bins", type=int, default=64)
    parser.add_argument("--min-samples-leaf", type=int, default=20)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--early-stopping-rounds", type=int)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--engine", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--stored-target-scale",
        default="raw",
        help="Label for regression targets as stored in the .npz (for example raw or train_standardized).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    arrays = _load_split(arguments.split)
    _validate_task(arrays, arguments.task)
    protocol = find_protocol(arguments.protocol_id)
    expected_runs = None if protocol is None else int(protocol["expected_runs"])
    expected_run_ids = None if protocol is None else [str(value) for value in protocol["expected_run_ids"]]
    run_axis = None if protocol is None else str(protocol["run_axis"])
    if expected_runs is not None and expected_runs > 1 and arguments.run_id is None:
        raise ValueError(f"Protocol {arguments.protocol_id} requires --run-id ({run_axis}).")
    run_id = "0" if arguments.run_id is None else arguments.run_id
    if expected_run_ids is not None and run_id not in expected_run_ids:
        raise ValueError(f"run-id {run_id!r} is not valid for protocol {arguments.protocol_id}: {expected_run_ids}.")
    random_state = arguments.random_state
    if random_state is None:
        random_state = int(run_id) if run_axis == "seed" else 42
    parameters = {
        "n_estimators": arguments.n_estimators,
        "learning_rate": arguments.learning_rate,
        "max_bins": arguments.max_bins,
        "min_samples_leaf": arguments.min_samples_leaf,
        "max_depth": arguments.max_depth,
        "l2_regularization": arguments.l2_regularization,
        "early_stopping_rounds": arguments.early_stopping_rounds,
        "random_state": random_state,
        "engine": arguments.engine,
    }
    if arguments.task != "regression":
        parameters["decision_threshold"] = arguments.decision_threshold
    variants = ("shared", "independent") if arguments.variant == "both" else (arguments.variant,)
    variant_results = [
        run_variant(
            arrays,
            task=arguments.task,
            variant=variant,
            parameters=parameters,
            stored_target_scale=arguments.stored_target_scale,
        )
        for variant in variants
    ]
    metric_names = {name for variant_result in variant_results for name in variant_result["metrics"]}
    published_references = [
        asdict(score)
        for score in find_published_scores(
            arguments.dataset_id,
            arguments.protocol_id,
            metric_names,
        )
    ]
    eligible_reference_count = sum(
        reference["comparison_tier"] in {"direct", "protocol_reference"} for reference in published_references
    )
    protocol_result_complete = expected_runs == 1
    if expected_runs is not None and expected_runs > 1:
        reference_message = "All protocol runs must be combined with studies.aggregate before comparison."
    elif published_references and not eligible_reference_count:
        reference_message = (
            "Published rows are context-only archival records and are not eligible for a clean comparison."
        )
    elif published_references:
        reference_message = "Single-run protocol with at least one comparison-eligible published reference."
    else:
        reference_message = "No comparison-eligible published reference is catalogued for these metrics."
    result = {
        "schema_version": 1,
        "dataset_id": arguments.dataset_id,
        "protocol_id": arguments.protocol_id,
        "run_id": run_id,
        "task": arguments.task,
        "split_path": str(arguments.split),
        "split_sha256": _file_sha256(arguments.split),
        "dimensions": {
            "train_rows": len(arrays["x_train"]),
            "test_rows": len(arrays["x_test"]),
            "validation_rows": len(arrays.get("x_validation", [])),
            "features": arrays["x_train"].shape[1],
            "outputs": 1 if arrays["y_train"].ndim == 1 else arrays["y_train"].shape[1],
        },
        "parameters": parameters,
        "metric_config": {
            "binary_cross_entropy_clip": 1e-7,
            "decision_threshold": arguments.decision_threshold if arguments.task != "regression" else None,
            "f1_zero_division": 0.0,
            "regression_target_scale": "raw_inverse_transformed"
            if "target_offset" in arrays
            else arguments.stored_target_scale,
        },
        "target_transform": None
        if "target_offset" not in arrays
        else {
            "offset": arrays["target_offset"].tolist(),
            "scale": arrays["target_scale"].tolist(),
        },
        "protocol_runs": {
            "expected": expected_runs,
            "expected_ids": expected_run_ids,
            "axis": run_axis,
            "comparison_ready": expected_runs == 1,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": _git_metadata(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "numba": _package_version("numba"),
            "mbtr": mbtr.__version__,
            "threads": {
                name: os.environ.get(name)
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
            },
        },
        "timing_note": (
            "Single in-process measurements; run variants separately with counterbalanced cold/warm order "
            "for speed claims."
        ),
        "results": variant_results,
        "published_references": published_references,
        "published_reference_status": {
            "comparison_ready": protocol_result_complete and bool(eligible_reference_count),
            "comparison_eligible_reference_count": eligible_reference_count,
            "context_only_reference_count": sum(
                reference["comparison_tier"] == "context_only" for reference in published_references
            ),
            "message": reference_message,
        },
    }
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is None:
        sys.stdout.write(serialized)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
