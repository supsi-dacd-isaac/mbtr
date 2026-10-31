"""Shared, auditable writers for prepared multivariate benchmark splits."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PreparedRun:
    """One protocol-defined run before it is serialized to the NPZ contract."""

    dataset_id: str
    protocol_id: str
    run_id: str
    task: str
    arrays: Mapping[str, NDArray[Any]]
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    source: Mapping[str, Any]
    preprocessing: Mapping[str, Any]
    split: Mapping[str, Any]


@dataclass(frozen=True)
class WrittenRun:
    """Paths and identity of one materialized prepared run."""

    split_path: Path
    sidecar_path: Path
    split_sha256: str
    preparation_fingerprint: str
    reused: bool


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-compatible value using a stable canonical representation."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: NDArray[Any]) -> str:
    """Hash shape, dtype, and C-order values without relying on NPZ metadata."""

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def indices_sha256(**partitions: NDArray[Any]) -> str:
    """Hash named index arrays in their declared insertion order."""

    digest = hashlib.sha256()
    for name, values in partitions.items():
        array = np.ascontiguousarray(values, dtype=np.int64)
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def preparation_environment() -> dict[str, Any]:
    """Record the libraries and preparer source that can affect artifacts."""

    package_versions: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "torch"):
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None
    source_files = sorted(Path(__file__).parent.glob("prepare*.py"))
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "packages": package_versions,
        "preparer_source_sha256": {path.name: file_sha256(path) for path in source_files},
    }


def _target_width(value: NDArray[Any]) -> int:
    return 1 if value.ndim == 1 else int(value.shape[1])


def validate_prepared_run(run: PreparedRun) -> dict[str, NDArray[Any]]:
    """Validate and normalize an adapter result before writing it."""

    if run.task not in {"regression", "multilabel"}:
        raise ValueError(f"Unsupported prepared task {run.task!r}.")
    required = {"x_train", "y_train", "x_test", "y_test"}
    missing = required - set(run.arrays)
    if missing:
        raise ValueError(f"Prepared run is missing arrays: {sorted(missing)}.")

    arrays: dict[str, NDArray[Any]] = {}
    for name, value in run.arrays.items():
        array = np.ascontiguousarray(value)
        if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
            raise ValueError(f"Prepared array {name} must be numeric or Boolean, got {array.dtype}.")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Prepared array {name} contains non-finite values.")
        arrays[name] = array

    partition_names = ["train", "test"]
    if {"x_validation", "y_validation"} & set(arrays):
        if not {"x_validation", "y_validation"} <= set(arrays):
            raise ValueError("Validation features and targets must be provided together.")
        partition_names.append("validation")
    if {"x_calibration", "y_calibration"} & set(arrays):
        if not {"x_calibration", "y_calibration"} <= set(arrays):
            raise ValueError("Calibration features and targets must be provided together.")
        partition_names.append("calibration")

    n_features: int | None = None
    n_outputs: int | None = None
    for partition in partition_names:
        x = arrays[f"x_{partition}"]
        y = arrays[f"y_{partition}"]
        if x.ndim != 2 or y.ndim not in {1, 2} or not len(x) or not x.shape[1] or len(x) != len(y):
            raise ValueError(f"Invalid {partition} feature/target shapes: {x.shape}, {y.shape}.")
        n_features = x.shape[1] if n_features is None else n_features
        n_outputs = _target_width(y) if n_outputs is None else n_outputs
        if x.shape[1] != n_features or _target_width(y) != n_outputs:
            raise ValueError("Every partition must have the same feature and target widths.")

    assert n_features is not None and n_outputs is not None
    if len(run.feature_names) != n_features:
        raise ValueError(f"Expected {n_features} feature names, got {len(run.feature_names)}.")
    if len(set(run.feature_names)) != len(run.feature_names):
        raise ValueError("Prepared feature names must be unique.")
    if len(run.target_names) != n_outputs:
        raise ValueError(f"Expected {n_outputs} target names, got {len(run.target_names)}.")
    if len(set(run.target_names)) != len(run.target_names):
        raise ValueError("Prepared target names must be unique.")

    if run.task == "multilabel":
        if n_outputs < 2:
            raise ValueError("Multilabel data must contain at least two target columns.")
        if "target_offset" in arrays or "target_scale" in arrays:
            raise ValueError("Classification splits cannot contain target scaling arrays.")
        for partition in partition_names:
            target = arrays[f"y_{partition}"]
            if np.any((target != 0) & (target != 1)):
                raise ValueError(f"Multilabel targets in {partition} must contain only 0 and 1.")

    transform_keys = {"target_offset", "target_scale"}
    if transform_keys & set(arrays) and not transform_keys <= set(arrays):
        raise ValueError("target_offset and target_scale must be provided together.")
    if transform_keys <= set(arrays):
        if arrays["target_offset"].shape != (n_outputs,) or arrays["target_scale"].shape != (n_outputs,):
            raise ValueError("Target scaling arrays must contain one value per target.")
        if np.any(arrays["target_scale"] <= 0.0):
            raise ValueError("Target scales must be strictly positive.")

    source_indices: list[tuple[str, NDArray[Any]]] = []
    for name, array in arrays.items():
        if not name.startswith("source_index_"):
            continue
        partition = name.removeprefix("source_index_")
        feature_name = f"x_{partition}"
        if feature_name not in arrays:
            raise ValueError(f"{name} has no matching {feature_name} partition.")
        if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array.")
        if len(array) != len(arrays[feature_name]):
            raise ValueError(f"{name} must contain one source position per {partition} row.")
        if np.any(array < 0) or len(np.unique(array)) != len(array):
            raise ValueError(f"{name} must contain unique non-negative source positions.")
        source_indices.append((name, array))
    for index, (left_name, left) in enumerate(source_indices):
        for right_name, right in source_indices[index + 1 :]:
            if np.intersect1d(left, right).size:
                raise ValueError(f"{left_name} and {right_name} contain overlapping source positions.")
    return arrays


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **dict(arrays))  # type: ignore[arg-type]
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fingerprint_document(
    run: PreparedRun,
    array_hashes: Mapping[str, str],
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "dataset_id": run.dataset_id,
        "protocol_id": run.protocol_id,
        "run_id": str(run.run_id),
        "task": run.task,
        "feature_names": list(run.feature_names),
        "target_names": list(run.target_names),
        "source": run.source,
        "preprocessing": run.preprocessing,
        "split": run.split,
        "environment": environment,
        "array_sha256": dict(array_hashes),
    }


def _fingerprint(
    run: PreparedRun,
    arrays: Mapping[str, NDArray[Any]],
    environment: Mapping[str, Any],
) -> tuple[str, dict[str, str]]:
    array_hashes = {name: array_sha256(value) for name, value in sorted(arrays.items())}
    document = _fingerprint_document(run, array_hashes, environment)
    return canonical_sha256(document), array_hashes


def _existing_sidecar_is_valid(
    existing: Mapping[str, Any],
    *,
    split_path: Path,
    split_sha256: str,
    expected_fingerprint: str,
) -> bool:
    required = {
        "schema_version",
        "dataset_id",
        "protocol_id",
        "run_id",
        "task",
        "feature_names",
        "target_names",
        "source",
        "preprocessing",
        "split",
        "environment",
        "array_sha256",
    }
    if not required <= set(existing) or existing.get("split_sha256") != split_sha256:
        return False
    fingerprint_document = {name: existing[name] for name in required}
    stored_fingerprint = existing.get("preparation_fingerprint")
    if canonical_sha256(fingerprint_document) != stored_fingerprint or stored_fingerprint != expected_fingerprint:
        return False
    declared_hashes = existing.get("array_sha256")
    if not isinstance(declared_hashes, dict):
        return False
    try:
        with np.load(split_path, allow_pickle=False) as archive:
            if set(archive.files) != set(declared_hashes):
                return False
            observed_hashes = {name: array_sha256(np.asarray(archive[name])) for name in archive.files}
    except (OSError, ValueError):
        return False
    return observed_hashes == declared_hashes


def write_prepared_run(run: PreparedRun, output_directory: Path, *, replace: bool = False) -> WrittenRun:
    """Atomically materialize one prepared split and its provenance sidecar."""

    arrays = validate_prepared_run(run)
    environment = preparation_environment()
    fingerprint, array_hashes = _fingerprint(run, arrays, environment)
    split_path = output_directory / run.protocol_id / run.dataset_id / f"{run.run_id}.npz"
    sidecar_path = split_path.with_suffix(".json")

    if split_path.exists() or sidecar_path.exists():
        if split_path.is_file() and sidecar_path.is_file():
            existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
            existing_sha256 = file_sha256(split_path)
            if _existing_sidecar_is_valid(
                existing,
                split_path=split_path,
                split_sha256=existing_sha256,
                expected_fingerprint=fingerprint,
            ):
                return WrittenRun(split_path, sidecar_path, existing_sha256, fingerprint, True)
        if not replace:
            raise FileExistsError(
                f"Prepared output for {run.dataset_id}/{run.run_id} differs or is incomplete; "
                "pass --replace only after reviewing the changed source or protocol."
            )

    _atomic_npz(split_path, arrays)
    split_sha256 = file_sha256(split_path)
    sidecar = {
        "schema_version": 1,
        "artifact_type": "mbtr_multivariate_prepared_split",
        "created_at_utc": utc_now(),
        "dataset_id": run.dataset_id,
        "protocol_id": run.protocol_id,
        "run_id": str(run.run_id),
        "task": run.task,
        "split_sha256": split_sha256,
        "preparation_fingerprint": fingerprint,
        "array_sha256": array_hashes,
        "feature_names": list(run.feature_names),
        "target_names": list(run.target_names),
        "dimensions": {
            "train_rows": int(len(arrays["x_train"])),
            "validation_rows": int(len(arrays.get("x_validation", []))),
            "calibration_rows": int(len(arrays.get("x_calibration", []))),
            "test_rows": int(len(arrays["x_test"])),
            "features": int(arrays["x_train"].shape[1]),
            "outputs": _target_width(arrays["y_train"]),
        },
        "source": dict(run.source),
        "preprocessing": dict(run.preprocessing),
        "split": dict(run.split),
        "environment": environment,
    }
    _atomic_json(sidecar_path, sidecar)
    return WrittenRun(split_path, sidecar_path, split_sha256, fingerprint, False)
