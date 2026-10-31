"""Leakage-safe preparation of an official OpenML task split."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .provider import LoadedOpenMLTask

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration included in every prepared-fold fingerprint."""

    numeric_imputation: str = "mean"
    categorical_imputation: str = "most_frequent"
    categorical_encoding: str = "dense_one_hot_handle_unknown_ignore"
    max_categories: int = 256
    max_dense_bytes: int = 2 * 1024**3

    def validate(self) -> None:
        if self.numeric_imputation != "mean":
            raise ValueError("Only training-fold mean numeric imputation is supported.")
        if self.categorical_imputation != "most_frequent":
            raise ValueError("Only training-fold most-frequent categorical imputation is supported.")
        if self.max_categories < 2:
            raise ValueError("max_categories must be at least two.")
        if self.max_dense_bytes < 1:
            raise ValueError("max_dense_bytes must be positive.")


@dataclass(frozen=True)
class PreparedFold:
    """Paths and identity of a prepared fold artifact."""

    split_path: Path
    metadata_path: Path
    split_sha256: str
    fingerprint: str
    reused: bool


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def split_indices_sha256(train_indices: IntArray, test_indices: IntArray) -> str:
    digest = hashlib.sha256()
    for name, indices in ((b"train", train_indices), (b"test", test_indices)):
        values = np.ascontiguousarray(indices, dtype=np.int64)
        digest.update(name)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def validate_split_indices(
    train_indices: Any,
    test_indices: Any,
    *,
    n_rows: int,
) -> tuple[IntArray, IntArray]:
    train = np.asarray(train_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    for name, values in (("train", train), ("test", test)):
        if values.ndim != 1 or not len(values):
            raise ValueError(f"Official {name} indices must be a non-empty one-dimensional array.")
        if np.any(values < 0) or np.any(values >= n_rows):
            raise ValueError(f"Official {name} indices contain out-of-range row positions.")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Official {name} indices contain duplicates.")
    if np.intersect1d(train, test).size:
        raise ValueError("Official train and test indices overlap.")
    return np.ascontiguousarray(train), np.ascontiguousarray(test)


def _take_rows(value: Any, indices: IntArray) -> Any:
    iloc = getattr(value, "iloc", None)
    return iloc[indices] if iloc is not None else np.asarray(value)[indices]


def _to_numpy(value: Any) -> NDArray[Any]:
    converter = getattr(value, "to_numpy", None)
    array = converter() if converter is not None else np.asarray(value)
    return np.asarray(array)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(float(value)))
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        comparison = value != value
        return bool(comparison) if isinstance(comparison, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def _category_token(value: Any) -> str:
    kind = f"{type(value).__module__}.{type(value).__qualname__}"
    return f"{kind}:{value!s}"


def _display_value(value: Any) -> str:
    return f"{type(value).__name__}:{value!s}"


def _dtype_is_categorical(dtype: Any) -> bool:
    name = str(dtype).lower()
    kind = getattr(dtype, "kind", None)
    return kind in {"O", "U", "S", "b"} or any(
        marker in name for marker in ("category", "string", "object", "bool")
    )


def infer_categorical_mask(task: LoadedOpenMLTask) -> tuple[bool, ...]:
    mask = list(task.categorical_indicator)
    dtypes = getattr(task.x, "dtypes", None)
    if dtypes is not None:
        for index, dtype in enumerate(dtypes):
            mask[index] = mask[index] or _dtype_is_categorical(dtype)
    for index, name in enumerate(task.attribute_names):
        feature_type = task.feature_types.get(name, "").lower()
        if feature_type in {"nominal", "string", "categorical"}:
            mask[index] = True
        if feature_type in {"date", "datetime"}:
            raise ValueError(f"Feature {name!r} is a date; an explicit date encoding is required.")
    return tuple(mask)


def _numeric_column(values: NDArray[Any], *, name: str) -> FloatArray:
    result = np.empty(len(values), dtype=np.float64)
    for index, value in enumerate(values):
        if _is_missing(value):
            result[index] = np.nan
            continue
        try:
            result[index] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Feature {name!r} was marked numeric but contains non-numeric value {value!r}."
            ) from exc
    return result


def _encode_categorical_column(
    values: NDArray[Any],
    *,
    levels: list[str],
    mode: str,
    other: str | None,
) -> FloatArray:
    level_to_index = {level: index for index, level in enumerate(levels)}
    encoded = np.zeros((len(values), len(levels)), dtype=np.float64)
    for row, value in enumerate(values):
        candidate = mode if _is_missing(value) else _category_token(value)
        token = candidate if candidate in level_to_index else other
        if token is None:
            continue
        encoded[row, level_to_index[token]] = 1.0
    return encoded


def fit_transform_features(
    x_train: Any,
    x_test: Any,
    *,
    categorical_mask: tuple[bool, ...],
    attribute_names: tuple[str, ...],
    config: PreprocessingConfig,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Fit imputation/encoding only on ``x_train`` and transform both partitions."""

    config.validate()
    train = _to_numpy(x_train)
    test = _to_numpy(x_test)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("OpenML feature partitions must be two-dimensional with equal columns.")
    if train.shape[1] != len(categorical_mask) or train.shape[1] != len(attribute_names):
        raise ValueError("Feature matrix and metadata column counts disagree.")

    train_columns: list[FloatArray] = []
    test_columns: list[FloatArray] = []
    records: list[dict[str, Any]] = []
    output_names: list[str] = []
    for column, (name, categorical) in enumerate(zip(attribute_names, categorical_mask, strict=True)):
        training_values = train[:, column]
        test_values = test[:, column]
        if not categorical:
            training_numeric = _numeric_column(training_values, name=name)
            test_numeric = _numeric_column(test_values, name=name)
            observed = training_numeric[np.isfinite(training_numeric)]
            fill = 0.0 if not len(observed) else float(np.mean(observed))
            training_numeric = np.where(np.isfinite(training_numeric), training_numeric, fill)
            test_numeric = np.where(np.isfinite(test_numeric), test_numeric, fill)
            train_columns.append(training_numeric[:, None])
            test_columns.append(test_numeric[:, None])
            output_names.append(name)
            records.append(
                {
                    "input_feature": name,
                    "kind": "numeric",
                    "imputation_value": fill,
                    "training_missing": int(np.sum(~np.isfinite(_numeric_column(training_values, name=name)))),
                    "output_columns": [name],
                }
            )
            continue

        observed_values = [value for value in training_values if not _is_missing(value)]
        observed_tokens = [_category_token(value) for value in observed_values]
        counts = Counter(observed_tokens)
        if counts:
            mode = min(counts, key=lambda token: (-counts[token], token))
        else:
            mode = "__mbtr_all_missing__"
            counts[mode] = len(training_values)
        ranked = sorted(counts, key=lambda token: (-counts[token], token))
        truncated = len(ranked) > config.max_categories
        other: str | None = "__mbtr_other__" if truncated else None
        levels: list[str]
        if truncated:
            levels = ranked[: config.max_categories - 1] + ["__mbtr_other__"]
        else:
            levels = ranked
        level_to_index = {level: index for index, level in enumerate(levels)}
        train_encoded = _encode_categorical_column(
            training_values,
            levels=levels,
            mode=mode,
            other=other,
        )
        test_encoded = _encode_categorical_column(
            test_values,
            levels=levels,
            mode=mode,
            other=other,
        )
        train_columns.append(train_encoded)
        test_columns.append(test_encoded)
        encoded_names = [f"{name}=={level}" for level in levels]
        output_names.extend(encoded_names)
        display_by_token = {
            _category_token(value): _display_value(value)
            for value in observed_values
            if _category_token(value) in level_to_index
        }
        records.append(
            {
                "input_feature": name,
                "kind": "categorical",
                "training_missing": int(sum(_is_missing(value) for value in training_values)),
                "mode_token": mode,
                "mode_display": display_by_token.get(mode, mode),
                "levels": levels,
                "level_display": [display_by_token.get(level, level) for level in levels],
                "training_cardinality": len(ranked),
                "truncated": truncated,
                "unknown_policy": "other" if truncated else "all_zero",
                "output_columns": encoded_names,
            }
        )

    n_output_features = sum(column.shape[1] for column in train_columns)
    required_bytes = (len(train) + len(test)) * n_output_features * np.dtype(np.float64).itemsize
    if required_bytes > config.max_dense_bytes:
        raise MemoryError(
            f"Dense encoded fold needs about {required_bytes / 1024**2:.1f} MiB, above the configured "
            f"limit of {config.max_dense_bytes / 1024**2:.1f} MiB."
        )
    train_output = np.ascontiguousarray(np.column_stack(train_columns), dtype=np.float64)
    test_output = np.ascontiguousarray(np.column_stack(test_columns), dtype=np.float64)
    if not np.all(np.isfinite(train_output)) or not np.all(np.isfinite(test_output)):
        raise ValueError("Preprocessing produced non-finite features.")
    metadata = {
        "policy": asdict(config),
        "fit_partition": "outer_training_fold_only",
        "input_features": len(attribute_names),
        "output_features": n_output_features,
        "estimated_dense_bytes": required_bytes,
        "output_feature_names": output_names,
        "features": records,
    }
    return train_output, test_output, metadata


def _flatten_target(value: Any) -> NDArray[Any]:
    array = _to_numpy(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1:
        raise ValueError(f"OpenML scalar target must be one-dimensional; got {array.shape}.")
    if not len(array):
        raise ValueError("OpenML target must not be empty.")
    return array


def _target_token(value: Any) -> str:
    return _category_token(value)


def _label_matches(value: Any, declared: Any) -> bool:
    try:
        if bool(value == declared):
            return True
    except (TypeError, ValueError):
        pass
    return str(value) == str(declared)


def encode_binary_target(
    y_train: Any,
    y_test: Any,
    *,
    class_labels: tuple[Any, ...] | None,
) -> tuple[NDArray[np.int64], NDArray[np.int64], dict[str, Any]]:
    train = _flatten_target(y_train)
    test = _flatten_target(y_test)
    combined = list(train) + list(test)
    if any(_is_missing(value) for value in combined):
        raise ValueError("Classification targets contain missing values.")
    representatives: dict[str, Any] = {}
    for value in combined:
        representatives.setdefault(_target_token(value), value)
    if len(representatives) != 2:
        raise ValueError(f"Binary task has {len(representatives)} observed target classes, not two.")

    ordered_tokens: list[str] = []
    if class_labels is not None:
        if len(class_labels) != 2:
            raise ValueError(f"OpenML task declares {len(class_labels)} classes, not two.")
        for label in class_labels:
            matches = [token for token, value in representatives.items() if _label_matches(value, label)]
            if len(matches) != 1:
                raise ValueError(f"Could not map declared OpenML class label {label!r} exactly.")
            ordered_tokens.append(matches[0])
    else:
        categories = getattr(getattr(y_train, "dtype", None), "categories", None)
        if categories is not None:
            ordered_tokens = [
                _target_token(value)
                for value in categories
                if _target_token(value) in representatives
            ]
        if len(ordered_tokens) != 2:
            ordered_tokens = sorted(representatives)
    if len(set(ordered_tokens)) != 2:
        raise ValueError("Binary target class order is ambiguous.")
    mapping = {token: index for index, token in enumerate(ordered_tokens)}

    def transform(values: NDArray[Any]) -> NDArray[np.int64]:
        return np.asarray([mapping[_target_token(value)] for value in values], dtype=np.int64)

    encoded_train = transform(train)
    encoded_test = transform(test)
    if set(encoded_train.tolist()) != {0, 1}:
        raise ValueError("The outer training fold does not contain both binary classes.")
    metadata = {
        "encoding": "openml_declared_or_categorical_order",
        "negative_class": _display_value(representatives[ordered_tokens[0]]),
        "positive_class": _display_value(representatives[ordered_tokens[1]]),
        "mapping": [
            {
                "code": code,
                "token": token,
                "display": _display_value(representatives[token]),
            }
            for code, token in enumerate(ordered_tokens)
        ],
    }
    return encoded_train, encoded_test, metadata


def encode_regression_target(y_train: Any, y_test: Any) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    train_values = _flatten_target(y_train)
    test_values = _flatten_target(y_test)
    train = _numeric_column(train_values, name="target")
    test = _numeric_column(test_values, name="target")
    if not np.all(np.isfinite(train)) or not np.all(np.isfinite(test)):
        raise ValueError("Regression targets must be finite and cannot be imputed.")
    return np.ascontiguousarray(train), np.ascontiguousarray(test), {"encoding": "raw_float64"}


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_savez(path: Path, **arrays: NDArray[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
    os.replace(temporary, path)


def prepare_fold(
    task: LoadedOpenMLTask,
    *,
    train_indices: Any,
    test_indices: Any,
    task_kind: str,
    split_path: Path,
    run_id: str,
    repeat: int,
    fold: int,
    sample: int,
    config: PreprocessingConfig | None = None,
    force: bool = False,
) -> PreparedFold:
    """Prepare one official fold, using only its training rows to fit transforms."""

    settings = config or PreprocessingConfig()
    settings.validate()
    n_rows = len(task.x)
    if len(task.y) != n_rows:
        raise ValueError(f"Task {task.task_id} has different feature and target row counts.")
    train, test = validate_split_indices(train_indices, test_indices, n_rows=n_rows)
    index_hash = split_indices_sha256(train, test)
    task_metadata = task.metadata()
    metadata_path = split_path.with_suffix(".json")
    implementation_sha256 = file_sha256(Path(__file__))
    fingerprint = canonical_sha256(
        {
            "schema_version": 1,
            "task": task_metadata,
            "task_kind": task_kind,
            "run_id": run_id,
            "repeat": repeat,
            "fold": fold,
            "sample": sample,
            "indices_sha256": index_hash,
            "preprocessing": asdict(settings),
            "preparation_implementation_sha256": implementation_sha256,
        }
    )
    if split_path.exists() or metadata_path.exists():
        if split_path.exists() and metadata_path.exists():
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if previous.get("fingerprint") == fingerprint and previous.get("split_sha256") == file_sha256(
                split_path
            ):
                return PreparedFold(split_path, metadata_path, str(previous["split_sha256"]), fingerprint, True)
            if not force:
                raise ValueError(f"Prepared artifact {split_path} has a different fingerprint; pass --force.")

    x_train_source = _take_rows(task.x, train)
    x_test_source = _take_rows(task.x, test)
    y_train_source = _take_rows(task.y, train)
    y_test_source = _take_rows(task.y, test)
    categorical_mask = infer_categorical_mask(task)
    x_train, x_test, feature_metadata = fit_transform_features(
        x_train_source,
        x_test_source,
        categorical_mask=categorical_mask,
        attribute_names=task.attribute_names,
        config=settings,
    )
    y_train_output: NDArray[Any]
    y_test_output: NDArray[Any]
    if task_kind == "binary":
        y_train_output, y_test_output, target_metadata = encode_binary_target(
            y_train_source,
            y_test_source,
            class_labels=task.class_labels,
        )
    elif task_kind == "regression":
        y_train_output, y_test_output, target_metadata = encode_regression_target(
            y_train_source,
            y_test_source,
        )
    else:
        raise ValueError("OpenML pipeline task_kind must be 'binary' or 'regression'.")

    _atomic_savez(
        split_path,
        x_train=x_train,
        y_train=y_train_output,
        x_test=x_test,
        y_test=y_test_output,
        source_train_indices=train,
        source_test_indices=test,
    )
    split_hash = file_sha256(split_path)
    metadata = {
        "schema_version": 1,
        "artifact_type": "openml_prepared_fold",
        "fingerprint": fingerprint,
        "split_sha256": split_hash,
        "indices_sha256": index_hash,
        "preparation_implementation_sha256": implementation_sha256,
        "run_id": run_id,
        "repeat": repeat,
        "fold": fold,
        "sample": sample,
        "task_kind": task_kind,
        "task": task_metadata,
        "dimensions": {
            "train_rows": len(train),
            "test_rows": len(test),
            "input_features": len(task.attribute_names),
            "encoded_features": x_train.shape[1],
        },
        "preprocessing": feature_metadata,
        "target": target_metadata,
    }
    _atomic_write_json(metadata_path, metadata)
    return PreparedFold(split_path, metadata_path, split_hash, fingerprint, False)


__all__ = [
    "PreparedFold",
    "PreprocessingConfig",
    "canonical_sha256",
    "encode_binary_target",
    "file_sha256",
    "fit_transform_features",
    "infer_categorical_mask",
    "prepare_fold",
    "split_indices_sha256",
    "validate_split_indices",
]
