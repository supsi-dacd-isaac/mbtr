"""Exact preparation adapter for the pinned MOCR point-prediction corpus."""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .prepare_common import PreparedRun, file_sha256, indices_sha256

MOCR_REPOSITORY = "https://github.com/Vekteur/multi-output-conformal-regression"
MOCR_REVISION = "6d40a669492ef1c7860568315796a99a895d9548"
PROTOCOL_ID = "mocr_point_prediction_2025"
EXPECTED_TORCH_VERSION = "2.6.0"
EXPECTED_NUMPY_VERSION = "2.2.2"
EXPECTED_PANDAS_VERSION = "2.2.3"


@dataclass(frozen=True)
class MOCRDataset:
    dataset_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    targets: tuple[str | int, ...]
    read_kind: str
    expected_rows: int
    expected_features: int


DATASETS: dict[str, MOCRDataset] = {
    "mocr_households": MOCRDataset(
        "mocr_households",
        "data/camehl/households.csv",
        "8ee416404862dc199c7247ba207dd60c444eb37e9d4dfe76ea55319fa4607359",
        805_441,
        ("inc", "food", "house", "utili"),
        "households",
        7_207,
        14,
    ),
    "mocr_house": MOCRDataset(
        "mocr_house",
        "data/feldman/house.csv",
        "d0875baa0251b21d4bdc9d2ae940a4fe0bb6009824f23dd0e2a5b2bf04557b7e",
        2_515_206,
        ("lat", "price"),
        "header",
        21_613,
        17,
    ),
    "mocr_bio": MOCRDataset(
        "mocr_bio",
        "data/feldman/bio.csv",
        "4277cfcb4e91a181746cbc654f001b57951c9e6a80f4f795fdb5c807e0848f40",
        3_528_710,
        ("F7", "F9"),
        "header",
        45_730,
        8,
    ),
    "mocr_blog_data": MOCRDataset(
        "mocr_blog_data",
        "data/feldman/blog_data.csv",
        "f3cc953ea9913361be6d3bae2c393eef73eab57b1cb8e5bf8a22f1e5eb5dd05f",
        65_095_322,
        (60, 280),
        "no_header",
        52_397,
        269,
    ),
    "mocr_taxi": MOCRDataset(
        "mocr_taxi",
        "data/wang/taxi.csv",
        "7045c7f835fdf978a2f0e677ed2029ac17001b548c34ad95242ffacd9b101917",
        9_338_146,
        (-2, -1),
        "last_two",
        61_286,
        4,
    ),
}


def _read_frame(path: Path, specification: MOCRDataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    if specification.read_kind == "households":
        frame = pd.read_csv(path, index_col=0).drop(columns=["newid", "inc.a"])
        targets = list(specification.targets)
        return frame[frame.columns.difference(targets)], frame[targets]
    if specification.read_kind == "no_header":
        frame = pd.read_csv(path, header=None)
        targets = list(specification.targets)
        return frame[frame.columns.difference(targets)], frame[targets]
    frame = pd.read_csv(path)
    if specification.read_kind == "last_two":
        return frame.iloc[:, :-2], frame.iloc[:, -2:]
    targets = list(specification.targets)
    return frame[frame.columns.difference(targets)], frame[targets]


def _preprocess_like_upstream(
    x: pd.DataFrame,
    y: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray, dict[str, Any]]:
    """Mirror the pinned preprocessing, including its overwritten-mask behavior."""

    input_columns = list(x.columns)
    categorical = (x.dtypes == "object") | (x.dtypes == "bool")
    unique_counts = x.nunique()
    categorical |= (x.dtypes == "int64") & (unique_counts <= 10)
    categorical |= unique_counts == 2

    high_cardinality = categorical & (unique_counts > 20)
    high_cardinality_columns = list(x.columns[high_cardinality])
    x = x.drop(columns=high_cardinality_columns)
    categorical = categorical[~high_cardinality]

    pseudo_categorical = (~categorical) & (x.nunique() < 10)
    pseudo_categorical_columns = list(x.columns[pseudo_categorical])
    x = x.drop(columns=pseudo_categorical_columns)

    missing_columns_mask = pd.isnull(x).mean(axis=0) > 0.2
    missing_columns = list(x.columns[missing_columns_mask])
    x = x.drop(columns=missing_columns)
    missing_rows = pd.isnull(x).any(axis=1)
    kept_source_positions = np.flatnonzero(~missing_rows.to_numpy()).astype(np.int64)
    x = x.loc[~missing_rows]
    y = y.loc[~missing_rows]

    encoded = pd.get_dummies(x)
    x_array = encoded.to_numpy(dtype="float32")
    y_array = y.to_numpy(dtype="float32")
    if len(x_array) < 100:
        raise ValueError("MOCR preprocessing left fewer than 100 rows.")
    if not np.all(np.isfinite(x_array)) or not np.all(np.isfinite(y_array)):
        raise ValueError("MOCR preprocessing produced non-finite values.")
    metadata = {
        "implementation": "literal port of pinned moc/datamodules/preprocessing.py",
        "categorical_detection": (
            "object or bool; int64 with <=10 unique values; or any binary column. "
            "The pinned function overwrites an optional caller mask."
        ),
        "high_cardinality_threshold": 20,
        "pseudo_categorical_min_unique": 10,
        "missing_column_threshold": 0.2,
        "missing_row_policy": "drop rows with any remaining feature missing",
        "encoding": "pandas.get_dummies",
        "input_columns": [str(value) for value in input_columns],
        "removed_high_cardinality_columns": [str(value) for value in high_cardinality_columns],
        "removed_pseudo_categorical_columns": [str(value) for value in pseudo_categorical_columns],
        "removed_missing_columns": [str(value) for value in missing_columns],
        "removed_row_count": int(np.sum(missing_rows)),
        "output_feature_names": [str(value) for value in encoded.columns],
    }
    return (
        x_array,
        y_array,
        tuple(str(value) for value in encoded.columns),
        tuple(str(value) for value in y.columns),
        kept_source_positions,
        metadata,
    )


def _torch_module() -> Any:
    if np.__version__ != EXPECTED_NUMPY_VERSION or pd.__version__ != EXPECTED_PANDAS_VERSION:
        raise RuntimeError(
            "Exact MOCR preparation requires "
            f"numpy=={EXPECTED_NUMPY_VERSION} and pandas=={EXPECTED_PANDAS_VERSION}; "
            f"got numpy=={np.__version__} and pandas=={pd.__version__}."
        )
    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment-dependent message
        raise RuntimeError(
            f"Exact MOCR preparation requires torch=={EXPECTED_TORCH_VERSION}; "
            "install the project's 'mocr-study' optional dependency."
        ) from error
    if torch.__version__.split("+")[0] != EXPECTED_TORCH_VERSION:
        raise RuntimeError(
            f"Exact MOCR preparation requires torch=={EXPECTED_TORCH_VERSION}, got {torch.__version__}."
        )
    return torch


def _split_sizes(total: int) -> tuple[int, int, int, int]:
    sizes = np.asarray((0.4, 0.1, 0.3, 0.2), dtype=np.float64) * total
    calibration_index = 2
    excess = max(0.0, float(sizes[calibration_index] - 2048))
    sizes[calibration_index] -= excess
    recipients = (sizes != 0) & (np.arange(len(sizes)) != calibration_index)
    sizes[recipients] += excess / int(np.sum(recipients))
    integer_sizes = sizes.astype(int)
    integer_sizes[-1] = total - int(integer_sizes[:-1].sum())
    return tuple(int(value) for value in integer_sizes)  # type: ignore[return-value]


def _standardize(
    x: np.ndarray,
    y: np.ndarray,
    partitions: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    torch = _torch_module()
    x_tensor = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    y_tensor = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))
    train = torch.as_tensor(partitions["train"], dtype=torch.int64)
    x_mean = torch.mean(x_tensor[train], dim=0)
    x_scale = torch.std(x_tensor[train], dim=0) + 1e-7
    y_mean = torch.mean(y_tensor[train], dim=0)
    y_scale = torch.std(y_tensor[train], dim=0) + 1e-7
    arrays: dict[str, np.ndarray] = {}
    for name, positions in partitions.items():
        index = torch.as_tensor(positions, dtype=torch.int64)
        arrays[f"x_{name}"] = ((x_tensor[index] - x_mean) / x_scale).numpy()
        arrays[f"y_{name}"] = ((y_tensor[index] - y_mean) / y_scale).numpy()
    return arrays, x_mean.numpy(), x_scale.numpy(), y_mean.numpy(), y_scale.numpy()


def prepare_mocr_runs(
    source_root: Path,
    dataset_ids: list[str] | None = None,
) -> Iterator[PreparedRun]:
    """Prepare the five public datasets bundled at the pinned MOCR revision."""

    requested = list(DATASETS) if dataset_ids is None else dataset_ids
    unknown = sorted(set(requested) - set(DATASETS))
    if unknown:
        raise ValueError(f"MOCR datasets are unavailable from the public pinned checkout: {unknown}.")
    torch = _torch_module()
    for dataset_id in requested:
        specification = DATASETS[dataset_id]
        raw_path = source_root / specification.relative_path
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing pinned MOCR source file: {raw_path}")
        if raw_path.stat().st_size != specification.size_bytes or file_sha256(raw_path) != specification.sha256:
            raise ValueError(f"Pinned MOCR source checksum/size mismatch: {raw_path}")
        x_frame, y_frame = _read_frame(raw_path, specification)
        x, y, feature_names, target_names, kept_positions, preprocessing = _preprocess_like_upstream(
            x_frame, y_frame
        )
        if len(x) != specification.expected_rows or x.shape[1] != specification.expected_features:
            raise ValueError(
                f"Pinned MOCR dimensions changed for {dataset_id}: got {x.shape}, expected "
                f"({specification.expected_rows}, {specification.expected_features})."
            )

        for run_id in range(10):
            source_seed = 2000 + run_id
            sample_size = min(len(x), math.ceil(20_000 / 0.4))
            subsample = np.random.RandomState(source_seed).choice(len(x), sample_size, replace=False)
            sizes = _split_sizes(sample_size)
            permutation = torch.randperm(
                sample_size,
                generator=torch.Generator().manual_seed(source_seed),
            ).numpy()
            train_end = sizes[0]
            validation_end = train_end + sizes[1]
            calibration_end = validation_end + sizes[2]
            relative_partitions = {
                "train": permutation[:train_end],
                "validation": permutation[train_end:validation_end],
                "calibration": permutation[validation_end:calibration_end],
                "test": permutation[calibration_end:],
            }
            sampled_x = x[subsample]
            sampled_y = y[subsample]
            arrays, x_mean, x_scale, y_mean, y_scale = _standardize(
                sampled_x,
                sampled_y,
                relative_partitions,
            )
            arrays.update(
                {
                    "target_offset": y_mean,
                    "target_scale": y_scale,
                    "source_index_train": kept_positions[subsample[relative_partitions["train"]]],
                    "source_index_validation": kept_positions[subsample[relative_partitions["validation"]]],
                    "source_index_calibration": kept_positions[subsample[relative_partitions["calibration"]]],
                    "source_index_test": kept_positions[subsample[relative_partitions["test"]]],
                }
            )
            original_indices = {
                name: arrays[f"source_index_{name}"] for name in ("train", "validation", "calibration", "test")
            }
            yield (
                PreparedRun(
                    dataset_id=dataset_id,
                    protocol_id=PROTOCOL_ID,
                    run_id=str(run_id),
                    task="regression",
                    arrays=arrays,
                    feature_names=feature_names,
                    target_names=target_names,
                    source={
                        "repository_url": MOCR_REPOSITORY,
                        "repository_revision": MOCR_REVISION,
                        "relative_path": specification.relative_path,
                        "raw_sha256": specification.sha256,
                        "raw_size_bytes": specification.size_bytes,
                        "license": "MIT (upstream repository); verify dataset-specific terms before redistribution",
                    },
                    preprocessing={
                        **preprocessing,
                        "numeric_dtype": "float32",
                        "normalization": "torch training-partition mean/std with correction=1 and epsilon=1e-7",
                        "x_offset": x_mean.tolist(),
                        "x_scale": x_scale.tolist(),
                        "target_offset": y_mean.tolist(),
                        "target_scale": y_scale.tolist(),
                        "fit_partition": "training only",
                    },
                    split={
                        "upstream_run_id": run_id,
                        "source_seed": source_seed,
                        "torch_version": torch.__version__,
                        "torch_num_threads": torch.get_num_threads(),
                        "torch_num_interop_threads": torch.get_num_interop_threads(),
                        "subsample_cap": 50_000,
                        "subsample_size": sample_size,
                        "source_index_space": (
                            "zero-based positional CSV data rows before preprocessing; not pandas index labels "
                            "or positions within the seed-specific subsample"
                        ),
                        "subsample_indices_sha256": indices_sha256(subsample=subsample),
                        "partition_sizes": dict(
                            zip(("train", "validation", "calibration", "test"), sizes, strict=True)
                        ),
                        "source_indices_sha256": indices_sha256(**original_indices),
                        "ratios_before_calibration_cap": [0.4, 0.1, 0.3, 0.2],
                        "calibration_cap": 2048,
                    },
                )
            )
