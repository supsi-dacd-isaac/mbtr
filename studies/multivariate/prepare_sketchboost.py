"""Preparation adapter for the datasets used by the pinned SketchBoost code."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from .prepare_common import PreparedRun, file_sha256, indices_sha256

SKETCHBOOST_REPOSITORY = "https://github.com/sb-ai-lab/sketchboost-paper"
SKETCHBOOST_REVISION = "e479fde7a0dd5f624fd9efd654e2d192a12d325d"
RANDOM_PROTOCOL_ID = "sketchboost_random_holdout_cv5_2022"
OFFICIAL_PROTOCOL_ID = "sketchboost_official_holdout_cv5_2022"


@dataclass(frozen=True)
class DownloadSpec:
    dataset_id: str
    filename: str
    url: str
    size_bytes: int
    sha256: str


DOWNLOADS: dict[str, DownloadSpec] = {
    "rf1": DownloadSpec(
        "rf1",
        "rf1.csv",
        "https://www.openml.org/data/get_csv/21230440",
        3_018_094,
        "ac09ce9f85f9f6f690711a04cc77890cbcc633c666b1092bc950b1c6e7ba90a1",
    ),
    "scm20d": DownloadSpec(
        "scm20d",
        "scm20d.csv",
        "https://www.openml.org/data/get_csv/21230443",
        3_721_510,
        "f8fb8bd6d47f79150d4d7dd8da85f66d51ff862f8ad1c7b26f242a8a45466967",
    ),
    "delicious": DownloadSpec(
        "delicious",
        "Delicious.zip",
        "https://drive.usercontent.google.com/download?id=0B3lPMIHmG6vGdG1jZ19VS2NWRVU&export=download&confirm=t&resourcekey=0-viF3jLCBFuBSDjdlKL_32g",
        1_326_261,
        "9a4e79e9bd1835225e59eef09ba0ee29c5430f785045a9245943511836542806",
    ),
    "mediamill": DownloadSpec(
        "mediamill",
        "Mediamill.zip",
        "https://drive.usercontent.google.com/download?id=0B3lPMIHmG6vGY3B4TXRmZnZBTkk&export=download&confirm=t&resourcekey=0-Hv7nA6NTsIJshWk2gJwTYw",
        24_641_162,
        "5fee2e460f3a684cd0c0cf115ff0629d80fdacd0581f950972d21a1caa8d05f3",
    ),
}

REGRESSION_TARGETS: dict[str, tuple[str, ...]] = {
    "rf1": (
        "CHSI2_48H__0",
        "NASI2_48H__0",
        "EADM7_48H__0",
        "SCLM7_48H__0",
        "CLKM7_48H__0",
        "VALI2_48H__0",
        "NAPM7_48H__0",
        "DLDI4_48H__0",
    ),
    "scm20d": ("LBL", *(f"MTLp{index}A" for index in range(2, 17))),
}

XC_MEMBERS: dict[str, dict[str, str]] = {
    "delicious": {
        "Delicious/Delicious_data.txt": "7c6732b0ba3e2acb8b6f8cd147b83107f798e50fedf47c3429f549d53d971b9b",
        "Delicious/delicious_trSplit.txt": "8dbe2f693e62ae6e45ee617db8372fd5f547f4bb2f59e17c47f55c97b7977d1e",
        "Delicious/delicious_tstSplit.txt": "aac14f9f84e6fe3f77d81db8ceb04be935e72adf22d9284d78d58d7763f242fa",
    },
    "mediamill": {
        "Mediamill/Mediamill_data.txt": "0f6bb03ec9daa8762314d89c33df8dd5432778639ac32415b276be0efdc2dd7d",
        "Mediamill/mediamill_trSplit.txt": "6d15ad4cefd13267d438d85eed0ac63c784ac300734cc48167c8a51fdfa8b210",
        "Mediamill/mediamill_tstSplit.txt": "e36d75b0110ccfc174e7f3b9a4f3419b73b1dc9e1432f636a1049c898400b192",
    },
}

XC_DIMENSIONS = {
    "delicious": (16_105, 500, 983, 12_920, 3_185),
    "mediamill": (43_907, 120, 101, 30_993, 12_914),
}


def validate_download(path: Path, specification: DownloadSpec) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing source file: {path}")
    if path.stat().st_size != specification.size_bytes:
        raise ValueError(
            f"Source size mismatch for {specification.dataset_id}: "
            f"{path.stat().st_size} != {specification.size_bytes}."
        )
    digest = file_sha256(path)
    if digest != specification.sha256:
        raise ValueError(f"Source checksum mismatch for {specification.dataset_id}: {digest}.")


def _fit_mean_imputation(
    x_train: np.ndarray,
    *others: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    """Fit the RF1-only finite-value bridge on the current fold's fit rows."""

    means = np.nanmean(x_train, axis=0, dtype=np.float64).astype(np.float32)
    if not np.all(np.isfinite(means)):
        raise ValueError("At least one feature is entirely missing in the training fold.")

    def transform(value: np.ndarray) -> np.ndarray:
        output = np.asarray(value, dtype=np.float32).copy()
        missing_rows, missing_columns = np.where(~np.isfinite(output))
        output[missing_rows, missing_columns] = means[missing_columns]
        return np.ascontiguousarray(output)

    return (transform(x_train), *(transform(value) for value in others)), means


def _regression_runs(dataset_id: str, path: Path) -> list[PreparedRun]:
    specification = DOWNLOADS[dataset_id]
    validate_download(path, specification)
    frame = pd.read_csv(path, na_values="?")
    targets = REGRESSION_TARGETS[dataset_id]
    if not set(targets) <= set(frame.columns):
        raise ValueError(f"The {dataset_id} source is missing declared target columns.")
    feature_names = tuple(str(name) for name in frame.columns if name not in targets)
    x = frame.loc[:, list(feature_names)].to_numpy(dtype=np.float32)
    y = frame.loc[:, list(targets)].to_numpy(dtype=np.float32)
    missing_cells = int(np.sum(~np.isfinite(x)))
    expected = (9_125, 64, 8) if dataset_id == "rf1" else (8_966, 61, 16)
    if (len(x), x.shape[1], y.shape[1]) != expected or not np.all(np.isfinite(y)):
        raise ValueError(f"Unexpected {dataset_id} source dimensions or non-finite targets.")

    all_indices = np.arange(len(x), dtype=np.int64)
    pool_indices, test_indices = train_test_split(all_indices, test_size=0.2, random_state=42)
    runs: list[PreparedRun] = []
    for fold, (fit_positions, validation_positions) in enumerate(
        KFold(5, shuffle=True, random_state=42).split(pool_indices)
    ):
        train_indices = pool_indices[fit_positions]
        validation_indices = pool_indices[validation_positions]
        transformed, means = _fit_mean_imputation(
            x[train_indices],
            x[validation_indices],
            x[test_indices],
        )
        x_train, x_validation, x_test = transformed
        preprocessing: dict[str, object] = {
            "source_dtype": "float32 as in pinned SketchBoost preprocess.py",
            "missing_cells": missing_cells,
        }
        if missing_cells:
            preprocessing.update(
                {
                    "missing_value_policy": "training-fold feature mean; MBTR requires finite inputs",
                    "imputation_fit_partition": "fold training only",
                    "imputation_values": means.tolist(),
                    "comparison_note": (
                        "Same raw rows, targets, split, and metric as the published benchmark, but its "
                        "PyBoost/CatBoost/XGBoost methods handled RF1 missing values natively."
                    ),
                }
            )
        else:
            preprocessing["missing_value_policy"] = "none; the source contains no missing values"
        arrays = {
            "x_train": x_train,
            "y_train": y[train_indices],
            "x_validation": x_validation,
            "y_validation": y[validation_indices],
            "x_test": x_test,
            "y_test": y[test_indices],
            "source_index_train": train_indices,
            "source_index_validation": validation_indices,
            "source_index_test": test_indices,
        }
        runs.append(
            PreparedRun(
                dataset_id=dataset_id,
                protocol_id=RANDOM_PROTOCOL_ID,
                run_id=str(fold),
                task="regression",
                arrays=arrays,
                feature_names=feature_names,
                target_names=targets,
                source={
                    "repository_url": SKETCHBOOST_REPOSITORY,
                    "repository_revision": SKETCHBOOST_REVISION,
                    "download_url": specification.url,
                    "openml_file_id": 21_230_440 if dataset_id == "rf1" else 21_230_443,
                    "openml_dataset_id": 41_483 if dataset_id == "rf1" else 41_486,
                    "openml_dataset_version": 2,
                    "canonical_arff_md5": (
                        "744a74485a48f744cb68fd7f8a5dd1b5"
                        if dataset_id == "rf1"
                        else "5af7bb890c9de5f68a5179c051a6e2aa"
                    ),
                    "canonical_arff_sha256": (
                        "994f9e334811e040d2002e46d3ce06504e5d441249bf65448f53d6ea24b33cf0"
                        if dataset_id == "rf1"
                        else "2261a94bff5cfb617dbd2e7fd2c17d15d45135103a081917d1358c2ad13a4bd2"
                    ),
                    "raw_sha256": specification.sha256,
                    "raw_size_bytes": specification.size_bytes,
                    "raw_checksum_authority": "locally observed for OpenML's generated CSV export",
                    "license_status": "not recorded by the pinned SketchBoost data script",
                },
                preprocessing=preprocessing,
                split={
                    "outer": (
                        "sklearn train_test_split(test_size=0.2, random_state=42, "
                        "shuffle=True, stratify=None)"
                    ),
                    "inner": "sklearn KFold(5, shuffle=True, random_state=42)",
                    "fold": fold,
                    "source_index_space": "zero-based row positions in the parsed OpenML CSV",
                    "source_indices_sha256": indices_sha256(
                        train=train_indices,
                        validation=validation_indices,
                        test=test_indices,
                    ),
                    "uses_fixed_test_for_each_fold": True,
                },
            )
        )
    return runs


def _member_bytes(archive: zipfile.ZipFile, expected: dict[str, str]) -> dict[str, bytes]:
    files = [info for info in archive.infolist() if not info.is_dir()]
    observed = [info.filename for info in files]
    if len(observed) != len(set(observed)):
        raise ValueError("XC archive contains duplicate members.")
    if set(observed) != set(expected):
        raise ValueError(f"XC archive members differ: expected {sorted(expected)}, got {sorted(observed)}.")
    output: dict[str, bytes] = {}
    for info in files:
        if Path(info.filename).is_absolute() or ".." in Path(info.filename).parts:
            raise ValueError(f"Unsafe XC archive member {info.filename!r}.")
        payload = archive.read(info)
        import hashlib

        if hashlib.sha256(payload).hexdigest() != expected[info.filename]:
            raise ValueError(f"XC archive member checksum mismatch: {info.filename}.")
        output[info.filename] = payload
    return output


def _parse_xc_data(payload: bytes, expected: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    stream = io.StringIO(payload.decode("utf-8"))
    header = stream.readline().split()
    dimensions = tuple(int(value) for value in header)
    if dimensions != expected:
        raise ValueError(f"XC data header {dimensions} does not match {expected}.")
    rows, features, labels = dimensions
    x = np.zeros((rows, features), dtype=np.float32)
    y = np.zeros((rows, labels), dtype=np.float32)
    observed_rows = 0
    for line in stream:
        if not line.strip():
            continue
        if observed_rows >= rows:
            raise ValueError("XC data contains more rows than declared.")
        tokens = line.split()
        if tokens and ":" not in tokens[0]:
            for token in tokens.pop(0).split(","):
                label = int(token)
                if label < 0 or label >= labels:
                    raise ValueError(f"XC label index {label} is out of range.")
                y[observed_rows, label] = 1.0
        seen: set[int] = set()
        for token in tokens:
            index_text, separator, value_text = token.partition(":")
            if not separator:
                raise ValueError(f"Invalid XC feature token {token!r}.")
            index = int(index_text)
            if index < 0 or index >= features or index in seen:
                raise ValueError(f"Invalid or duplicate XC feature index {index}.")
            seen.add(index)
            x[observed_rows, index] = float(value_text)
        observed_rows += 1
    if observed_rows != rows or not np.all(np.isfinite(x)):
        raise ValueError(f"XC data contains {observed_rows} valid rows; expected {rows}.")
    return x, y


def _parse_first_split_column(payload: bytes, *, rows: int) -> np.ndarray:
    values: list[int] = []
    for line in payload.decode("utf-8").splitlines():
        tokens = line.split()
        if tokens:
            if len(tokens) != 10:
                raise ValueError("XC split rows must contain ten predefined split columns.")
            values.append(int(tokens[0]) - 1)
    indices = np.asarray(values, dtype=np.int64)
    if not len(indices) or np.any(indices < 0) or np.any(indices >= rows) or len(np.unique(indices)) != len(indices):
        raise ValueError("XC first-column split contains invalid or duplicate indices.")
    return indices


def _classification_runs(dataset_id: str, path: Path) -> list[PreparedRun]:
    specification = DOWNLOADS[dataset_id]
    validate_download(path, specification)
    with zipfile.ZipFile(path) as archive:
        members = _member_bytes(archive, XC_MEMBERS[dataset_id])
    prefix = "Delicious" if dataset_id == "delicious" else "Mediamill"
    lower = dataset_id
    rows, n_features, n_labels, expected_train, expected_test = XC_DIMENSIONS[dataset_id]
    x, y = _parse_xc_data(members[f"{prefix}/{prefix}_data.txt"], (rows, n_features, n_labels))
    train_indices = _parse_first_split_column(members[f"{prefix}/{lower}_trSplit.txt"], rows=rows)
    test_indices = _parse_first_split_column(members[f"{prefix}/{lower}_tstSplit.txt"], rows=rows)
    if len(train_indices) != expected_train or len(test_indices) != expected_test:
        raise ValueError(f"Unexpected {dataset_id} official split sizes.")
    if len(np.intersect1d(train_indices, test_indices)) or len(np.union1d(train_indices, test_indices)) != rows:
        raise ValueError(f"The {dataset_id} train/test split is not a disjoint complete partition.")

    feature_names = tuple(f"x{index}" for index in range(n_features))
    target_names = tuple(f"label_{index}" for index in range(n_labels))
    runs: list[PreparedRun] = []
    for fold, (fit_positions, validation_positions) in enumerate(
        KFold(5, shuffle=True, random_state=42).split(train_indices)
    ):
        fit_indices = train_indices[fit_positions]
        validation_indices = train_indices[validation_positions]
        arrays = {
            "x_train": x[fit_indices],
            "y_train": y[fit_indices],
            "x_validation": x[validation_indices],
            "y_validation": y[validation_indices],
            "x_test": x[test_indices],
            "y_test": y[test_indices],
            "source_index_train": fit_indices,
            "source_index_validation": validation_indices,
            "source_index_test": test_indices,
        }
        runs.append(
            PreparedRun(
                dataset_id=dataset_id,
                protocol_id=OFFICIAL_PROTOCOL_ID,
                run_id=str(fold),
                task="multilabel",
                arrays=arrays,
                feature_names=feature_names,
                target_names=target_names,
                source={
                    "repository_url": SKETCHBOOST_REPOSITORY,
                    "repository_revision": SKETCHBOOST_REVISION,
                    "download_url": specification.url,
                    "raw_sha256": specification.sha256,
                    "raw_size_bytes": specification.size_bytes,
                    "raw_checksum_authority": "locally observed; the publisher provides no checksum manifest",
                    "archive_member_sha256": XC_MEMBERS[dataset_id],
                    "source_collection": "Extreme Classification Repository legacy datasets",
                    "license_status": "not specified in the downloaded archive or pinned SketchBoost data script",
                },
                preprocessing={
                    "parser": "pinned SketchBoost zero-based sparse text expanded to dense float32",
                    "dense_model_bytes": int(x.nbytes + y.nbytes),
                    "label_encoding": "binary indicator matrix",
                },
                split={
                    "outer": "first (column 0) one-based XC train/test split, converted to zero-based",
                    "inner": "sklearn KFold(5, shuffle=True, random_state=42)",
                    "fold": fold,
                    "source_index_space": "zero-based row positions in the parsed XC sparse data file",
                    "source_indices_sha256": indices_sha256(
                        train=fit_indices,
                        validation=validation_indices,
                        test=test_indices,
                    ),
                    "uses_fixed_test_for_each_fold": True,
                },
            )
        )
    return runs


def prepare_sketchboost_runs(source_directory: Path, dataset_ids: list[str] | None = None) -> list[PreparedRun]:
    """Prepare the four directly downloadable SketchBoost datasets."""

    requested = list(DOWNLOADS) if dataset_ids is None else dataset_ids
    unknown = sorted(set(requested) - set(DOWNLOADS))
    if unknown:
        raise ValueError(f"SketchBoost public adapter does not provide: {unknown}.")
    runs: list[PreparedRun] = []
    for dataset_id in requested:
        path = source_directory / DOWNLOADS[dataset_id].filename
        if dataset_id in REGRESSION_TARGETS:
            runs.extend(_regression_runs(dataset_id, path))
        else:
            runs.extend(_classification_runs(dataset_id, path))
    return runs
