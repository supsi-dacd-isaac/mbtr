"""Fetch and prepare the directly available multivariate benchmark datasets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .prepare_common import PreparedRun, WrittenRun, file_sha256, write_prepared_run
from .prepare_kdis import (
    KDIS_ARCHIVE_MANIFEST,
    KDIS_LICENSE_STATUS,
    KDIS_PROTOCOL_ID,
    KDIS_REPOSITORY_URL,
    load_kdis_archive,
)
from .prepare_mocr import DATASETS as MOCR_DATASETS
from .prepare_mocr import MOCR_REVISION, prepare_mocr_runs
from .prepare_sketchboost import DOWNLOADS as SKETCHBOOST_DOWNLOADS
from .prepare_sketchboost import prepare_sketchboost_runs

PUBLIC_DATASET_IDS = (
    "rf1",
    "scm20d",
    "mocr_households",
    "mocr_house",
    "mocr_bio",
    "mocr_blog_data",
    "mocr_taxi",
    "mediamill",
    "delicious",
    "emotions",
    "scene",
    "yeast",
)
UNAVAILABLE_DATASET_IDS = {
    "moa": "Kaggle competition authentication and acceptance of the competition data terms are required.",
    "mocr_meps19": "The pinned MOCR repository does not bundle this external dataset.",
    "mocr_meps20": "The pinned MOCR repository does not bundle this external dataset.",
    "mocr_meps21": "The pinned MOCR repository does not bundle this external dataset.",
}
KDIS_ORIGINAL_CITATIONS = {
    "emotions": "Tsoumakas et al. (2008), Correlation-based pruning of stacked binary relevance models",
    "scene": "Boutell et al. (2004), Learning multi-label scene classification",
    "yeast": "Elisseeff and Weston (2001), A kernel method for multi-labelled classification",
}


@dataclass(frozen=True)
class RemoteFile:
    dataset_id: str
    path: Path
    url: str
    size_bytes: int
    sha256: str


def _selected_dataset_ids(values: list[str] | None) -> list[str]:
    requested = list(PUBLIC_DATASET_IDS) if not values else list(dict.fromkeys(values))
    unavailable = {name: UNAVAILABLE_DATASET_IDS[name] for name in requested if name in UNAVAILABLE_DATASET_IDS}
    unknown = sorted(set(requested) - set(PUBLIC_DATASET_IDS) - set(UNAVAILABLE_DATASET_IDS))
    if unavailable:
        details = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        raise ValueError(f"Requested datasets are not directly available: {details}")
    if unknown:
        raise ValueError(f"Unknown multivariate datasets: {unknown}.")
    return [name for name in PUBLIC_DATASET_IDS if name in requested]


def remote_files(raw_directory: Path, dataset_ids: Iterable[str]) -> list[RemoteFile]:
    """Resolve pinned remote identities and local cache paths."""

    files: list[RemoteFile] = []
    for dataset_id in dataset_ids:
        if dataset_id in SKETCHBOOST_DOWNLOADS:
            sketch_source = SKETCHBOOST_DOWNLOADS[dataset_id]
            files.append(
                RemoteFile(
                    dataset_id,
                    raw_directory / "sketchboost" / sketch_source.filename,
                    sketch_source.url,
                    sketch_source.size_bytes,
                    sketch_source.sha256,
                )
            )
        elif dataset_id in MOCR_DATASETS:
            mocr_source = MOCR_DATASETS[dataset_id]
            files.append(
                RemoteFile(
                    dataset_id,
                    raw_directory / "mocr" / mocr_source.relative_path,
                    f"https://raw.githubusercontent.com/Vekteur/multi-output-conformal-regression/"
                    f"{MOCR_REVISION}/{mocr_source.relative_path}",
                    mocr_source.size_bytes,
                    mocr_source.sha256,
                )
            )
        elif dataset_id in KDIS_ARCHIVE_MANIFEST:
            kdis_source = KDIS_ARCHIVE_MANIFEST[dataset_id]
            files.append(
                RemoteFile(
                    dataset_id,
                    raw_directory / "kdis" / Path(kdis_source.url).name,
                    kdis_source.url,
                    kdis_source.size_bytes,
                    kdis_source.sha256,
                )
            )
    return files


def _valid_cached_file(source: RemoteFile) -> bool:
    return (
        source.path.is_file()
        and source.path.stat().st_size == source.size_bytes
        and file_sha256(source.path) == source.sha256
    )


def fetch_remote_file(
    source: RemoteFile,
    *,
    replace: bool = False,
    timeout_seconds: float = 180.0,
) -> dict[str, Any]:
    """Fetch one pinned file atomically, or verify and reuse its cached copy."""

    if _valid_cached_file(source):
        return {**asdict(source), "path": str(source.path), "status": "reused"}
    if source.path.exists() and not replace:
        raise ValueError(
            f"Cached source {source.path} does not match its pinned size/checksum. "
            "Pass --replace only if it is safe to replace that file."
        )

    source.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = source.path.with_name(f".{source.path.name}.{os.getpid()}.download")
    request = urllib.request.Request(source.url, headers={"User-Agent": "mbtr-benchmark-preparer/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response, temporary.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
        observed_size = temporary.stat().st_size
        observed_sha256 = file_sha256(temporary)
        if observed_size != source.size_bytes or observed_sha256 != source.sha256:
            raise ValueError(
                f"Downloaded {source.dataset_id} failed integrity validation: "
                f"size={observed_size}, sha256={observed_sha256}."
            )
        os.replace(temporary, source.path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**asdict(source), "path": str(source.path), "status": "downloaded"}


def fetch_public_sources(
    raw_directory: Path,
    dataset_ids: list[str],
    *,
    replace: bool = False,
    timeout_seconds: float = 180.0,
) -> list[dict[str, Any]]:
    """Download or verify all selected public raw sources."""

    results: list[dict[str, Any]] = []
    for source in remote_files(raw_directory, dataset_ids):
        result = fetch_remote_file(source, replace=replace, timeout_seconds=timeout_seconds)
        results.append(result)
        print(f"{result['status']:>10}  {source.dataset_id:<18} {source.path}", file=sys.stderr, flush=True)
    return results


def _kdis_runs(raw_directory: Path, dataset_id: str) -> Iterable[PreparedRun]:
    specification = KDIS_ARCHIVE_MANIFEST[dataset_id]
    path = raw_directory / "kdis" / Path(specification.url).name
    prepared = load_kdis_archive(path, dataset_id)
    for fold in prepared.folds:
        yield PreparedRun(
            dataset_id=dataset_id,
            protocol_id=KDIS_PROTOCOL_ID,
            run_id=fold.run_id,
            task="multilabel",
            arrays=fold.arrays(),
            feature_names=prepared.feature_names,
            target_names=prepared.target_names,
            source={
                "repository_url": KDIS_REPOSITORY_URL,
                "download_url": specification.url,
                "raw_sha256": specification.sha256,
                "raw_size_bytes": specification.size_bytes,
                "raw_checksum_authority": "locally observed; KDIS publishes no checksum manifest",
                "last_modified": specification.last_modified,
                "etag": specification.etag,
                "license_status": KDIS_LICENSE_STATUS,
                "original_dataset_citation": KDIS_ORIGINAL_CITATIONS[dataset_id],
            },
            preprocessing={
                "format": "Mulan ARFF plus XML label metadata",
                "feature_policy": "numeric attributes in ARFF declaration order",
                "target_policy": "XML identifies the label set; ARFF declaration order fixes target order",
                "missing_value_policy": "rejected; official selected folds contain no missing values",
            },
            split={
                "partition": "fixed KDIS iterative-stratified five-fold archive",
                "fold_number": fold.fold_number,
                "train_member": fold.train_member,
                "test_member": fold.test_member,
                "generator": "MLDA; upstream version/seed not published",
                "protocol_scope": (
                    "Fixed KDIS 2017 five-fold archive; not the UCO 2024 ten-partition comparison."
                ),
            },
        )


def _runs_for_dataset(raw_directory: Path, dataset_id: str) -> Iterable[PreparedRun]:
    if dataset_id in SKETCHBOOST_DOWNLOADS:
        return prepare_sketchboost_runs(raw_directory / "sketchboost", [dataset_id])
    if dataset_id in MOCR_DATASETS:
        return prepare_mocr_runs(raw_directory / "mocr", [dataset_id])
    if dataset_id in KDIS_ARCHIVE_MANIFEST:
        return _kdis_runs(raw_directory, dataset_id)
    raise ValueError(f"No public preparation adapter for {dataset_id!r}.")


def prepare_public_splits(
    raw_directory: Path,
    output_directory: Path,
    dataset_ids: list[str],
    *,
    replace: bool = False,
) -> list[WrittenRun]:
    """Prepare and atomically write every selected protocol run."""

    written: list[WrittenRun] = []
    for dataset_id in dataset_ids:
        for run in _runs_for_dataset(raw_directory, dataset_id):
            result = write_prepared_run(run, output_directory, replace=replace)
            written.append(result)
            status = "reused" if result.reused else "prepared"
            print(
                f"{status:>10}  {run.dataset_id:<18} run={run.run_id}  {result.split_path}",
                file=sys.stderr,
                flush=True,
            )
    return written


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "prepare", "all"))
    parser.add_argument(
        "--raw-directory",
        type=Path,
        default=Path("studies/data/multivariate-raw"),
        help="Pinned raw download cache (default: studies/data/multivariate-raw).",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("studies/data/multivariate"),
        help="Prepared split root (default: studies/data/multivariate).",
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        help="Prepare one public dataset; repeat as needed. The default is all 12 public datasets.",
    )
    parser.add_argument("--replace", action="store_true", help="Replace mismatched raw or prepared artifacts.")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.timeout_seconds <= 0:
        raise ValueError("--timeout-seconds must be positive.")
    dataset_ids = _selected_dataset_ids(arguments.dataset_id)
    fetched: list[dict[str, Any]] = []
    written: list[WrittenRun] = []
    if arguments.action in {"fetch", "all"}:
        fetched = fetch_public_sources(
            arguments.raw_directory,
            dataset_ids,
            replace=arguments.replace,
            timeout_seconds=arguments.timeout_seconds,
        )
    if arguments.action in {"prepare", "all"}:
        written = prepare_public_splits(
            arguments.raw_directory,
            arguments.output_directory,
            dataset_ids,
            replace=arguments.replace,
        )
    summary = {
        "action": arguments.action,
        "datasets": dataset_ids,
        "fetched": len(fetched),
        "downloaded": sum(item["status"] == "downloaded" for item in fetched),
        "reused_downloads": sum(item["status"] == "reused" for item in fetched),
        "prepared_runs": len(written),
        "written_runs": sum(not item.reused for item in written),
        "reused_runs": sum(item.reused for item in written),
        "raw_directory": str(arguments.raw_directory),
        "output_directory": str(arguments.output_directory),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
