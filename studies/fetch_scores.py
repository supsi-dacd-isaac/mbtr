"""Download, verify, and normalize pinned public study score tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import urllib.request
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SOURCE_FILE = Path(__file__).with_name("references") / "remote_sources.json"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_source_record(source: dict[str, Any]) -> None:
    source_id = source.get("id")
    revision = source.get("revision")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("Every remote score source needs a non-empty string id.")
    if not isinstance(revision, str) or _COMMIT.fullmatch(revision) is None:
        raise ValueError(f"Remote score source {source_id!r} needs a full 40-character commit revision.")
    artifacts = source.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"results", "task_ids"}:
        raise ValueError(f"Remote score source {source_id!r} needs results and task_ids artifacts.")
    filenames: set[str] = set()
    for artifact_id, artifact in artifacts.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"Artifact {artifact_id!r} for {source_id!r} must be an object.")
        filename = artifact.get("filename")
        url = artifact.get("url")
        digest = artifact.get("sha256")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename:
            raise ValueError(f"Artifact {artifact_id!r} for {source_id!r} has an unsafe filename.")
        if filename in filenames:
            raise ValueError(f"Remote score source {source_id!r} has duplicate artifact filenames.")
        filenames.add(filename)
        if not isinstance(url, str) or not url.startswith("https://") or revision not in url:
            raise ValueError(f"Artifact {artifact_id!r} for {source_id!r} must use its pinned revision URL.")
        if "/main/" in url or "/master/" in url:
            raise ValueError(f"Artifact {artifact_id!r} for {source_id!r} uses a mutable branch URL.")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"Artifact {artifact_id!r} for {source_id!r} needs a SHA-256 digest.")
    methods = source.get("methods")
    if not isinstance(methods, dict) or not methods or not all(
        isinstance(key, str) and key and isinstance(value, str) and value for key, value in methods.items()
    ):
        raise ValueError(f"Remote score source {source_id!r} needs a method-name mapping.")
    task_map = source.get("task_map")
    if not isinstance(task_map, list) or not task_map:
        raise ValueError(f"Remote score source {source_id!r} needs an exact task_map.")
    names = [entry.get("source_task_name") for entry in task_map if isinstance(entry, dict)]
    identifiers = [entry.get("openml_task_id") for entry in task_map if isinstance(entry, dict)]
    if (
        len(names) != len(task_map)
        or not all(isinstance(name, str) and name for name in names)
        or not all(
            isinstance(identifier, int) and not isinstance(identifier, bool) and identifier > 0
            for identifier in identifiers
        )
        or len(set(names)) != len(names)
        or len(set(identifiers)) != len(identifiers)
    ):
        raise ValueError(f"Remote score source {source_id!r} has an invalid or duplicate task_map.")
    expected_rows = source.get("expected_rows")
    expected_cross_product = len(task_map) * len(methods)
    if expected_rows != expected_cross_product:
        raise ValueError(
            f"Remote score source {source_id!r} expected_rows must equal the task/method cross-product "
            f"({expected_cross_product})."
        )


def load_remote_sources() -> dict[str, dict[str, Any]]:
    """Return validated pinned remote-score records keyed by source id."""

    records = json.loads(SOURCE_FILE.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("remote_sources.json must contain a list of objects.")
    for record in records:
        _validate_source_record(record)
    sources = {str(record["id"]): record for record in records}
    if len(sources) != len(records):
        raise ValueError("remote_sources.json contains duplicate ids.")
    return sources


def parse_ctr23_scores(data: bytes) -> list[dict[str, str | float | int]]:
    """Parse the upstream CTR23 ``task_id,learner_id,rmse`` table strictly."""

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CTR23 results must be UTF-8 CSV.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["task_id", "learner_id", "rmse"]:
        raise ValueError(f"Unexpected CTR23 result columns: {reader.fieldnames}.")
    rows: list[dict[str, str | float | int]] = []
    keys: set[tuple[str, str]] = set()
    for line_number, row in enumerate(reader, start=2):
        source_task_name = (row.get("task_id") or "").strip()
        source_method = (row.get("learner_id") or "").strip()
        raw_rmse = (row.get("rmse") or "").strip()
        try:
            rmse = float(raw_rmse)
        except ValueError as exc:
            raise ValueError(f"Invalid RMSE on CTR23 result line {line_number}.") from exc
        if not source_task_name or not source_method or not math.isfinite(rmse) or rmse < 0.0:
            raise ValueError(f"Invalid CTR23 result row on line {line_number}.")
        key = (source_task_name, source_method)
        if key in keys:
            raise ValueError(f"Duplicate CTR23 task/method row: {key}.")
        keys.add(key)
        rows.append(
            {
                "source_task_name": source_task_name,
                "source_method": source_method,
                "rmse": rmse,
                "source_line": line_number,
            }
        )
    return rows


def parse_ctr23_task_ids(data: bytes) -> list[int]:
    """Parse the ordered numeric task-id companion file strictly."""

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CTR23 task ids must be UTF-8 CSV.") from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames != ["id"]:
        raise ValueError(f"Unexpected CTR23 task-id columns: {reader.fieldnames}.")
    identifiers: list[int] = []
    for line_number, row in enumerate(reader, start=2):
        raw = (row.get("id") or "").strip()
        try:
            identifier = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid OpenML task id on CTR23 id line {line_number}.") from exc
        if identifier <= 0 or str(identifier) != raw:
            raise ValueError(f"Invalid OpenML task id on CTR23 id line {line_number}.")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("CTR23 task-id file contains duplicate ids.")
    return identifiers


def validate_score_data(
    source: dict[str, Any],
    artifacts: Mapping[str, bytes],
) -> list[dict[str, str | float | int]]:
    """Verify artifact hashes and normalize a complete CTR23 score cross-product."""

    if source.get("format") != "ctr23_csv_v1":
        raise ValueError(f"Unsupported remote score format: {source.get('format')!r}.")
    expected_artifacts = source["artifacts"]
    if set(artifacts) != set(expected_artifacts):
        raise ValueError(f"Expected score artifacts {sorted(expected_artifacts)}; got {sorted(artifacts)}.")
    for artifact_id, specification in expected_artifacts.items():
        digest = _sha256(artifacts[artifact_id])
        if digest != specification["sha256"]:
            raise ValueError(
                f"Checksum mismatch for {source['id']}/{artifact_id}: "
                f"expected {specification['sha256']}, got {digest}."
            )

    raw_rows = parse_ctr23_scores(artifacts["results"])
    task_ids = parse_ctr23_task_ids(artifacts["task_ids"])
    task_map = source["task_map"]
    manifest_ids = [int(entry["openml_task_id"]) for entry in task_map]
    if task_ids != manifest_ids:
        raise ValueError("CTR23 ids.csv no longer matches the pinned exact task map.")
    name_to_id = {str(entry["source_task_name"]): int(entry["openml_task_id"]) for entry in task_map}
    methods = {str(key): str(value) for key, value in source["methods"].items()}
    observed = {(str(row["source_task_name"]), str(row["source_method"])) for row in raw_rows}
    expected = {(name, method) for name in name_to_id for method in methods}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing or unexpected or len(raw_rows) != source["expected_rows"]:
        details = []
        if missing:
            details.append(f"missing {missing[:5]}" + (" ..." if len(missing) > 5 else ""))
        if unexpected:
            details.append(f"unexpected {unexpected[:5]}" + (" ..." if len(unexpected) > 5 else ""))
        if len(raw_rows) != source["expected_rows"]:
            details.append(f"row count {len(raw_rows)} != {source['expected_rows']}")
        raise ValueError("Incomplete CTR23 task/method cross-product: " + "; ".join(details) + ".")

    normalized: list[dict[str, str | float | int]] = []
    for row in raw_rows:
        source_task_name = str(row["source_task_name"])
        source_method = str(row["source_method"])
        normalized.append(
            {
                "openml_task_id": name_to_id[source_task_name],
                "source_task_name": source_task_name,
                "source_method": source_method,
                "method": methods[source_method],
                "metric": str(source["metric"]),
                "value": float(row["rmse"]),
                "rmse": float(row["rmse"]),
                "comparison_tier": str(source["comparison_tier"]),
                "source_line": int(row["source_line"]),
            }
        )
    return normalized


def _cached_artifacts(source: dict[str, Any], source_directory: Path) -> dict[str, bytes]:
    return {
        artifact_id: (source_directory / specification["filename"]).read_bytes()
        for artifact_id, specification in source["artifacts"].items()
    }


def _read_response(response: Any) -> tuple[bytes, dict[str, Any]]:
    data = response.read()
    status = getattr(response, "status", None)
    headers = getattr(response, "headers", None)
    return data, {
        "http_status": status,
        "content_type": None if headers is None else headers.get("Content-Type"),
        "etag": None if headers is None else headers.get("ETag"),
    }


def _download_artifacts(
    source: dict[str, Any],
    opener: Callable[[urllib.request.Request], Any],
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    artifacts: dict[str, bytes] = {}
    response_metadata: dict[str, dict[str, Any]] = {}
    for artifact_id, specification in source["artifacts"].items():
        request = urllib.request.Request(specification["url"], headers={"User-Agent": "mbtr-studies/1"})
        with opener(request) as response:
            artifacts[artifact_id], response_metadata[artifact_id] = _read_response(response)
    return artifacts, response_metadata


def fetch_score_source(
    source_id: str,
    cache_directory: Path,
    *,
    offline: bool = False,
    refresh: bool = False,
    force: bool = False,
    opener: Callable[[urllib.request.Request], Any] = urllib.request.urlopen,
) -> tuple[Path, list[dict[str, str | float | int]]]:
    """Return a verified cached score source, fetching only when explicitly allowed.

    ``offline`` never calls ``opener``. ``refresh`` re-fetches every artifact,
    validating all bytes before replacing any valid cache. ``force`` is a
    backwards-compatible alias for ``refresh``.
    """

    if force:
        refresh = True
    if offline and refresh:
        raise ValueError("offline and refresh cannot be used together.")
    sources = load_remote_sources()
    if source_id not in sources:
        raise ValueError(f"Unknown remote score source {source_id!r}; choose from {sorted(sources)}.")
    source = sources[source_id]
    source_directory = cache_directory / source_id
    cached_files_exist = all(
        (source_directory / specification["filename"]).is_file()
        for specification in source["artifacts"].values()
    )
    cache_error: ValueError | OSError | None = None
    if cached_files_exist and not refresh:
        try:
            cached = _cached_artifacts(source, source_directory)
            return source_directory, validate_score_data(source, cached)
        except (ValueError, OSError) as exc:
            cache_error = exc
    if offline:
        if cache_error is not None:
            raise ValueError(f"Cached score source {source_id!r} is invalid: {cache_error}") from cache_error
        raise FileNotFoundError(f"Score source {source_id!r} is not fully cached in {source_directory}.")

    downloaded, response_metadata = _download_artifacts(source, opener)
    rows = validate_score_data(source, downloaded)
    # All network bytes have passed schema and checksum validation before the
    # first existing file is replaced.
    for artifact_id, specification in source["artifacts"].items():
        _atomic_write(source_directory / specification["filename"], downloaded[artifact_id])
    retrieved_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "schema_version": 1,
        "source_id": source_id,
        "revision": source["revision"],
        "retrieved_at_utc": retrieved_at,
        "artifacts": {
            artifact_id: {
                "url": specification["url"],
                "sha256": _sha256(downloaded[artifact_id]),
                **response_metadata[artifact_id],
            }
            for artifact_id, specification in source["artifacts"].items()
        },
    }
    _atomic_write(source_directory / "metadata.json", (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode())
    return source_directory, rows


def scores_by_dataset(
    rows: list[dict[str, str | float | int]],
) -> dict[str, dict[str, float]]:
    """Index normalized values by exact upstream task name and canonical method."""

    indexed: dict[str, dict[str, float]] = {}
    for row in rows:
        indexed.setdefault(str(row["source_task_name"]), {})[str(row["method"])] = float(row["value"])
    return indexed


def scores_by_openml_task(
    rows: list[dict[str, str | float | int]],
) -> dict[int, dict[str, float]]:
    """Index normalized values by numeric OpenML task id and canonical method."""

    indexed: dict[int, dict[str, float]] = {}
    for row in rows:
        indexed.setdefault(int(row["openml_task_id"]), {})[str(row["method"])] = float(row["value"])
    return indexed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(load_remote_sources()))
    parser.add_argument("--cache-directory", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="Require an existing verified cache; never use network.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch immutable artifacts and verify before replacing cache.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    path, rows = fetch_score_source(
        arguments.source,
        arguments.cache_directory,
        offline=arguments.offline,
        refresh=arguments.refresh,
    )
    print(json.dumps({"cache_directory": str(path), "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
