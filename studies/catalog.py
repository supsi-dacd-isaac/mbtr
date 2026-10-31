"""Load and validate study protocols and curated published scores."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CATALOG_DIRECTORY = Path(__file__).with_name("catalog")
REFERENCE_DIRECTORY = Path(__file__).with_name("references")


@dataclass(frozen=True)
class PublishedScore:
    """One value transcribed from a cited public result table."""

    dataset_id: str
    task: str
    protocol_id: str
    method: str
    structure: str
    metric: str
    value: float
    std: float | None
    direction: str
    reported_scale: str
    comparison_tier: str
    source_url: str
    source_locator: str
    notes: str


def _load_json(name: str) -> list[dict[str, Any]]:
    path = CATALOG_DIRECTORY / name
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{path} must contain a JSON list of objects.")
    return value


def load_datasets() -> list[dict[str, Any]]:
    return _load_json("datasets.json")


def load_protocols() -> list[dict[str, Any]]:
    return _load_json("protocols.json")


def load_suites() -> list[dict[str, Any]]:
    return _load_json("suites.json")


def find_protocol(protocol_id: str) -> dict[str, Any] | None:
    """Return one protocol record by id, or ``None`` for an ad-hoc run."""

    return next((protocol for protocol in load_protocols() if protocol.get("id") == protocol_id), None)


def load_published_scores() -> list[PublishedScore]:
    path = REFERENCE_DIRECTORY / "published_scores.csv"
    records: list[PublishedScore] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            standard_deviation = None if not row["std"].strip() else float(row["std"])
            records.append(
                PublishedScore(
                    dataset_id=row["dataset_id"],
                    task=row["task"],
                    protocol_id=row["protocol_id"],
                    method=row["method"],
                    structure=row["structure"],
                    metric=row["metric"],
                    value=float(row["value"]),
                    std=standard_deviation,
                    direction=row["direction"],
                    reported_scale=row["reported_scale"],
                    comparison_tier=row["comparison_tier"],
                    source_url=row["source_url"],
                    source_locator=row["source_locator"],
                    notes=row["notes"],
                )
            )
    return records


def find_published_scores(
    dataset_id: str,
    protocol_id: str,
    metrics: set[str] | None = None,
) -> list[PublishedScore]:
    """Return cited rows matching one dataset/protocol and optional metrics."""

    return [
        score
        for score in load_published_scores()
        if score.dataset_id == dataset_id
        and score.protocol_id == protocol_id
        and (metrics is None or score.metric in metrics)
    ]


def _ids(records: list[dict[str, Any]], *, kind: str, errors: list[str]) -> set[str]:
    identifiers = [record.get("id") for record in records]
    invalid = [identifier for identifier in identifiers if not isinstance(identifier, str) or not identifier]
    if invalid:
        errors.append(f"Every {kind} must have a non-empty string id.")
    strings = [identifier for identifier in identifiers if isinstance(identifier, str)]
    duplicates = sorted({identifier for identifier in strings if strings.count(identifier) > 1})
    if duplicates:
        errors.append(f"Duplicate {kind} ids: {duplicates}.")
    return set(strings)


def validate_catalog() -> None:
    """Raise ``ValueError`` if catalog references or score metadata disagree."""

    datasets = load_datasets()
    protocols = load_protocols()
    suites = load_suites()
    scores = load_published_scores()
    errors: list[str] = []
    dataset_ids = _ids(datasets, kind="dataset", errors=errors)
    protocol_ids = _ids(protocols, kind="protocol", errors=errors)
    _ids(suites, kind="suite", errors=errors)
    datasets_by_id = {record["id"]: record for record in datasets if isinstance(record.get("id"), str)}
    protocols_by_id = {record["id"]: record for record in protocols if isinstance(record.get("id"), str)}

    for dataset in datasets:
        if not str(dataset.get("source_url", "")).startswith("https://"):
            errors.append(f"Dataset {dataset.get('id')} needs an HTTPS source URL.")
        for dimension in ("n_samples", "n_features", "n_outputs"):
            value = dataset.get(dimension)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                errors.append(f"Dataset {dataset.get('id')} has invalid {dimension}: {value}.")

    for protocol in protocols:
        unknown = sorted(set(protocol.get("datasets", [])) - dataset_ids)
        if unknown:
            errors.append(f"Protocol {protocol.get('id')} references unknown datasets: {unknown}.")
        if not str(protocol.get("source_url", "")).startswith("https://"):
            errors.append(f"Protocol {protocol.get('id')} needs an HTTPS source URL.")
        expected_runs = protocol.get("expected_runs")
        expected_run_ids = protocol.get("expected_run_ids")
        if not isinstance(expected_runs, int) or isinstance(expected_runs, bool) or expected_runs < 1:
            errors.append(f"Protocol {protocol.get('id')} needs a positive expected_runs value.")
        if not isinstance(expected_run_ids, list) or len(expected_run_ids) != expected_runs:
            errors.append(f"Protocol {protocol.get('id')} expected_run_ids must match expected_runs.")
        elif not all(isinstance(run_id, str) and run_id for run_id in expected_run_ids):
            errors.append(f"Protocol {protocol.get('id')} expected_run_ids must be non-empty strings.")
        elif len(set(expected_run_ids)) != len(expected_run_ids):
            errors.append(f"Protocol {protocol.get('id')} has duplicate expected_run_ids.")

    for suite in suites:
        if not str(suite.get("source_url", "")).startswith("https://"):
            errors.append(f"Suite {suite.get('id')} needs an HTTPS source URL.")

    seen_scores: set[tuple[str, str, str, str]] = set()
    for score in scores:
        if score.dataset_id not in dataset_ids:
            errors.append(f"Score references unknown dataset {score.dataset_id}.")
        if score.protocol_id not in protocol_ids:
            errors.append(f"Score references unknown protocol {score.protocol_id}.")
        elif score.dataset_id not in protocols_by_id[score.protocol_id].get("datasets", []):
            errors.append(
                f"Score {score.method}/{score.dataset_id} is not covered by protocol {score.protocol_id}."
            )
        if score.dataset_id in datasets_by_id and score.task != datasets_by_id[score.dataset_id].get("task"):
            errors.append(
                f"Score {score.method}/{score.dataset_id} has task {score.task} inconsistent with its dataset."
            )
        if score.direction not in {"minimize", "maximize"}:
            errors.append(f"Score {score.method}/{score.dataset_id} has invalid direction {score.direction}.")
        if score.comparison_tier not in {"direct", "protocol_reference", "context_only"}:
            errors.append(
                f"Score {score.method}/{score.dataset_id} has invalid comparison tier {score.comparison_tier}."
            )
        if not math.isfinite(score.value) or score.std is not None and not math.isfinite(score.std):
            errors.append(f"Score {score.method}/{score.dataset_id} is not finite.")
        if not score.source_url.startswith("https://"):
            errors.append(f"Score {score.method}/{score.dataset_id} needs an HTTPS source URL.")
        key = (score.dataset_id, score.protocol_id, score.method, score.metric)
        if key in seen_scores:
            errors.append(f"Duplicate published score: {key}.")
        seen_scores.add(key)

    if errors:
        raise ValueError("Invalid studies catalog:\n- " + "\n- ".join(errors))


__all__ = [
    "PublishedScore",
    "find_protocol",
    "find_published_scores",
    "load_datasets",
    "load_protocols",
    "load_published_scores",
    "load_suites",
    "validate_catalog",
]
