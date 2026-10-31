"""Aggregate completed fold/repeat/seed study runs with protocol checks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import find_protocol


def _same(documents: list[dict[str, Any]], field: str) -> Any:
    first = documents[0].get(field)
    if any(document.get(field) != first for document in documents[1:]):
        raise ValueError(f"All study runs must have the same {field}.")
    return first


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
    }


def aggregate_documents(
    documents: list[dict[str, Any]],
    *,
    allow_incomplete: bool = False,
    expected_run_ids: Sequence[str] | None = None,
    run_axis: str | None = None,
) -> dict[str, Any]:
    """Validate and aggregate JSON documents emitted by :mod:`studies.run`."""

    if not documents:
        raise ValueError("At least one study result is required.")
    if any(document.get("schema_version") != 1 for document in documents):
        raise ValueError("All study runs must use schema_version 1.")
    dataset_id = str(_same(documents, "dataset_id"))
    protocol_id = str(_same(documents, "protocol_id"))
    task = str(_same(documents, "task"))
    metric_config = _same(documents, "metric_config")
    target_transforms = [document.get("target_transform") for document in documents]
    transform_presence = {transform is not None for transform in target_transforms}
    if len(transform_presence) != 1:
        raise ValueError("All study runs must use the same target-transform policy.")
    if target_transforms[0] is None:
        target_transform_policy = "none"
    elif all(transform == target_transforms[0] for transform in target_transforms[1:]):
        target_transform_policy = "shared"
    else:
        signatures = [
            (
                tuple(sorted(transform)) if isinstance(transform, dict) else (),
                len(transform.get("offset", [])) if isinstance(transform, dict) else -1,
                len(transform.get("scale", [])) if isinstance(transform, dict) else -1,
            )
            for transform in target_transforms
        ]
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError("All study runs must use compatible per-run target transforms.")
        target_transform_policy = "per_run_train_fitted"

    model_configuration = dict(documents[0]["parameters"])
    model_configuration.pop("random_state", None)
    for document in documents[1:]:
        candidate = dict(document["parameters"])
        candidate.pop("random_state", None)
        if candidate != model_configuration:
            raise ValueError("All study runs must have the same model parameters apart from random_state.")

    run_ids = [str(document["run_id"]) for document in documents]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("Study run_ids must be unique.")
    protocol = find_protocol(protocol_id)
    if expected_run_ids is None:
        expected_ids = sorted(run_ids) if protocol is None else [str(value) for value in protocol["expected_run_ids"]]
    else:
        expected_ids = [str(value) for value in expected_run_ids]
        if not expected_ids or any(not value for value in expected_ids):
            raise ValueError("expected_run_ids must contain non-empty values.")
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected_run_ids must be unique.")
    unexpected_ids = sorted(set(run_ids) - set(expected_ids))
    if unexpected_ids:
        raise ValueError(f"Unexpected run_ids for protocol {protocol_id}: {unexpected_ids}.")
    missing_ids = sorted(set(expected_ids) - set(run_ids))
    complete = not missing_ids
    if missing_ids and not allow_incomplete:
        raise ValueError(f"Protocol {protocol_id} is incomplete; missing run_ids: {missing_ids}.")

    variants = [result["variant"] for result in documents[0]["results"]]
    if len(set(variants)) != len(variants):
        raise ValueError("A study run contains duplicate variants.")
    if any({result["variant"] for result in document["results"]} != set(variants) for document in documents[1:]):
        raise ValueError("All study runs must contain the same variants.")
    references = _same(documents, "published_references")
    eligible_references = [
        reference for reference in references if reference.get("comparison_tier") in {"direct", "protocol_reference"}
    ]
    revisions = {(document.get("code") or {}).get("revision") for document in documents}
    if len(revisions) != 1:
        raise ValueError("All study runs must use the same repository revision.")
    ordered_documents = sorted(documents, key=lambda document: expected_ids.index(str(document["run_id"])))
    aggregate_results: list[dict[str, Any]] = []
    for variant in variants:
        variant_results = []
        for document in ordered_documents:
            matches = [result for result in document["results"] if result["variant"] == variant]
            if len(matches) != 1:
                raise ValueError(f"Every run must contain variant {variant!r} exactly once.")
            variant_results.append(matches[0])
        metric_names = {name for result in variant_results for name in result["metrics"]}
        metric_summaries: dict[str, Any] = {}
        for name in sorted(metric_names):
            observed = [
                (str(document["run_id"]), float(result["metrics"][name]))
                for document, result in zip(ordered_documents, variant_results, strict=True)
                if name in result["metrics"]
            ]
            summary = _summary([value for _, value in observed])
            if len(observed) != len(variant_results):
                observed_ids = [run_id for run_id, _ in observed]
                summary["observed_run_ids"] = observed_ids
                summary["missing_run_ids"] = [
                    str(document["run_id"])
                    for document in ordered_documents
                    if str(document["run_id"]) not in set(observed_ids)
                ]
            metric_summaries[name] = summary
        aggregate_results.append(
            {
                "variant": variant,
                "metrics": metric_summaries,
                "fit_seconds": _summary([float(result["fit_seconds"]) for result in variant_results]),
                "predict_seconds": _summary([float(result["predict_seconds"]) for result in variant_results]),
                "metric_seconds": _summary([float(result["metric_seconds"]) for result in variant_results]),
                "serialized_model_bytes": _summary(
                    [float(result["serialized_model_bytes"]) for result in variant_results]
                ),
                "resolved_engines": sorted(
                    {engine for result in variant_results for engine in result["resolved_engines"]}
                ),
            }
        )

    return {
        "schema_version": 1,
        "result_type": "aggregate",
        "dataset_id": dataset_id,
        "protocol_id": protocol_id,
        "task": task,
        "run_axis": run_axis if run_axis is not None else (None if protocol is None else protocol["run_axis"]),
        "expected_run_ids": expected_ids,
        "observed_run_ids": [str(document["run_id"]) for document in ordered_documents],
        "missing_run_ids": missing_ids,
        "complete": complete,
        "model_configuration": model_configuration,
        "random_states": [document["parameters"].get("random_state") for document in ordered_documents],
        "metric_config": metric_config,
        "target_transform": (
            target_transforms[0]
            if target_transform_policy in {"none", "shared"}
            else {"policy": "per_run_train_fitted"}
        ),
        "target_transform_policy": target_transform_policy,
        "target_transforms_by_run": [
            {"run_id": document["run_id"], "transform": document.get("target_transform")}
            for document in ordered_documents
        ],
        "results": aggregate_results,
        "published_references": references if complete else [],
        "published_references_withheld": 0 if complete else len(references),
        "comparison_eligible_reference_count": len(eligible_references) if complete else 0,
        "context_only_reference_count": (
            sum(reference.get("comparison_tier") == "context_only" for reference in references) if complete else 0
        ),
        "comparison_ready": complete and bool(eligible_references),
        "runs": [
            {
                "run_id": document["run_id"],
                "split_sha256": document["split_sha256"],
                "created_at_utc": document.get("created_at_utc"),
                "code": document.get("code"),
                "environment": document.get("environment"),
            }
            for document in ordered_documents
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Per-run JSON files from studies.run.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--expected-run-id", action="append", dest="expected_run_ids")
    parser.add_argument("--run-axis")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in arguments.results]
    aggregate = aggregate_documents(
        documents,
        allow_incomplete=arguments.allow_incomplete,
        expected_run_ids=arguments.expected_run_ids,
        run_axis=arguments.run_axis,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
