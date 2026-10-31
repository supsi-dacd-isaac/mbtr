"""Task-balanced OpenML suite reports and CTR23 comparisons."""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from statistics import median
from typing import Any

import numpy as np


def _variant_metrics(aggregate: dict[str, Any], variant: str = "shared") -> dict[str, float]:
    hierarchy = aggregate.get("hierarchy", {})
    hierarchical = hierarchy.get("variants", {}).get(variant, {}).get("metrics", {})
    if hierarchical:
        return {
            name: float(summary["task_mean"])
            for name, summary in hierarchical.items()
            if summary.get("task_mean") is not None
        }
    matches = [result for result in aggregate.get("results", []) if result.get("variant") == variant]
    if len(matches) != 1:
        return {}
    return {
        name: float(summary["mean"])
        for name, summary in matches[0].get("metrics", {}).items()
        if summary.get("mean") is not None
    }


def _average_rank(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][1] == ordered[start][1]:
            stop += 1
        rank = ((start + 1) + stop) / 2.0
        for method, _ in ordered[start:stop]:
            ranks[method] = rank
        start = stop
    return ranks


def _reference_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    indexed: dict[int, dict[str, float]] = {}
    for row in rows:
        raw_task_id = row.get("openml_task_id", row.get("task_id"))
        if raw_task_id is None:
            raise ValueError("Every normalized reference row needs an exact numeric OpenML task id.")
        task_id = int(raw_task_id)
        method = str(row.get("method", row.get("learner_id", "")))
        raw_value = row.get("value", row.get("rmse"))
        if raw_value is None:
            raise ValueError(f"Reference row for task {task_id} has no score value.")
        value = float(raw_value)
        if not method or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid normalized reference row for task {task_id}.")
        methods = indexed.setdefault(task_id, {})
        if method in methods:
            raise ValueError(f"Duplicate reference method {method!r} for task {task_id}.")
        methods[method] = value
    return indexed


def build_report(
    lock: dict[str, Any],
    aggregates: list[dict[str, Any]],
    *,
    reference_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a suite report in which every task has equal weight."""

    aggregate_by_task = {
        int(aggregate["openml"]["task_id"]): aggregate
        for aggregate in aggregates
        if aggregate.get("openml", {}).get("task_id") is not None
    }
    references = _reference_index(reference_rows or [])
    task_rows: list[dict[str, Any]] = []
    for task in lock.get("tasks", []):
        if task.get("status") != "prepared":
            continue
        task_id = int(task["task_id"])
        aggregate = aggregate_by_task.get(task_id)
        metrics = {} if aggregate is None else _variant_metrics(aggregate)
        method_scores = references.get(task_id, {})
        row: dict[str, Any] = {
            "task_id": task_id,
            "dataset_id": task.get("dataset_id"),
            "dataset_name": task.get("dataset_name"),
            "task_kind": task.get("task_kind"),
            "selection_complete": bool(aggregate and aggregate.get("complete")),
            "official_complete": bool(aggregate and aggregate.get("openml", {}).get("official_complete")),
            "observed_runs": 0 if aggregate is None else len(aggregate.get("observed_run_ids", [])),
            "official_runs": task.get("official_expected_runs"),
            "metrics": metrics,
            "reference_scores": method_scores,
            "reference_comparison_tier": "context_only" if method_scores else None,
        }
        if "global_rmse" in metrics and method_scores and row["official_complete"]:
            scores = {"MBTR": metrics["global_rmse"], **method_scores}
            ranks = _average_rank(scores)
            best_method = min(method_scores, key=lambda name: (method_scores[name], name))
            best_value = method_scores[best_method]
            row.update(
                {
                    "best_reference_method": best_method,
                    "best_reference_rmse": best_value,
                    "mbtr_to_best_reference_ratio": None
                    if best_value == 0.0
                    else metrics["global_rmse"] / best_value,
                    "mbtr_rank": ranks["MBTR"],
                }
            )
        task_rows.append(row)

    summary: dict[str, Any] = {}
    task_kinds = {row["task_kind"] for row in task_rows}
    if task_kinds == {"binary"}:
        metric_values: dict[str, list[float]] = defaultdict(list)
        for row in task_rows:
            if row["selection_complete"]:
                for name, value in row["metrics"].items():
                    metric_values[name].append(float(value))
        summary["equal_task_mean_metrics"] = {
            name: float(np.mean(values)) for name, values in sorted(metric_values.items()) if values
        }

    if references:
        methods = sorted({method for values in references.values() for method in values})
        comparisons: dict[str, Any] = {}
        mbtr_ranks: list[float] = []
        for row in task_rows:
            if row.get("mbtr_rank") is not None and row["official_complete"]:
                mbtr_ranks.append(float(row["mbtr_rank"]))
        for method in methods:
            ratios: list[float] = []
            wins = ties = losses = 0
            for row in task_rows:
                if not row["official_complete"] or "global_rmse" not in row["metrics"]:
                    continue
                reference = row["reference_scores"].get(method)
                if reference is None:
                    continue
                mbtr_value = float(row["metrics"]["global_rmse"])
                if reference > 0.0 and mbtr_value > 0.0:
                    ratios.append(mbtr_value / float(reference))
                if mbtr_value < reference:
                    wins += 1
                elif mbtr_value > reference:
                    losses += 1
                else:
                    ties += 1
            comparisons[method] = {
                "tasks": wins + ties + losses,
                "mbtr_wins": wins,
                "ties": ties,
                "mbtr_losses": losses,
                "median_rmse_ratio": None if not ratios else float(median(ratios)),
                "geometric_mean_rmse_ratio": None
                if not ratios
                else float(math.exp(np.mean(np.log(ratios)))),
            }
        summary["reference_comparisons"] = comparisons
        summary["mean_mbtr_rank"] = None if not mbtr_ranks else float(np.mean(mbtr_ranks))
        summary["reference_note"] = (
            "CTR23 values are contextual: official outer resampling and mean-fold RMSE align, but MBTR's "
            "preprocessing and tuning budget do not reproduce each published learner's protocol. Published "
            "XGBoost uses nested random search with 500 evaluations per outer fit, ridge uses internal "
            "cross-validation, and the remaining methods follow different fixed or model-internal policies."
        )

    return {
        "schema_version": 1,
        "result_type": "openml_suite_report",
        "suite": lock.get("suite"),
        "selection": lock.get("selection"),
        "tasks_resolved": len(lock.get("tasks", [])),
        "tasks_reported": len(task_rows),
        "tasks_with_complete_selection": sum(bool(row["selection_complete"]) for row in task_rows),
        "tasks_with_complete_official_resampling": sum(bool(row["official_complete"]) for row in task_rows),
        "reference_tasks_matched": sum(bool(row["reference_scores"]) for row in task_rows),
        "task_rows": task_rows,
        "suite_summary": summary,
        "aggregation_policy": (
            "Fold metrics are averaged within repeat, repeat means within task, and suite summaries give "
            "each task equal weight. Raw regression RMSE is never averaged across datasets."
        ),
    }


def report_csv(report: dict[str, Any]) -> str:
    """Return one flat task-level CSV table."""

    metric_names = sorted(
        {name for row in report["task_rows"] for name in row.get("metrics", {})}
    )
    reference_methods = sorted(
        {name for row in report["task_rows"] for name in row.get("reference_scores", {})}
    )
    fields = [
        "task_id",
        "dataset_id",
        "dataset_name",
        "task_kind",
        "selection_complete",
        "official_complete",
        "observed_runs",
        "official_runs",
        *[f"mbtr_{name}" for name in metric_names],
        *[f"reference_{method}_rmse" for method in reference_methods],
        "mbtr_rank",
        "mbtr_to_best_reference_ratio",
    ]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in report["task_rows"]:
        flat = {name: row.get(name) for name in fields}
        flat.update({f"mbtr_{name}": value for name, value in row.get("metrics", {}).items()})
        flat.update(
            {f"reference_{method}_rmse": value for method, value in row.get("reference_scores", {}).items()}
        )
        writer.writerow(flat)
    return stream.getvalue()


def report_markdown(report: dict[str, Any]) -> str:
    """Return a compact human-readable report."""

    suite = report.get("suite", {}) or {}
    lines = [
        f"# {suite.get('alias', suite.get('name', 'OpenML'))} MBTR study",
        "",
        report["aggregation_policy"],
        "",
        f"- Tasks reported: {report['tasks_reported']}",
        f"- Complete selected workloads: {report['tasks_with_complete_selection']}",
        f"- Complete official resampling: {report['tasks_with_complete_official_resampling']}",
        f"- Reference tasks matched by exact task ID: {report['reference_tasks_matched']}",
        "",
        "| Task | Dataset | MBTR metric | Official complete | Reference context |",
        "|---:|---|---:|:---:|---|",
    ]
    for row in report["task_rows"]:
        metrics = row.get("metrics", {})
        if "global_rmse" in metrics:
            metric = f"RMSE {metrics['global_rmse']:.6g}"
        elif "binary_cross_entropy" in metrics:
            metric = f"log loss {metrics['binary_cross_entropy']:.6g}"
        else:
            metric = "—"
        references = row.get("reference_scores", {})
        reference = "—" if not references else f"{len(references)} methods (context only)"
        lines.append(
            f"| {row['task_id']} | {row['dataset_name']} | {metric} | "
            f"{'yes' if row['official_complete'] else 'no'} | {reference} |"
        )
    note = report.get("suite_summary", {}).get("reference_note")
    if note:
        lines.extend(["", f"> {note}"])
    return "\n".join(lines) + "\n"


__all__ = ["build_report", "report_csv", "report_markdown"]
