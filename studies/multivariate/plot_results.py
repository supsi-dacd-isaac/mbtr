"""Generate README figures from a multivariate benchmark aggregate."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

EXTERNAL_VARIANTS = (
    "xgboost",
    "lightgbm",
    "catboost",
    "xgboost_multi_output",
)
METHOD_LABELS = (
    "XGBoost independent",
    "LightGBM independent",
    "CatBoost shared",
    "XGBoost shared",
)
METHOD_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
METHOD_HATCHES = ("", "///", "\\\\", "xx")
DATASET_LABELS = {
    "rf1": "RF1",
    "scm20d": "SCM20D",
    "mocr_households": "Households",
    "mocr_house": "House",
    "mocr_bio": "Bio",
    "mocr_blog_data": "Blog",
    "mocr_taxi": "Taxi",
    "mediamill": "Mediamill",
    "delicious": "Delicious",
}
README_SNAPSHOT_DATASET_IDS = (
    "rf1",
    "scm20d",
    "mocr_households",
    "mocr_house",
    "mocr_bio",
    "mocr_blog_data",
    "mocr_taxi",
    "mediamill",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path(
            "studies/results/"
            "multivariate-all-models-mbtr-refresh-20261031-es1000-p50/aggregate.json"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("docs/source/_static"),
    )
    parser.add_argument(
        "--dataset-id",
        action="append",
        dest="dataset_ids",
        help=(
            "Dataset to include, in display order; repeat to override the pinned "
            "eight-dataset README snapshot."
        ),
    )
    return parser


def _complete_datasets(
    aggregate: dict[str, Any],
    dataset_ids: Sequence[str] = README_SNAPSHOT_DATASET_IDS,
) -> list[dict[str, Any]]:
    """Keep only datasets with complete standard and external-model runs."""

    datasets_by_id = {str(dataset["dataset_id"]): dataset for dataset in aggregate["datasets"]}
    completed: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        dataset = datasets_by_id.get(dataset_id)
        if dataset is None:
            continue
        if not dataset.get("standard_complete"):
            continue
        variants = dataset.get("external_variants", {})
        if not all(variants.get(name, {}).get("complete") for name in EXTERNAL_VARIANTS):
            continue
        completed.append(dataset)
    return completed


def _quality_ratio(dataset: dict[str, Any], method_index: int) -> float:
    value = dataset["external_variants"][EXTERNAL_VARIANTS[method_index]].get(
        "over_standard_ratio"
    )
    return np.nan if value is None else float(value)


def _time_ratio(dataset: dict[str, Any], method_index: int) -> float:
    standard = dataset.get("timing", {}).get("standard_fit_seconds", {}).get("median")
    if standard is None or float(standard) <= 0.0:
        return np.nan
    value = (
        dataset["external_variants"][EXTERNAL_VARIANTS[method_index]]
        .get("timing", {})
        .get("fit_seconds", {})
        .get("median")
    )
    return np.nan if value is None else float(value) / float(standard)


def _matrix(
    datasets: list[dict[str, Any]],
    value_function: Callable[[dict[str, Any], int], float],
) -> np.ndarray:
    return np.asarray(
        [
            [value_function(dataset, method_index) for method_index in range(len(METHOD_LABELS))]
            for dataset in datasets
        ],
        dtype=np.float64,
    )


def _plot_grouped_bars(
    values: np.ndarray,
    *,
    dataset_labels: list[str],
    title: str,
    x_axis_label: str,
    description: str,
    output: Path,
    logarithmic: bool,
) -> None:
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on optional plotting extra
        raise RuntimeError(
            "Generating benchmark figures requires the 'plot' optional dependencies."
        ) from error

    if values.shape != (len(dataset_labels), len(METHOD_LABELS)):
        raise ValueError(
            "Bar-plot values must have one row per dataset and one column per external method."
        )
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Bar-plot ratios must all be finite and positive.")

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "mbtr-multivariate-benchmark",
        }
    )
    height = 3.2 + 0.75 * len(dataset_labels)
    figure, axis = plt.subplots(figsize=(12.5, height), constrained_layout=True)
    dataset_positions = np.arange(len(dataset_labels), dtype=np.float64)
    bar_height = 0.16
    offsets = (
        np.arange(len(METHOD_LABELS), dtype=np.float64)
        - (len(METHOD_LABELS) - 1) / 2.0
    ) * (bar_height + 0.025)

    axis.axvline(1.0, color="#303030", linewidth=1.4, linestyle="--", zorder=1)
    for method_index, (method_label, color, hatch) in enumerate(
        zip(METHOD_LABELS, METHOD_COLORS, METHOD_HATCHES, strict=True)
    ):
        method_values = values[:, method_index]
        positions = dataset_positions + offsets[method_index]
        axis.barh(
            positions,
            method_values - 1.0,
            left=1.0,
            height=bar_height,
            color=color,
            edgecolor="#202020",
            linewidth=0.55,
            hatch=hatch,
            label=method_label,
            zorder=3,
        )
        for position, value in zip(positions, method_values, strict=True):
            value_is_lower = bool(value < 1.0)
            axis.annotate(
                f"{value:.3f}×",
                xy=(float(value), float(position)),
                xytext=(-4 if value_is_lower else 4, 0),
                textcoords="offset points",
                ha="right" if value_is_lower else "left",
                va="center",
                fontsize=8,
                color="#202020",
                zorder=4,
            )

    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", pad=52)
    axis.set_xlabel(x_axis_label, labelpad=10)
    axis.set_yticks(dataset_positions, labels=dataset_labels)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.tick_params(axis="x", length=4)
    axis.invert_yaxis()
    axis.xaxis.grid(True, color="#D8D8D8", linewidth=0.8, zorder=0)
    for boundary in np.arange(len(dataset_labels) - 1, dtype=np.float64) + 0.5:
        axis.axhline(boundary, color="#E8E8E8", linewidth=0.8, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=len(METHOD_LABELS),
        frameon=False,
        borderaxespad=0.0,
        handlelength=2.0,
        columnspacing=1.5,
    )

    if logarithmic:
        log_values = np.log10(np.append(values.ravel(), 1.0))
        log_span = max(float(np.ptp(log_values)), 1.0)
        padding = 0.12 * log_span
        axis.set_xscale("log")
        axis.set_xlim(
            10.0 ** (float(np.min(log_values)) - padding),
            10.0 ** (float(np.max(log_values)) + padding),
        )
        ticks = [0.25, 0.5, 1.0, 2.0, 4.0]
        axis.set_xticks(ticks, labels=[f"{value:g}×" for value in ticks])
        axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    else:
        minimum = min(float(np.min(values)), 1.0)
        maximum = max(float(np.max(values)), 1.0)
        span = max(maximum - minimum, 0.2)
        axis.set_xlim(max(0.0, minimum - 0.14 * span), maximum + 0.14 * span)
        axis.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda value, _: f"{value:g}×")
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_format = output.suffix.removeprefix(".") or "svg"
    figure.savefig(
        output,
        format=output_format,
        bbox_inches="tight",
        metadata={
            "Creator": "MBTR multivariate benchmark",
            "Date": None,
            "Title": title,
            "Description": description,
        },
    )
    plt.close(figure)
    if output_format.lower() == "svg":
        svg = output.read_text(encoding="utf-8")
        svg = svg.replace(
            "<svg ",
            '<svg role="img" aria-labelledby="plot-title plot-description" ',
            1,
        )
        svg = svg.replace("<title>", '<title id="plot-title">', 1)
        svg = svg.replace(
            "</title>",
            f'</title>\n <desc id="plot-description">{escape(description)}</desc>',
            1,
        )
        output.write_text(svg, encoding="utf-8")


def generate_figures(
    aggregate_path: Path,
    output_directory: Path,
    dataset_ids: Sequence[str] = README_SNAPSHOT_DATASET_IDS,
) -> tuple[Path, Path]:
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    datasets = _complete_datasets(aggregate, dataset_ids)
    if not datasets:
        raise ValueError("The aggregate contains no complete standard-plus-external datasets.")
    dataset_labels = [
        DATASET_LABELS.get(str(dataset["dataset_id"]), str(dataset["dataset_id"]))
        for dataset in datasets
    ]
    quality = _matrix(datasets, _quality_ratio)
    timing = _matrix(datasets, _time_ratio)
    quality_path = output_directory / "multivariate-benchmark-quality.svg"
    timing_path = output_directory / "multivariate-benchmark-time.svg"
    _plot_grouped_bars(
        quality,
        dataset_labels=dataset_labels,
        title="External-model headline test loss relative to standard MBTR",
        x_axis_label="Test-loss ratio; bars extend from standard MBTR at 1× (lower is better)",
        description=(
            f"{len(datasets)} complete datasets by four external methods. "
            "Grouped horizontal bars extend from standard MBTR at a ratio of 1 "
            "to each method's headline test-loss ratio; lower and farther left "
            "is better."
        ),
        output=quality_path,
        logarithmic=False,
    )
    _plot_grouped_bars(
        timing,
        dataset_labels=dataset_labels,
        title="External-model fit time relative to standard MBTR",
        x_axis_label="Median fit-time ratio; bars extend from 1× (lower is faster, log scale)",
        description=(
            f"{len(datasets)} complete datasets by four external methods. "
            "Grouped horizontal bars on a logarithmic axis extend from standard "
            "MBTR at a ratio of 1 to each method's median fit-time ratio; lower "
            "and farther left is faster."
        ),
        output=timing_path,
        logarithmic=True,
    )
    return quality_path, timing_path


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    dataset_ids = arguments.dataset_ids or README_SNAPSHOT_DATASET_IDS
    outputs = generate_figures(arguments.aggregate, arguments.output_directory, dataset_ids)
    print("\n".join(str(path) for path in outputs))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
