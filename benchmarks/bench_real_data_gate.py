"""Gate implementation optimizations on fixed RF1 and SCM20D folds.

This is intentionally a small, sequential wall-clock benchmark rather than an
ASV benchmark.  It uses the prepared fold-0 study artifacts and the response
features selected by the completed leakage-safe standard fits.  The fixed
selection isolates implementation speed from feature-selection variability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mbtr import MBTRegressor

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GateDataset:
    split: Path
    response_features: tuple[int, ...]


DATASETS = {
    "rf1": GateDataset(
        ROOT / "studies/data/multivariate/sketchboost_random_holdout_cv5_2022/rf1/0.npz",
        (2, 0, 7, 4),
    ),
    "scm20d": GateDataset(
        ROOT / "studies/data/multivariate/sketchboost_random_holdout_cv5_2022/scm20d/0.npz",
        (45, 51, 56, 38, 1, 0, 44, 46),
    ),
}


def _parameters(
    n_estimators: int,
    engine: str,
    output_sketch_size: int | None,
) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "max_bins": 64,
        "min_samples_leaf": 20,
        "max_depth": 4,
        "l2_regularization": 0.1,
        "early_stopping_rounds": None,
        "random_state": 42,
        "engine": engine,
        "output_sketch_size": output_sketch_size,
        "verbose": 0,
    }


def _load(dataset: GateDataset) -> dict[str, np.ndarray]:
    if not dataset.split.is_file():
        raise FileNotFoundError(f"Prepared split not found: {dataset.split}")
    with np.load(dataset.split, allow_pickle=False) as archive:
        required = ("x_train", "y_train", "x_validation", "y_validation", "x_test", "y_test")
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"{dataset.split} is missing arrays: {', '.join(missing)}")
        return {name: np.ascontiguousarray(archive[name], dtype=np.float64) for name in required}


def _warm_numba(arrays: dict[str, np.ndarray], selected: tuple[int, ...]) -> None:
    """Compile the optional linear histogram kernel outside timed fits."""

    rows = min(128, len(arrays["x_train"]))
    x = arrays["x_train"][:rows]
    y = arrays["y_train"][:rows]
    design = x[:, selected]
    mean = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    design = np.ascontiguousarray((design - mean) / scale)
    MBTRegressor(
        n_estimators=1,
        learning_rate=0.05,
        max_bins=8,
        min_samples_leaf=2,
        max_depth=1,
        l2_regularization=0.1,
        engine="numba",
    ).fit(x, y)
    MBTRegressor(
        objective="linear_regression",
        n_estimators=1,
        learning_rate=0.05,
        max_bins=8,
        min_samples_leaf=2,
        max_depth=1,
        l2_regularization=0.1,
        engine="numba",
        refit_leaves=False,
    ).fit(x, y, response_features=design)


def _prediction_digest(prediction: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(prediction, dtype=np.float64)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _measurement(
    *,
    dataset_id: str,
    variant: str,
    model: Any,
    fit_seconds: float,
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    difference = np.asarray(prediction, dtype=np.float64) - target
    return {
        "dataset_id": dataset_id,
        "variant": variant,
        "fit_seconds": fit_seconds,
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "prediction_sha256": _prediction_digest(prediction),
        "resolved_engine": str(model.engine_),
        "n_estimators_fitted": int(
            model.n_estimators_ if hasattr(model, "n_estimators_") else model.linear_model_.n_estimators_
        ),
    }


def _run_dataset(
    dataset_id: str,
    *,
    n_estimators: int,
    engine: str,
    output_sketch_size: int | None,
    variants: set[str],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    specification = DATASETS[dataset_id]
    arrays = _load(specification)
    _warm_numba(arrays, specification.response_features)
    parameters = _parameters(n_estimators, engine, output_sketch_size)
    measurements: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}

    if "standard" in variants:
        standard = MBTRegressor(**parameters)
        started = time.perf_counter()
        standard.fit(
            arrays["x_train"],
            arrays["y_train"],
            eval_set=(arrays["x_validation"], arrays["y_validation"]),
        )
        fit_seconds = time.perf_counter() - started
        prediction = np.asarray(standard.predict(arrays["x_test"]), dtype=np.float64)
        measurement = _measurement(
            dataset_id=dataset_id,
            variant="standard",
            model=standard,
            fit_seconds=fit_seconds,
            prediction=prediction,
            target=arrays["y_test"],
        )
        measurements.append(measurement)
        predictions[f"{dataset_id}__standard"] = prediction
        print(
            f"{dataset_id} standard: fit={fit_seconds:.6f}s RMSE={measurement['rmse']:.12g} "
            f"engine={measurement['resolved_engine']}",
            file=sys.stderr,
            flush=True,
        )

    if "linear" in variants:
        # This CLI-only gate intentionally depends on the repository study
        # helpers.  Keep the import lazy so ASV can discover the installable
        # benchmark suite without packaging ``studies`` with MBTR.
        from studies.multivariate.models import SelectedLinearResponseRegressor

        linear = SelectedLinearResponseRegressor(specification.response_features, parameters)
        started = time.perf_counter()
        linear.fit(
            arrays["x_train"],
            arrays["y_train"],
            eval_set=(arrays["x_validation"], arrays["y_validation"]),
        )
        fit_seconds = time.perf_counter() - started
        prediction = np.asarray(linear.predict(arrays["x_test"]), dtype=np.float64)
        measurement = _measurement(
            dataset_id=dataset_id,
            variant="linear",
            model=linear,
            fit_seconds=fit_seconds,
            prediction=prediction,
            target=arrays["y_test"],
        )
        measurements.append(measurement)
        predictions[f"{dataset_id}__linear"] = prediction
        print(
            f"{dataset_id} linear: fit={fit_seconds:.6f}s RMSE={measurement['rmse']:.12g} "
            f"engine={measurement['resolved_engine']}",
            file=sys.stderr,
            flush=True,
        )
    return measurements, predictions


def _compare_predictions(
    predictions: dict[str, np.ndarray],
    reference_path: Path,
) -> dict[str, dict[str, Any]]:
    with np.load(reference_path, allow_pickle=False) as archive:
        comparisons: dict[str, dict[str, Any]] = {}
        for name, prediction in predictions.items():
            if name not in archive:
                raise ValueError(f"Reference predictions are missing {name!r}.")
            reference = np.asarray(archive[name], dtype=np.float64)
            if reference.shape != prediction.shape:
                raise ValueError(
                    f"Reference {name!r} has shape {reference.shape}; current shape is {prediction.shape}."
                )
            absolute = np.abs(reference - prediction)
            comparisons[name] = {
                "allclose_rtol_1e-12_atol_1e-12": bool(np.allclose(reference, prediction, rtol=1e-12, atol=1e-12)),
                "max_abs_difference": float(np.max(absolute, initial=0.0)),
            }
        return comparisons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        action="append",
        choices=tuple(DATASETS),
        help="Dataset to run; repeat to select both. Defaults to RF1 and SCM20D.",
    )
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--engine", choices=("auto", "numpy", "numba"), default="auto")
    parser.add_argument("--output-sketch-size", type=int)
    parser.add_argument(
        "--variant",
        action="append",
        choices=("standard", "linear"),
        help="Variant to run; repeat to select both. Defaults to both variants.",
    )
    parser.add_argument("--save-predictions", type=Path)
    parser.add_argument("--reference-predictions", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.n_estimators < 1:
        raise ValueError("--n-estimators must be positive.")
    if arguments.repeat < 1:
        raise ValueError("--repeat must be positive.")
    dataset_ids = arguments.dataset_id or list(DATASETS)
    variants = set(arguments.variant or ("standard", "linear"))
    measurements: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for repetition in range(arguments.repeat):
        for dataset_id in dataset_ids:
            dataset_measurements, dataset_predictions = _run_dataset(
                dataset_id,
                n_estimators=arguments.n_estimators,
                engine=arguments.engine,
                output_sketch_size=arguments.output_sketch_size,
                variants=variants,
            )
            for measurement in dataset_measurements:
                measurement["repetition"] = repetition
            measurements.extend(dataset_measurements)
            predictions.update(dataset_predictions)

    summaries: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        for variant in sorted(variants):
            selected = [
                item["fit_seconds"]
                for item in measurements
                if item["dataset_id"] == dataset_id and item["variant"] == variant
            ]
            if selected:
                summaries.append(
                    {
                        "dataset_id": dataset_id,
                        "variant": variant,
                        "fit_seconds_median": float(np.median(selected)),
                        "fit_seconds_min": float(np.min(selected)),
                        "fit_seconds_max": float(np.max(selected)),
                    }
                )

    result: dict[str, Any] = {
        "schema_version": 1,
        "n_estimators": arguments.n_estimators,
        "repeat": arguments.repeat,
        "measurements": measurements,
        "summaries": summaries,
    }
    if arguments.reference_predictions is not None:
        result["prediction_comparisons"] = _compare_predictions(
            predictions,
            arguments.reference_predictions,
        )
    if arguments.save_predictions is not None:
        arguments.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(arguments.save_predictions, **predictions)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(serialized, encoding="utf-8")
    sys.stdout.write(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
