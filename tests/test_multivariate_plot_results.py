from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from studies.multivariate.plot_results import (
    EXTERNAL_VARIANTS,
    METHOD_LABELS,
    _complete_datasets,
    _matrix,
    _quality_ratio,
    _time_ratio,
    generate_figures,
)


def _aggregate() -> dict[str, Any]:
    ratios = (0.8, 0.9, 1.1, 1.2)
    fit_seconds = (5.0, 10.0, 20.0, 40.0)
    external_variants = {
        variant: {
            "complete": True,
            "over_standard_ratio": ratio,
            "timing": {"fit_seconds": {"median": fit_time}},
        }
        for variant, ratio, fit_time in zip(
            EXTERNAL_VARIANTS, ratios, fit_seconds, strict=True
        )
    }
    return {
        "datasets": [
            {
                "dataset_id": "rf1",
                "task": "regression",
                "standard_complete": True,
                "timing": {
                    "standard_fit_seconds": {"median": 10.0},
                    "two_stage_fit_seconds": {"median": 0.01},
                },
                "paired_comparison": {
                    "metrics": {
                        "global_rmse": {
                            "linear_over_standard_ratio": {"mean": 0.001}
                        }
                    }
                },
                "external_variants": external_variants,
            }
        ]
    }


def test_plot_matrices_use_only_external_models() -> None:
    datasets = _complete_datasets(_aggregate(), ("rf1",))

    quality = _matrix(datasets, _quality_ratio)
    timing = _matrix(datasets, _time_ratio)

    assert METHOD_LABELS == (
        "XGBoost independent",
        "LightGBM independent",
        "CatBoost shared",
        "XGBoost shared",
    )
    assert quality.shape == (1, 4)
    np.testing.assert_allclose(quality, [[0.8, 0.9, 1.1, 1.2]])
    np.testing.assert_allclose(timing, [[0.5, 1.0, 2.0, 4.0]])
    assert 0.001 not in quality
    assert 0.001 not in timing


def test_generate_figures_writes_accessible_external_only_svgs(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    aggregate_path = tmp_path / "aggregate.json"
    aggregate_path.write_text(json.dumps(_aggregate()), encoding="utf-8")

    quality_path, timing_path = generate_figures(
        aggregate_path,
        tmp_path,
        dataset_ids=("rf1",),
    )

    quality_svg = quality_path.read_text(encoding="utf-8")
    timing_svg = timing_path.read_text(encoding="utf-8")
    for svg in (quality_svg, timing_svg):
        assert 'role="img"' in svg
        assert '<desc id="plot-description">' in svg
        assert "four external methods" in svg
        assert "XGBoost independent" in svg
        assert "LightGBM independent" in svg
        assert "CatBoost shared" in svg
        assert "XGBoost shared" in svg
        assert "Linear MBTR" not in svg
        assert "not run" not in svg

    assert "0.800×" in quality_svg
    assert "1.200×" in quality_svg
    assert "0.500×" in timing_svg
    assert "4.000×" in timing_svg
