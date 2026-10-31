from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("sklearn")

from studies.multivariate.prepare_mocr import (  # noqa: E402
    _preprocess_like_upstream,
    _split_sizes,
    prepare_mocr_runs,
)
from studies.multivariate.prepare_sketchboost import (  # noqa: E402
    _fit_mean_imputation,
    _parse_first_split_column,
    _parse_xc_data,
)


def test_xc_sparse_parser_handles_rows_with_and_without_positive_labels() -> None:
    payload = b"3 3 2\n0,1 0:1 2:0.5\n0:2\n1 1:3\n"

    x, y = _parse_xc_data(payload, (3, 3, 2))

    np.testing.assert_array_equal(x, [[1.0, 0.0, 0.5], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    np.testing.assert_array_equal(y, [[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
    assert x.dtype == np.float32
    assert y.dtype == np.float32


def test_xc_split_parser_uses_only_first_one_based_column_and_preserves_order() -> None:
    payload = b"3 9 9 9 9 9 9 9 9 9\n1 8 8 8 8 8 8 8 8 8\n2 7 7 7 7 7 7 7 7 7\n"

    np.testing.assert_array_equal(_parse_first_split_column(payload, rows=3), [2, 0, 1])


def test_sketchboost_imputation_is_fit_only_on_training_rows() -> None:
    train = np.asarray([[1.0, np.nan], [3.0, 5.0]], dtype=np.float32)
    validation = np.asarray([[100.0, np.nan]], dtype=np.float32)
    test = np.asarray([[np.nan, 9.0]], dtype=np.float32)

    transformed, means = _fit_mean_imputation(train, validation, test)

    np.testing.assert_array_equal(means, [2.0, 5.0])
    np.testing.assert_array_equal(transformed[0], [[1.0, 5.0], [3.0, 5.0]])
    np.testing.assert_array_equal(transformed[1], [[100.0, 5.0]])
    np.testing.assert_array_equal(transformed[2], [[2.0, 9.0]])


def test_mocr_split_sizes_match_upstream_calibration_cap() -> None:
    assert _split_sizes(7_207) == (2_920, 758, 2_048, 1_481)
    assert _split_sizes(21_613) == (10_123, 3_639, 2_048, 5_803)
    assert _split_sizes(45_730) == (22_182, 8_463, 2_048, 13_037)
    assert _split_sizes(50_000) == (24_317, 9_317, 2_048, 14_318)


def test_mocr_preprocessing_preserves_the_pinned_mask_quirks() -> None:
    rows = 120
    frame = pd.DataFrame(
        {
            "continuous": np.arange(rows, dtype=np.int64),
            "low_integer": np.arange(rows, dtype=np.int64) % 5,
            "binary_float": np.arange(rows, dtype=np.float64) % 2,
            "pseudo_float": np.ones(rows, dtype=np.float64),
            "high_cardinality_text": [f"value_{index % 25}" for index in range(rows)],
            "mostly_missing": [np.nan] * 30 + list(np.arange(90, dtype=np.float64)),
            "one_missing_row": np.arange(rows, dtype=np.float64),
        }
    )
    frame.loc[7, "one_missing_row"] = np.nan
    target = pd.DataFrame({"y0": np.arange(rows), "y1": np.arange(rows) + 1})

    x, y, names, targets, positions, metadata = _preprocess_like_upstream(frame, target)

    assert names == ("continuous", "low_integer", "binary_float", "one_missing_row")
    assert targets == ("y0", "y1")
    assert x.shape == (119, 4)
    assert y.shape == (119, 2)
    assert 7 not in positions
    assert metadata["removed_high_cardinality_columns"] == ["high_cardinality_text"]
    assert metadata["removed_pseudo_categorical_columns"] == ["pseudo_float"]
    assert metadata["removed_missing_columns"] == ["mostly_missing"]


def test_torch_26_randperm_matches_pinned_mocr_golden_identity() -> None:
    torch = pytest.importorskip("torch")
    if torch.__version__.split("+")[0] != "2.6.0":
        pytest.skip("Golden identity is pinned to torch 2.6.0")
    values = torch.randperm(7_207, generator=torch.Generator().manual_seed(2_000)).numpy()

    np.testing.assert_array_equal(values[:5], [1906, 2049, 5928, 4260, 3987])
    assert hashlib.sha256(values.astype("<i8").tobytes()).hexdigest() == (
        "6198cd199162594134e7a6394faeaa935151752704a8eb3dc59fb36ce5b72c1d"
    )


def test_mocr_households_run_zero_source_positions_match_upstream() -> None:
    source = Path("studies/data/multivariate-raw/mocr")
    if not source.is_dir():
        pytest.skip("Pinned MOCR raw checkout is not present")
    run = next(prepare_mocr_runs(source, ["mocr_households"]))

    np.testing.assert_array_equal(run.arrays["source_index_train"][:5], [4904, 1647, 4694, 1983, 3864])
    digest = hashlib.sha256(run.arrays["source_index_train"].astype("<i8").tobytes()).hexdigest()
    assert digest == "48b1e0d5a9d0cd28a304deadc976d1bdbdcf06f3c1452c85e5cd125b9bfdcf77"
    assert run.split["source_seed"] == 2000
    assert run.split["source_index_space"].startswith("zero-based positional CSV")
