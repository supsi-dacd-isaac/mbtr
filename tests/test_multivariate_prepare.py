from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from studies.multivariate.prepare_common import PreparedRun, validate_prepared_run, write_prepared_run


def _run(*, task: str = "regression") -> PreparedRun:
    y_train = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    y_test = np.asarray([[5.0, 6.0]])
    if task == "multilabel":
        y_train = np.asarray([[0, 1], [1, 0]], dtype=np.int8)
        y_test = np.asarray([[1, 1]], dtype=np.int8)
    return PreparedRun(
        dataset_id="fixture",
        protocol_id="fixture_protocol",
        run_id="0",
        task=task,
        arrays={
            "x_train": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "y_train": y_train,
            "x_test": np.asarray([[5.0, 6.0]]),
            "y_test": y_test,
            "source_index_train": np.asarray([2, 0]),
            "source_index_test": np.asarray([1]),
        },
        feature_names=("a", "b"),
        target_names=("y0", "y1"),
        source={"url": "https://example.test/data", "sha256": "abc"},
        preprocessing={"fit_partition": "training only"},
        split={"seed": 42},
    )


def test_writer_is_atomic_validated_and_idempotent(tmp_path) -> None:
    run = _run()
    first = write_prepared_run(run, tmp_path)
    second = write_prepared_run(run, tmp_path)

    assert not first.reused
    assert second.reused
    assert first.split_sha256 == second.split_sha256
    sidecar = json.loads(first.sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["split_sha256"] == first.split_sha256
    assert sidecar["preparation_fingerprint"] == first.preparation_fingerprint
    assert sidecar["dimensions"] == {
        "calibration_rows": 0,
        "features": 2,
        "outputs": 2,
        "test_rows": 1,
        "train_rows": 2,
        "validation_rows": 0,
    }
    with np.load(first.split_path, allow_pickle=False) as archive:
        assert set(archive.files) == set(run.arrays)
        np.testing.assert_array_equal(archive["source_index_train"], [2, 0])


def test_writer_rejects_stale_outputs_unless_replace_is_explicit(tmp_path) -> None:
    run = _run()
    first = write_prepared_run(run, tmp_path)
    changed = replace(run, split={"seed": 43})

    with pytest.raises(FileExistsError, match="differs or is incomplete"):
        write_prepared_run(changed, tmp_path)
    replaced = write_prepared_run(changed, tmp_path, replace=True)

    assert not replaced.reused
    assert replaced.preparation_fingerprint != first.preparation_fingerprint


def test_writer_rejects_tampered_sidecar_metadata(tmp_path) -> None:
    run = _run()
    written = write_prepared_run(run, tmp_path)
    sidecar = json.loads(written.sidecar_path.read_text(encoding="utf-8"))
    sidecar["feature_names"][0] = "tampered"
    written.sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(FileExistsError, match="differs or is incomplete"):
        write_prepared_run(run, tmp_path)


def test_multilabel_validation_rejects_nonbinary_targets() -> None:
    run = _run(task="multilabel")
    validate_prepared_run(run)
    arrays = dict(run.arrays)
    arrays["y_test"] = np.asarray([[1, 2]])

    with pytest.raises(ValueError, match="only 0 and 1"):
        validate_prepared_run(replace(run, arrays=arrays))


def test_validation_rejects_mismatched_names_and_partial_auxiliary_pairs() -> None:
    run = _run()
    with pytest.raises(ValueError, match="feature names"):
        validate_prepared_run(replace(run, feature_names=("only_one",)))

    arrays = dict(run.arrays)
    arrays["x_validation"] = np.ones((1, 2))
    with pytest.raises(ValueError, match="provided together"):
        validate_prepared_run(replace(run, arrays=arrays))


@pytest.mark.parametrize(
    ("indices", "message"),
    [
        (np.asarray([[0, 1]]), "one-dimensional integer"),
        (np.asarray([0.0, 1.0]), "one-dimensional integer"),
        (np.asarray([0]), "one source position"),
        (np.asarray([0, 0]), "unique non-negative"),
        (np.asarray([-1, 0]), "unique non-negative"),
    ],
)
def test_validation_checks_source_index_contract(indices, message: str) -> None:
    run = _run()
    arrays = dict(run.arrays)
    arrays["source_index_train"] = indices

    with pytest.raises(ValueError, match=message):
        validate_prepared_run(replace(run, arrays=arrays))


def test_validation_rejects_source_index_overlap() -> None:
    run = _run()
    arrays = dict(run.arrays)
    arrays["source_index_test"] = np.asarray([2])

    with pytest.raises(ValueError, match="overlapping source positions"):
        validate_prepared_run(replace(run, arrays=arrays))
