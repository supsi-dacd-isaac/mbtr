from __future__ import annotations

import hashlib
from dataclasses import replace
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from studies.multivariate.prepare_kdis import (
    KDIS_ARCHIVE_MANIFEST,
    KDISArchiveSpec,
    get_kdis_archive_spec,
    parse_kdis_archive,
)


def _arff(
    rows: tuple[str, ...],
    *,
    feature_kind: str = "numeric",
    label_domain: str = "{0,1}",
) -> str:
    return "\n".join(
        (
            "@relation Tiny",
            "",
            f"@attribute feature_z {feature_kind}",
            f"@attribute label_a {label_domain}",
            "@attribute feature_x numeric",
            f"@attribute label_b {label_domain}",
            "",
            "@data",
            *rows,
            "",
        )
    )


def _archive(
    *,
    train_arff: str | None = None,
    test_arff: str | None = None,
    xml_member: str = "Tiny.xml",
) -> bytes:
    train = train_arff or _arff(("1,1,10,0", "2,0,20,1"))
    test = test_arff or _arff(("3,1,30,1",))
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<labels xmlns="http://mulan.sourceforge.net/labels">'
        '<label name="label_b"/><label name="label_a"/>'
        "</labels>"
    )
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for fold in range(1, 6):
            archive.writestr(f"Tiny-test{fold}.arff", test)
            archive.writestr(f"Tiny-train{fold}.arff", train)
        archive.writestr(xml_member, xml)
    return stream.getvalue()


def _spec(payload: bytes) -> KDISArchiveSpec:
    return KDISArchiveSpec(
        dataset_id="tiny",
        dataset_name="Tiny",
        url="https://example.test/Tiny-Stratified5FoldsCV-Mulan.zip",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        n_samples=3,
        n_features=2,
        n_labels=2,
        fold_sizes=((2, 1),) * 5,
    )


def test_fixed_manifest_pins_official_mulan_archives() -> None:
    expected = {
        "emotions": (
            794_355,
            "f94c8d549027fe89ee1d1b9cb4492eb363a9fa874ca36649990de522cf43537f",
        ),
        "scene": (
            12_014_875,
            "f1056e74c3d1f20e80ac70963ca4f80749561fcd76cba2fd6be0c7602d6e4d4b",
        ),
        "yeast": (
            4_294_549,
            "a634193af1626a59cf44a91742e1392f700f8b9094d7b0665a53125aaf01e312",
        ),
    }

    assert set(KDIS_ARCHIVE_MANIFEST) == set(expected)
    for dataset_id, (size_bytes, sha256) in expected.items():
        spec = get_kdis_archive_spec(dataset_id.upper())
        assert spec.size_bytes == size_bytes
        assert spec.sha256 == sha256
        assert spec.url.startswith("https://")
        assert spec.url.endswith(f"{spec.dataset_name}-Stratified5FoldsCV-Mulan.zip")
        assert len(spec.expected_members) == 11


def test_parser_uses_xml_as_a_set_and_preserves_arff_declaration_order() -> None:
    payload = _archive()
    prepared = parse_kdis_archive(payload, spec=_spec(payload))

    assert prepared.feature_names == ("feature_z", "feature_x")
    assert prepared.target_names == ("label_a", "label_b")
    assert prepared.archive_sha256 == hashlib.sha256(payload).hexdigest()
    assert tuple(fold.run_id for fold in prepared.folds) == ("0", "1", "2", "3", "4")
    assert tuple(fold.fold_number for fold in prepared.folds) == (1, 2, 3, 4, 5)

    first = prepared.folds[0]
    np.testing.assert_array_equal(first.x_train, [[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_array_equal(first.y_train, [[1, 0], [0, 1]])
    np.testing.assert_array_equal(first.x_test, [[3.0, 30.0]])
    np.testing.assert_array_equal(first.y_test, [[1, 1]])
    assert first.x_train.dtype == np.float64
    assert first.y_train.dtype == np.int64
    assert set(first.arrays()) == {"x_train", "y_train", "x_test", "y_test"}


def test_parser_rejects_size_and_hash_mismatches_before_parsing() -> None:
    payload = _archive()
    spec = _spec(payload)

    with pytest.raises(ValueError, match="size mismatch"):
        parse_kdis_archive(payload, spec=replace(spec, size_bytes=len(payload) + 1))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        parse_kdis_archive(payload, spec=replace(spec, sha256="0" * 64))


def test_parser_rejects_traversal_members_even_when_archive_hash_matches() -> None:
    payload = _archive(xml_member="../Tiny.xml")

    with pytest.raises(ValueError, match="safe flat filename"):
        parse_kdis_archive(payload, spec=_spec(payload))


@pytest.mark.parametrize(
    ("train_arff", "message"),
    [
        (_arff(("?,1,10,0", "2,0,20,1")), "non-finite feature values"),
        (
            _arff(("1,2,10,0", "2,0,20,1"), label_domain="{0,1,2}"),
            "not declared binary",
        ),
        (
            _arff(("low,1,10,0", "high,0,20,1"), feature_kind="{low,high}"),
            "is not numeric",
        ),
    ],
)
def test_parser_rejects_invalid_feature_and_label_data(train_arff: str, message: str) -> None:
    payload = _archive(train_arff=train_arff)

    with pytest.raises(ValueError, match=message):
        parse_kdis_archive(payload, spec=_spec(payload))
