"""Parse the fixed KDIS iterative-stratified multilabel archives.

This module deliberately performs no network access and writes no artifacts.  A
caller may download one of the pinned archives, pass its bytes to
``parse_kdis_archive``, and decide how to persist the returned folds.
"""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile, ZipInfo

import numpy as np
from numpy.typing import NDArray
from scipy.io import arff

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

KDIS_REPOSITORY_URL = "https://www.uco.es/kdis/mllresources/"
KDIS_PROTOCOL_ID = "kdis_iterative_5fold_2017"
KDIS_LICENSE_STATUS = "Dataset license not specified by the KDIS repository."
MULAN_LABEL_NAMESPACE = "http://mulan.sourceforge.net/labels"
_ARCHIVE_BASE_URL = (
    "https://www.uco.es/grupos/kdis/MLLResources/ucobigfiles/"
    "Datasets/Partitions"
)


@dataclass(frozen=True)
class KDISArchiveSpec:
    """Pinned identity and expected dimensions of one KDIS Mulan archive."""

    dataset_id: str
    dataset_name: str
    url: str
    size_bytes: int
    sha256: str
    n_samples: int
    n_features: int
    n_labels: int
    fold_sizes: tuple[tuple[int, int], ...]
    last_modified: str | None = None
    etag: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_name:
            raise ValueError("KDIS dataset identifiers must be non-empty.")
        if not self.url.startswith("https://"):
            raise ValueError("KDIS archive URLs must use HTTPS.")
        if self.size_bytes < 1:
            raise ValueError("KDIS archive size must be positive.")
        if len(self.sha256) != 64 or any(character not in "0123456789abcdef" for character in self.sha256):
            raise ValueError("KDIS archive SHA-256 must be 64 lowercase hexadecimal characters.")
        if min(self.n_samples, self.n_features, self.n_labels) < 1:
            raise ValueError("KDIS dataset dimensions must be positive.")
        if len(self.fold_sizes) != 5:
            raise ValueError("KDIS iterative-stratified archives must contain five folds.")
        for train_rows, test_rows in self.fold_sizes:
            if train_rows < 1 or test_rows < 1 or train_rows + test_rows != self.n_samples:
                raise ValueError("Every KDIS fold must be a non-empty partition of all samples.")

    @property
    def expected_members(self) -> tuple[str, ...]:
        """Return the exact flat member names expected in the Mulan ZIP."""

        tests = tuple(f"{self.dataset_name}-test{fold}.arff" for fold in range(1, 6))
        trains = tuple(f"{self.dataset_name}-train{fold}.arff" for fold in range(1, 6))
        return (*tests, *trains, f"{self.dataset_name}.xml")


@dataclass(frozen=True)
class KDISFold:
    """One parsed outer fold, ready to map onto the benchmark NPZ contract."""

    dataset_id: str
    run_id: str
    fold_number: int
    train_member: str
    test_member: str
    x_train: FloatArray
    y_train: IntArray
    x_test: FloatArray
    y_test: IntArray

    def arrays(self) -> dict[str, FloatArray | IntArray]:
        """Return an NPZ-compatible mapping without performing any writes."""

        return {
            "x_train": self.x_train,
            "y_train": self.y_train,
            "x_test": self.x_test,
            "y_test": self.y_test,
        }


@dataclass(frozen=True)
class PreparedKDISArchive:
    """Validated dataset metadata and all five parsed train/test folds."""

    spec: KDISArchiveSpec
    archive_sha256: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    folds: tuple[KDISFold, ...]


@dataclass(frozen=True)
class _ParsedPartition:
    x: FloatArray
    y: IntArray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]


KDIS_ARCHIVE_MANIFEST: Mapping[str, KDISArchiveSpec] = MappingProxyType(
    {
        "emotions": KDISArchiveSpec(
            dataset_id="emotions",
            dataset_name="Emotions",
            url=f"{_ARCHIVE_BASE_URL}/Emotions-Stratified5FoldsCV-Mulan.zip",
            size_bytes=794_355,
            sha256="f94c8d549027fe89ee1d1b9cb4492eb363a9fa874ca36649990de522cf43537f",
            n_samples=593,
            n_features=72,
            n_labels=6,
            fold_sizes=((477, 116), (475, 118), (471, 122), (475, 118), (474, 119)),
            last_modified="Tue, 13 Jun 2017 09:10:46 GMT",
            etag='"593fac16-c1ef3"',
        ),
        "scene": KDISArchiveSpec(
            dataset_id="scene",
            dataset_name="Scene",
            url=f"{_ARCHIVE_BASE_URL}/Scene-Stratified5FoldsCV-Mulan.zip",
            size_bytes=12_014_875,
            sha256="f1056e74c3d1f20e80ac70963ca4f80749561fcd76cba2fd6be0c7602d6e4d4b",
            n_samples=2_407,
            n_features=294,
            n_labels=6,
            fold_sizes=((1_926, 481), (1_928, 479), (1_922, 485), (1_924, 483), (1_928, 479)),
            last_modified="Tue, 13 Jun 2017 09:11:52 GMT",
            etag='"593fac58-b7551b"',
        ),
        "yeast": KDISArchiveSpec(
            dataset_id="yeast",
            dataset_name="Yeast",
            url=f"{_ARCHIVE_BASE_URL}/Yeast-Stratified5FoldsCV-Mulan.zip",
            size_bytes=4_294_549,
            sha256="a634193af1626a59cf44a91742e1392f700f8b9094d7b0665a53125aaf01e312",
            n_samples=2_417,
            n_features=103,
            n_labels=14,
            fold_sizes=((1_927, 490), (1_936, 481), (1_936, 481), (1_943, 474), (1_926, 491)),
            last_modified="Tue, 13 Jun 2017 09:12:40 GMT",
            etag='"593fac88-418795"',
        ),
    }
)


def get_kdis_archive_spec(dataset_id: str) -> KDISArchiveSpec:
    """Return the pinned archive specification for a supported dataset."""

    normalized = dataset_id.strip().lower()
    try:
        return KDIS_ARCHIVE_MANIFEST[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported KDIS dataset {dataset_id!r}; expected one of {sorted(KDIS_ARCHIVE_MANIFEST)}."
        ) from exc


def _validate_members(infos: list[ZipInfo], spec: KDISArchiveSpec) -> None:
    names: list[str] = []
    for info in infos:
        name = info.filename
        path = PurePosixPath(name)
        if (
            not name
            or "\\" in name
            or path.is_absolute()
            or len(path.parts) != 1
            or path.parts[0] in {".", ".."}
        ):
            raise ValueError(f"KDIS archive member {name!r} is not a safe flat filename.")
        if info.is_dir():
            raise ValueError(f"KDIS archive member {name!r} is a directory, not a regular file.")
        mode = (info.external_attr >> 16) & 0o170000
        if mode not in {0, stat.S_IFREG}:
            raise ValueError(f"KDIS archive member {name!r} is not a regular file.")
        if info.flag_bits & 0x1:
            raise ValueError(f"KDIS archive member {name!r} is encrypted.")
        names.append(name)

    if len(names) != len(set(names)):
        raise ValueError("KDIS archive contains duplicate member names.")
    expected = set(spec.expected_members)
    observed = set(names)
    if observed != expected:
        raise ValueError(
            "KDIS archive member mismatch: "
            f"missing={sorted(expected - observed)} unexpected={sorted(observed - expected)}."
        )


def _read_member(archive: ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except (BadZipFile, OSError, RuntimeError) as exc:
        raise ValueError(f"Could not read KDIS archive member {name!r}.") from exc


def _parse_label_names(payload: bytes, spec: KDISArchiveSpec) -> frozenset[str]:
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ValueError(f"KDIS label metadata for {spec.dataset_id!r} is not valid XML.") from exc

    expected_root = f"{{{MULAN_LABEL_NAMESPACE}}}labels"
    if root.tag != expected_root:
        raise ValueError("KDIS Mulan XML uses an unexpected root element or namespace.")
    elements = root.findall(f"{{{MULAN_LABEL_NAMESPACE}}}label")
    names = [element.get("name") for element in elements]
    if any(not name for name in names):
        raise ValueError("KDIS Mulan XML contains a label without a name.")
    concrete_names = [str(name) for name in names]
    if len(concrete_names) != len(set(concrete_names)):
        raise ValueError("KDIS Mulan XML contains duplicate label names.")
    if len(concrete_names) != spec.n_labels:
        raise ValueError(
            f"KDIS XML declares {len(concrete_names)} labels; expected {spec.n_labels}."
        )
    return frozenset(concrete_names)


def _decode_nominal(values: NDArray[np.generic], *, name: str) -> NDArray[np.str_]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, (bytes, np.bytes_)):
            try:
                decoded.append(bytes(value).decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError(f"KDIS label {name!r} contains invalid UTF-8 data.") from exc
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype=np.str_)


def _parse_arff(
    payload: bytes,
    *,
    member: str,
    label_set: frozenset[str],
    spec: KDISArchiveSpec,
) -> _ParsedPartition:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"KDIS ARFF member {member!r} is not valid UTF-8.") from exc
    try:
        data, metadata = arff.loadarff(StringIO(text))
    except (arff.ParseArffError, TypeError, ValueError) as exc:
        raise ValueError(f"KDIS ARFF member {member!r} could not be parsed.") from exc

    attribute_names = tuple(metadata.names())
    if len(attribute_names) != len(set(attribute_names)):
        raise ValueError(f"KDIS ARFF member {member!r} contains duplicate attributes.")
    target_names = tuple(name for name in attribute_names if name in label_set)
    feature_names = tuple(name for name in attribute_names if name not in label_set)
    if frozenset(target_names) != label_set:
        missing = sorted(label_set - frozenset(target_names))
        raise ValueError(f"KDIS ARFF member {member!r} is missing XML labels {missing}.")
    if len(feature_names) != spec.n_features or len(target_names) != spec.n_labels:
        raise ValueError(
            f"KDIS ARFF member {member!r} has {len(feature_names)} features and "
            f"{len(target_names)} labels; expected {spec.n_features} and {spec.n_labels}."
        )

    feature_columns: list[FloatArray] = []
    for name in feature_names:
        kind, _ = metadata[name]
        if kind != "numeric":
            raise ValueError(f"KDIS feature {name!r} in {member!r} is not numeric.")
        feature_columns.append(np.asarray(data[name], dtype=np.float64))
    x = np.column_stack(feature_columns).astype(np.float64, copy=False)
    if not len(x):
        raise ValueError(f"KDIS ARFF member {member!r} contains no rows.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"KDIS ARFF member {member!r} contains non-finite feature values.")

    target_columns: list[IntArray] = []
    for name in target_names:
        kind, domain = metadata[name]
        if kind != "nominal" or domain is None or set(domain) != {"0", "1"}:
            raise ValueError(f"KDIS label {name!r} in {member!r} is not declared binary {{0,1}}.")
        decoded = _decode_nominal(np.asarray(data[name]), name=name)
        observed = set(decoded.tolist())
        if not observed <= {"0", "1"}:
            raise ValueError(f"KDIS label {name!r} in {member!r} contains values outside {{0,1}}.")
        target_columns.append(decoded.astype(np.int64))
    y = np.column_stack(target_columns).astype(np.int64, copy=False)
    if not np.all((y == 0) | (y == 1)):
        raise ValueError(f"KDIS ARFF member {member!r} contains non-binary targets.")

    return _ParsedPartition(
        x=np.ascontiguousarray(x),
        y=np.ascontiguousarray(y),
        feature_names=feature_names,
        target_names=target_names,
    )


def parse_kdis_archive(
    payload: bytes | bytearray | memoryview,
    *,
    spec: KDISArchiveSpec,
) -> PreparedKDISArchive:
    """Validate and parse one pinned KDIS Mulan archive entirely in memory."""

    archive_bytes = bytes(payload)
    if len(archive_bytes) != spec.size_bytes:
        raise ValueError(
            f"KDIS archive size mismatch for {spec.dataset_id!r}: "
            f"observed {len(archive_bytes)}, expected {spec.size_bytes}."
        )
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if archive_sha256 != spec.sha256:
        raise ValueError(
            f"KDIS archive SHA-256 mismatch for {spec.dataset_id!r}: "
            f"observed {archive_sha256}, expected {spec.sha256}."
        )

    try:
        archive = ZipFile(BytesIO(archive_bytes))
    except (BadZipFile, OSError) as exc:
        raise ValueError(f"KDIS archive for {spec.dataset_id!r} is not a valid ZIP file.") from exc

    folds: list[KDISFold] = []
    feature_names: tuple[str, ...] | None = None
    target_names: tuple[str, ...] | None = None
    with archive:
        _validate_members(archive.infolist(), spec)
        xml_member = f"{spec.dataset_name}.xml"
        label_set = _parse_label_names(_read_member(archive, xml_member), spec)
        for fold_number, (expected_train_rows, expected_test_rows) in enumerate(spec.fold_sizes, start=1):
            train_member = f"{spec.dataset_name}-train{fold_number}.arff"
            test_member = f"{spec.dataset_name}-test{fold_number}.arff"
            train = _parse_arff(
                _read_member(archive, train_member),
                member=train_member,
                label_set=label_set,
                spec=spec,
            )
            test = _parse_arff(
                _read_member(archive, test_member),
                member=test_member,
                label_set=label_set,
                spec=spec,
            )
            if train.x.shape[0] != expected_train_rows or test.x.shape[0] != expected_test_rows:
                raise ValueError(
                    f"KDIS fold {fold_number} row mismatch: observed "
                    f"{train.x.shape[0]}/{test.x.shape[0]}, expected "
                    f"{expected_train_rows}/{expected_test_rows}."
                )
            if train.feature_names != test.feature_names or train.target_names != test.target_names:
                raise ValueError(f"KDIS fold {fold_number} train/test ARFF schemas differ.")
            if feature_names is None:
                feature_names = train.feature_names
                target_names = train.target_names
            elif train.feature_names != feature_names or train.target_names != target_names:
                raise ValueError("KDIS ARFF schemas differ across folds.")
            folds.append(
                KDISFold(
                    dataset_id=spec.dataset_id,
                    run_id=str(fold_number - 1),
                    fold_number=fold_number,
                    train_member=train_member,
                    test_member=test_member,
                    x_train=train.x,
                    y_train=train.y,
                    x_test=test.x,
                    y_test=test.y,
                )
            )

    if feature_names is None or target_names is None:
        raise ValueError("KDIS archive did not contain any parsed folds.")
    return PreparedKDISArchive(
        spec=spec,
        archive_sha256=archive_sha256,
        feature_names=feature_names,
        target_names=target_names,
        folds=tuple(folds),
    )


def load_kdis_archive(path: str | Path, dataset_id: str) -> PreparedKDISArchive:
    """Read and parse a local archive using its fixed official specification."""

    spec = get_kdis_archive_spec(dataset_id)
    return parse_kdis_archive(Path(path).read_bytes(), spec=spec)


__all__ = [
    "KDIS_ARCHIVE_MANIFEST",
    "KDIS_LICENSE_STATUS",
    "KDIS_PROTOCOL_ID",
    "KDIS_REPOSITORY_URL",
    "KDISArchiveSpec",
    "KDISFold",
    "PreparedKDISArchive",
    "get_kdis_archive_spec",
    "load_kdis_archive",
    "parse_kdis_archive",
]
