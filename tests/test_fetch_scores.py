from __future__ import annotations

import hashlib
import io
import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

import studies.fetch_scores as fetch_scores


class Response(io.BytesIO):
    status = 200
    headers = {"Content-Type": "text/csv", "ETag": '"fixture"'}


def fixture_source() -> tuple[dict[str, Any], dict[str, bytes]]:
    results = (
        b"task_id,learner_id,rmse\n"
        b"alpha,a,1.0\n"
        b"alpha,b,2.0\n"
        b"beta,a,3.0\n"
        b"beta,b,0.0\n"
    )
    task_ids = b"id\n101\n202\n"
    revision = "a" * 40
    artifacts = {"results": results, "task_ids": task_ids}
    source = {
        "id": "fixture",
        "suite_id": 353,
        "revision": revision,
        "format": "ctr23_csv_v1",
        "schema_version": 1,
        "artifacts": {
            artifact_id: {
                "filename": f"{artifact_id}.csv",
                "url": f"https://raw.githubusercontent.com/org/repo/{revision}/{artifact_id}.csv",
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for artifact_id, data in artifacts.items()
        },
        "expected_rows": 4,
        "methods": {"a": "Method A", "b": "Method B"},
        "task_map": [
            {"source_task_name": "alpha", "openml_task_id": 101},
            {"source_task_name": "beta", "openml_task_id": 202},
        ],
        "metric": "mean_resample_rmse",
        "comparison_tier": "context_only",
    }
    return source, artifacts


def install_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: dict[str, Any]) -> None:
    registry = tmp_path / "remote_sources.json"
    registry.write_text(json.dumps([source]), encoding="utf-8")
    monkeypatch.setattr(fetch_scores, "SOURCE_FILE", registry)


def artifact_opener(source: dict[str, Any], artifacts: dict[str, bytes]):
    by_url = {
        specification["url"]: artifacts[artifact_id]
        for artifact_id, specification in source["artifacts"].items()
    }

    def open_request(request: Any) -> Response:
        return Response(by_url[request.full_url])

    return open_request


def test_ctr23_normalization_uses_exact_numeric_task_map() -> None:
    source, artifacts = fixture_source()

    rows = fetch_scores.validate_score_data(source, artifacts)

    assert len(rows) == 4
    assert rows[0] == {
        "comparison_tier": "context_only",
        "method": "Method A",
        "metric": "mean_resample_rmse",
        "openml_task_id": 101,
        "rmse": 1.0,
        "source_line": 2,
        "source_method": "a",
        "source_task_name": "alpha",
        "value": 1.0,
    }
    assert fetch_scores.scores_by_openml_task(rows)[202] == {"Method A": 3.0, "Method B": 0.0}
    assert fetch_scores.scores_by_dataset(rows)["alpha"] == {"Method A": 1.0, "Method B": 2.0}


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (b"name,learner_id,rmse\n", "columns"),
        (b"task_id,learner_id,rmse\nalpha,a,nan\n", "Invalid CTR23"),
        (b"task_id,learner_id,rmse\nalpha,a,-1\n", "Invalid CTR23"),
        (
            b"task_id,learner_id,rmse\nalpha,a,1\nalpha,a,2\n",
            "Duplicate CTR23",
        ),
    ],
)
def test_ctr23_result_parser_rejects_schema_and_values(replacement: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        fetch_scores.parse_ctr23_scores(replacement)


def test_validation_rejects_hash_drift_and_incomplete_cross_product() -> None:
    source, artifacts = fixture_source()
    bad_hash = dict(source)
    bad_hash["artifacts"] = {name: dict(value) for name, value in source["artifacts"].items()}
    bad_hash["artifacts"]["results"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Checksum mismatch"):
        fetch_scores.validate_score_data(bad_hash, artifacts)

    incomplete_results = artifacts["results"].replace(b"beta,b,0.0\n", b"")
    incomplete = {**artifacts, "results": incomplete_results}
    incomplete_source = dict(source)
    incomplete_source["artifacts"] = {name: dict(value) for name, value in source["artifacts"].items()}
    incomplete_source["artifacts"]["results"]["sha256"] = hashlib.sha256(incomplete_results).hexdigest()
    with pytest.raises(ValueError, match="cross-product"):
        fetch_scores.validate_score_data(incomplete_source, incomplete)


def test_registry_rejects_mutable_or_unpinned_urls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, _ = fixture_source()
    source["artifacts"]["results"]["url"] = (
        "https://raw.githubusercontent.com/org/repo/main/results.csv"
    )
    install_source(monkeypatch, tmp_path, source)

    with pytest.raises(ValueError, match="pinned revision|mutable branch"):
        fetch_scores.load_remote_sources()


def test_fetch_cache_hit_and_corrupt_offline_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, artifacts = fixture_source()
    install_source(monkeypatch, tmp_path, source)
    cache = tmp_path / "cache"
    calls = 0
    real_opener = artifact_opener(source, artifacts)

    def counted_opener(request: Any) -> Response:
        nonlocal calls
        calls += 1
        return real_opener(request)

    directory, rows = fetch_scores.fetch_score_source("fixture", cache, opener=counted_opener)
    assert calls == 2
    assert len(rows) == 4
    assert json.loads((directory / "metadata.json").read_text())["revision"] == "a" * 40

    def network_bomb(request: Any) -> Response:
        raise AssertionError(f"unexpected network request: {request.full_url}")

    cached_directory, cached_rows = fetch_scores.fetch_score_source(
        "fixture", cache, offline=True, opener=network_bomb
    )
    assert cached_directory == directory
    assert cached_rows == rows
    (directory / "results.csv").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cached score source.*invalid"):
        fetch_scores.fetch_score_source("fixture", cache, offline=True, opener=network_bomb)


def test_failed_refresh_preserves_previous_valid_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source, artifacts = fixture_source()
    install_source(monkeypatch, tmp_path, source)
    cache = tmp_path / "cache"
    directory, _ = fetch_scores.fetch_score_source(
        "fixture", cache, opener=artifact_opener(source, artifacts)
    )
    before = {path.name: path.read_bytes() for path in directory.glob("*.csv")}
    calls = 0

    def failing_opener(request: Any) -> Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise urllib.error.URLError("fixture failure")
        return Response(artifacts["results"])

    with pytest.raises(urllib.error.URLError):
        fetch_scores.fetch_score_source("fixture", cache, refresh=True, opener=failing_opener)
    assert {path.name: path.read_bytes() for path in directory.glob("*.csv")} == before
    _, rows = fetch_scores.fetch_score_source("fixture", cache, offline=True)
    assert len(rows) == 4


def test_pinned_ctr23_registry_is_complete() -> None:
    source = fetch_scores.load_remote_sources()["openml_ctr23_results"]

    assert source["revision"] == "eb9c32031d5a000dd7cac51c55f03dcc546d3d7d"
    assert len(source["task_map"]) == 35
    assert source["expected_rows"] == 175
    assert source["artifacts"]["results"]["sha256"] == (
        "3d68cac3c94f898e5df753db33cd7a25bcc568625e5a7d83f002b3520ff7d58d"
    )
    assert source["artifacts"]["task_ids"]["sha256"] == (
        "1f8a701fcf913481d390c862891ef8f40df5df843f0c1210e48c99f8d2d1b381"
    )
