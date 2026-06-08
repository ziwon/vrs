from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.make_fixture_runs import write_fixture_run
from vrs.web.api import create_app
from vrs.web.artifacts import RunArtifactStore, UnsafePathError


def test_web_api_import_does_not_load_heavy_modules() -> None:
    code = (
        "import sys; import vrs.web.api; "
        "mods={'torch','ultralytics','transformers','cv2'} & set(sys.modules); "
        "print(','.join(sorted(mods))); raise SystemExit(1 if mods else 0)"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_single_stream_run_lists_and_reads_alerts(tmp_path: Path) -> None:
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    runs = client.get("/api/runs").json()["runs"]
    fixture = next(run for run in runs if run["name"] == "fixture")
    assert fixture["layout"] == "single"
    assert fixture["alert_count"] == 4

    response = client.get("/api/runs/fixture/alerts")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 4
    assert body["errors"] == []
    assert body["alerts"][0]["class_name"] == "fire"
    assert body["alerts"][0]["thumbnail_url"].startswith("/api/runs/fixture/thumbnails/")


def test_streams_endpoint_reports_rtsp_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRS_SAMPLE_RTSP_URL", "rtsp://example.local:8554/falldown")
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    body = client.get("/api/streams").json()
    assert body["rtsp_sample"]["url"] == "rtsp://example.local:8554/falldown"
    assert body["rtsp_sample"]["path"] == "falldown"
    assert any(stream["id"] == "falldown" for stream in body["streams"])
    assert {"cam-01", "cam-02"}.issubset({stream["id"] for stream in body["streams"]})


def test_multi_stream_run_lists_and_filters_by_stream(tmp_path: Path) -> None:
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    runs = client.get("/api/runs").json()["runs"]
    fixture = next(run for run in runs if run["name"] == "fixture_multi")
    assert fixture["layout"] == "multi"
    assert fixture["streams"] == ["cam-01", "cam-02"]
    assert fixture["alert_count"] == 4

    body = client.get("/api/runs/fixture_multi/alerts", params={"stream_id": "cam-02"}).json()
    assert body["total"] == 2
    assert {alert["stream_id"] for alert in body["alerts"]} == {"cam-02"}
    assert body["alerts"][0]["thumbnail_url"].endswith("?stream_id=cam-02")


def test_missing_run_and_missing_alerts_are_empty(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    client = TestClient(create_app(tmp_path))

    runs = client.get("/api/runs").json()["runs"]
    assert runs == [{"name": "empty", "layout": "empty", "streams": [], "alert_count": 0, "updated_at": None}]

    body = client.get("/api/runs/does-not-exist/alerts").json()
    assert body["run"]["layout"] == "empty"
    assert body["alerts"] == []
    assert body["errors"] == []


def test_malformed_jsonl_line_reports_error_without_crashing(tmp_path: Path) -> None:
    run = tmp_path / "bad"
    run.mkdir()
    (run / "alerts.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"class_name": "fire", "severity": "critical", "true_alert": True}),
                "{not json",
                json.dumps(["not", "an", "object"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    body = client.get("/api/runs/bad/alerts").json()
    assert body["total"] == 1
    assert len(body["alerts"]) == 1
    assert [(error["line"], error["error"]) for error in body["errors"]] == [
        (2, "Expecting property name enclosed in double quotes"),
        (3, "expected JSON object"),
    ]


def test_alert_filters_limit_offset_and_since_line(tmp_path: Path) -> None:
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    filtered = client.get(
        "/api/runs/fixture/alerts",
        params={"class_name": "fire", "severity": "critical", "true_alert": "true"},
    ).json()
    assert filtered["total"] == 1
    assert filtered["alerts"][0]["class_name"] == "fire"

    false_alerts = client.get("/api/runs/fixture/alerts", params={"true_alert": "false"}).json()
    assert false_alerts["total"] == 1
    assert false_alerts["alerts"][0]["class_name"] == "falldown"

    page = client.get("/api/runs/fixture/alerts", params={"offset": 1, "limit": 2}).json()
    assert page["total"] == 4
    assert [alert["class_name"] for alert in page["alerts"]] == ["smoke", "falldown"]

    tail = client.get("/api/runs/fixture/tail", params={"since_line": 2}).json()
    assert [alert["_line"] for alert in tail["alerts"]] == [3, 4]


def test_thumbnail_serving_and_path_traversal_prevention(tmp_path: Path) -> None:
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/runs/fixture/thumbnails/thumbnails/fire.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    store = RunArtifactStore(tmp_path)
    with pytest.raises(UnsafePathError):
        store.thumbnail_path("fixture", "../fixture/alerts.jsonl")

    escaped = client.get("/api/runs/fixture/thumbnails/%2e%2e/fixture/alerts.jsonl")
    assert escaped.status_code in {400, 404}
