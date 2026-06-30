from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from base64 import urlsafe_b64encode
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vrs.api.api import create_app
from vrs.api.artifacts import RunArtifactStore, UnsafePathError
from vrs.schemas import CandidateAlert, Detection, VerifiedAlert
from vrs.sinks.jsonl_sink import JsonlSink

_FIXTURE_SCRIPT = Path("scripts/make_fixture_runs.py")
_FIXTURE_SPEC = importlib.util.spec_from_file_location("make_fixture_runs", _FIXTURE_SCRIPT)
assert _FIXTURE_SPEC is not None
assert _FIXTURE_SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_FIXTURE_SPEC)
sys.modules[_FIXTURE_SPEC.name] = _FIXTURE
_FIXTURE_SPEC.loader.exec_module(_FIXTURE)
write_fixture_run = _FIXTURE.write_fixture_run


def test_web_api_import_does_not_load_heavy_modules() -> None:
    code = (
        "import sys; import vrs.api.api; "
        "mods={'torch','ultralytics','transformers','cv2'} & set(sys.modules); "
        "print(','.join(sorted(mods))); raise SystemExit(1 if mods else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
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


def test_streams_endpoint_reports_rtsp_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VRS_SAMPLE_RTSP_URL", "rtsp://example.local:8554/falldown")
    streams_path = tmp_path / "streams.yaml"
    streams_path.write_text(
        """
streams:
  - id: falldown
    name: Falldown local
    location: fixture
    rtsp: rtsp://example.local:8554/falldown
    fps: 30
  - id: fire
    name: Fire local
    location: fixture
    rtsp: rtsp://example.local:8554/fire
    fps: 30
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VRS_STREAMS_PATH", str(streams_path))
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    body = client.get("/api/streams").json()
    assert body["rtsp_sample"]["url"] == "rtsp://example.local:8554/falldown"
    assert body["rtsp_sample"]["path"] == "falldown"
    assert any(stream["id"] == "fire" for stream in body["streams"])
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
    assert runs == [
        {"name": "empty", "layout": "empty", "streams": [], "alert_count": 0, "updated_at": None}
    ]

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
    assert tail["next_cursor"]


def test_tail_cursor_is_per_stream(tmp_path: Path) -> None:
    run = tmp_path / "fixture_multi"
    cam01 = run / "cam-01"
    cam02 = run / "cam-02"
    cam01.mkdir(parents=True)
    cam02.mkdir(parents=True)
    with (cam01 / "alerts.jsonl").open("w", encoding="utf-8") as fh:
        for idx in range(100):
            fh.write(json.dumps({"class_name": "fire", "severity": "critical", "idx": idx}) + "\n")
    (cam02 / "alerts.jsonl").write_text(
        json.dumps({"class_name": "smoke", "severity": "high", "idx": 1}) + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    first = client.get("/api/runs/fixture_multi/tail", params={"limit": 500}).json()
    cursor = first["next_cursor"]
    assert max(alert["_line"] for alert in first["alerts"] if alert["stream_id"] == "cam-01") == 100

    with (cam02 / "alerts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"class_name": "weapon", "severity": "critical", "idx": 2}) + "\n")

    second = client.get("/api/runs/fixture_multi/tail", params={"cursor": cursor}).json()
    assert [
        (alert["stream_id"], alert["_line"], alert["class_name"]) for alert in second["alerts"]
    ] == [("cam-02", 2, "weapon")]


@pytest.mark.parametrize(
    "cursor",
    [
        "not valid!",
        urlsafe_b64encode(b"\xff").decode("ascii").rstrip("="),
        urlsafe_b64encode(b"[]").decode("ascii").rstrip("="),
        urlsafe_b64encode(json.dumps({"../cam": 1}).encode("utf-8")).decode("ascii").rstrip("="),
        urlsafe_b64encode(json.dumps({"cam-01": "1"}).encode("utf-8")).decode("ascii").rstrip("="),
        urlsafe_b64encode(json.dumps({"cam-01": -1}).encode("utf-8")).decode("ascii").rstrip("="),
        urlsafe_b64encode(json.dumps({"cam-01": True}).encode("utf-8")).decode("ascii").rstrip("="),
    ],
)
def test_tail_rejects_malformed_cursor_values(tmp_path: Path, cursor: str) -> None:
    write_fixture_run(tmp_path)
    client = TestClient(create_app(tmp_path))

    response = client.get("/api/runs/fixture_multi/tail", params={"cursor": cursor})

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid tail cursor"


def test_tail_latest_mode_prefers_newest_alert_over_old_stream_backlog(tmp_path: Path) -> None:
    run = tmp_path / "fixture_multi"
    cam01 = run / "cam-01"
    cam02 = run / "cam-02"
    cam01.mkdir(parents=True)
    cam02.mkdir(parents=True)
    with (cam01 / "alerts.jsonl").open("w", encoding="utf-8") as fh:
        for idx in range(100):
            fh.write(
                json.dumps(
                    {
                        "stream_id": "cam-01",
                        "class_name": "fire",
                        "severity": "critical",
                        "ts": f"2026-01-01T00:00:{idx % 60:02d}Z",
                        "peak_pts_s": idx,
                    }
                )
                + "\n"
            )
    (cam02 / "alerts.jsonl").write_text(
        json.dumps(
            {
                "stream_id": "cam-02",
                "class_name": "smoke",
                "severity": "high",
                "ts": "2026-01-02T00:00:00Z",
                "peak_pts_s": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(tmp_path))

    first = client.get("/api/runs/fixture_multi/tail", params={"mode": "latest", "limit": 1}).json()

    assert [(alert["stream_id"], alert["class_name"]) for alert in first["alerts"]] == [
        ("cam-02", "smoke")
    ]

    with (cam02 / "alerts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "stream_id": "cam-02",
                    "class_name": "weapon",
                    "severity": "critical",
                    "ts": "2026-01-02T00:00:01Z",
                    "peak_pts_s": 2,
                }
            )
            + "\n"
        )

    second = client.get(
        "/api/runs/fixture_multi/tail", params={"cursor": first["next_cursor"]}
    ).json()

    assert [
        (alert["stream_id"], alert["_line"], alert["class_name"]) for alert in second["alerts"]
    ] == [("cam-02", 2, "weapon")]


def test_jsonl_cache_invalidates_when_file_grows(tmp_path: Path) -> None:
    run = tmp_path / "cache"
    run.mkdir()
    path = run / "alerts.jsonl"
    path.write_text(json.dumps({"class_name": "fire"}) + "\n", encoding="utf-8")
    store = RunArtifactStore(tmp_path)

    assert store.read_alerts("cache").total == 1
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"class_name": "smoke"}) + "\n")
    result = store.read_alerts("cache")

    assert result.total == 2
    assert [alert["class_name"] for alert in result.alerts] == ["fire", "smoke"]


def test_jsonl_sink_adds_written_at_without_breaking_shape(tmp_path: Path) -> None:
    candidate = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=1.0,
        peak_frame_index=5,
        peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(1, 2, 3, 4))],
    )
    alert = VerifiedAlert(
        candidate=candidate,
        true_alert=True,
        confidence=0.8,
        false_negative_class=None,
        rationale="test",
    )
    path = tmp_path / "alerts.jsonl"

    with JsonlSink(path) as sink:
        sink.write(alert)

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["class_name"] == "fire"
    assert record["written_at"].endswith("Z")


def test_policy_endpoint_reads_configured_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """
watch:
  - name: test-event
    detector: ["test prompt"]
    verifier: "test verifier"
    severity: low
    min_score: 0.1
    min_persist_frames: 1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("VRS_POLICY_PATH", str(policy_path))
    client = TestClient(create_app(tmp_path))

    body = client.get("/api/policy").json()

    assert body["path"] == str(policy_path)
    assert body["watch"][0]["name"] == "test-event"


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
