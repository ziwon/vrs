import json
from pathlib import Path

import numpy as np
import pytest

from vrs.contracts import evidence_ref_v1, object_manifest_v1, stream_v1
from vrs.schemas import CandidateAlert, Detection, Frame, VerifiedAlert


def test_contract_schema_files_are_versioned_json() -> None:
    schema_dir = Path("contracts/schemas")
    names = {
        "detection.v1.schema.json",
        "candidate_alert.v1.schema.json",
        "verified_alert.v1.schema.json",
        "evidence_ref.v1.schema.json",
        "stream.v1.schema.json",
        "object_manifest.v1.schema.json",
    }
    assert {p.name for p in schema_dir.glob("*.schema.json")} >= names
    for name in names:
        payload = json.loads((schema_dir / name).read_text(encoding="utf-8"))
        assert payload["title"] == name.removesuffix(".schema.json")


def test_detection_to_contract_preserves_existing_dataclass_shape() -> None:
    frame = Frame(index=7, pts_s=1.75, image=np.zeros((10, 20, 3), dtype=np.uint8))
    det = Detection(
        class_name="fire",
        score=0.92,
        xyxy=(1, 2, 3, 4),
        raw_label="open flame",
        track_id=42,
    )

    contract = det.to_contract(stream_id="cam-01", frame=frame, detector_id="yoloe")

    assert contract["schema_version"] == "detection.v1"
    assert contract["record_type"] == "detection"
    assert contract["stream_id"] == "cam-01"
    assert contract["frame_index"] == 7
    assert contract["pts_s"] == 1.75
    assert contract["bbox_xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert contract["track_id"] == 42
    assert contract["detector_id"] == "yoloe"
    assert contract["detection_id"].startswith("det_")
    assert contract["idempotency_key"] == contract["detection_id"]
    assert contract == det.to_contract(stream_id="cam-01", frame=frame, detector_id="yoloe")


def test_candidate_and_verified_contracts_keep_legacy_json_unchanged() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    cand = CandidateAlert(
        class_name="smoke",
        severity="high",
        start_pts_s=3.0,
        peak_pts_s=4.0,
        peak_frame_index=16,
        peak_detections=[
            Detection(class_name="smoke", score=0.81, xyxy=(20, 10, 80, 50), track_id=5)
        ],
        keyframes=[image],
        keyframe_pts=[3.5, 4.0],
        track_id=5,
    )
    verified = VerifiedAlert(
        candidate=cand,
        true_alert=True,
        confidence=0.73,
        false_negative_class=None,
        rationale="visible smoke",
        trajectory_xy_norm=[(0.2, 0.3)],
        thumbnail_path="thumbnails/smoke.jpg",
    )

    legacy = verified.to_json()
    contract = verified.to_contract(stream_id="cam-yard")
    retried = VerifiedAlert(
        candidate=cand,
        true_alert=False,
        confidence=0.11,
        false_negative_class="fire",
        rationale="retry produced a different verifier outcome",
        trajectory_xy_norm=[(0.4, 0.5)],
        thumbnail_path="thumbnails/smoke.jpg",
    ).to_contract(stream_id="cam-yard")

    assert "schema_version" not in legacy
    assert contract["schema_version"] == "verified_alert.v1"
    assert contract["stream_id"] == "cam-yard"
    assert contract["alert_id"].startswith("ver_")
    assert contract["event_id"].startswith("evt_")
    assert contract["candidate_alert_id"] == contract["candidate"]["alert_id"]
    assert contract["idempotency_key"] == contract["alert_id"]
    assert contract["candidate"]["schema_version"] == "candidate_alert.v1"
    assert contract["candidate"]["idempotency_key"] == contract["candidate"]["alert_id"]
    assert contract["candidate"]["peak_detections"][0]["schema_version"] == "detection.v1"
    assert contract["verification_id"].startswith("verify_")
    assert contract["verification_attempt"] == 1
    assert contract["verdict_version"] == "1"
    assert contract["bbox_xywh_norm"] == pytest.approx([0.1, 0.1, 0.3, 0.4])
    assert contract["trajectory_xy_norm"] == [[0.2, 0.3]]
    assert contract["thumbnail_path"] == "thumbnails/smoke.jpg"
    assert retried["event_id"] == contract["event_id"]
    assert retried["candidate_alert_id"] == contract["candidate_alert_id"]
    assert retried["alert_id"] == contract["alert_id"]
    assert retried["idempotency_key"] == contract["idempotency_key"]


def test_verified_contract_keeps_candidate_and_verified_ids_separate() -> None:
    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=1.0,
        peak_pts_s=2.0,
        peak_frame_index=8,
        peak_detections=[Detection(class_name="fire", score=0.91, xyxy=(1, 2, 30, 40))],
    )
    verified = VerifiedAlert(
        candidate=cand,
        true_alert=True,
        confidence=0.8,
        false_negative_class=None,
        rationale="visible fire",
    )

    contract = verified.to_contract(
        stream_id="cam-yard",
        alert_id="verifier-alert-1",
        candidate_alert_id="candidate-alert-1",
        verifier_id="vlm-a",
        verification_id="verify-1",
        verification_attempt=2,
        verdict_version="policy-2026-06-30",
        model_id="cosmos-reason2",
        model_version="w4a16",
        prompt_id="safety",
        prompt_version="v3",
    )

    assert contract["alert_id"] == "verifier-alert-1"
    assert contract["candidate_alert_id"] == "candidate-alert-1"
    assert contract["candidate"]["alert_id"] == "candidate-alert-1"
    assert contract["verification_id"] == "verify-1"
    assert contract["verification_attempt"] == 2
    assert contract["verdict_version"] == "policy-2026-06-30"
    assert contract["model_id"] == "cosmos-reason2"
    assert contract["model_version"] == "w4a16"
    assert contract["prompt_id"] == "safety"
    assert contract["prompt_version"] == "v3"


def test_evidence_stream_and_manifest_contracts() -> None:
    evidence = evidence_ref_v1(
        uri="s3://vrs-evidence/run-1/cam-01/frame.jpg",
        kind="thumbnail",
        media_type="image/jpeg",
        sha256="abc123",
        size_bytes=1234,
    )
    stream = stream_v1(stream_id="cam-01", source_uri="rtsp://example.local/live")
    manifest = object_manifest_v1(
        manifest_id="manifest-1",
        run_id="run-1",
        stream_id="cam-01",
        event_id="evt-1",
        alert_id="alert-1",
        evidence_refs=[evidence],
        record_refs=[{"uri": "s3://vrs-evidence/run-1/cam-01/record.json"}],
        records=[stream],
    )

    assert evidence["schema_version"] == "evidence_ref.v1"
    assert stream["schema_version"] == "stream.v1"
    assert manifest["schema_version"] == "object_manifest.v1"
    assert manifest["evidence_refs"][0]["uri"].startswith("s3://")
    assert manifest["event_id"] == "evt-1"
    assert manifest["alert_id"] == "alert-1"
    assert manifest["record_refs"][0]["uri"].endswith("record.json")
    assert manifest["records"][0]["record_type"] == "stream"
