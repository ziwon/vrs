import json
from pathlib import Path

import numpy as np

from vrs.schemas import CandidateAlert, Detection, VerifiedAlert
from vrs.sinks.manifest_sink import ObjectManifestSink


def _alert(*, thumbnail_path: str | None = None) -> VerifiedAlert:
    return VerifiedAlert(
        candidate=CandidateAlert(
            class_name="fire",
            severity="critical",
            start_pts_s=0.0,
            peak_pts_s=1.0,
            peak_frame_index=4,
            peak_detections=[Detection(class_name="fire", score=0.9, xyxy=(1.0, 2.0, 10.0, 20.0))],
            keyframes=[np.zeros((32, 32, 3), dtype=np.uint8)],
            keyframe_pts=[1.0],
        ),
        true_alert=True,
        confidence=0.8,
        false_negative_class=None,
        rationale="visible fire",
        thumbnail_path=thumbnail_path,
    )


def test_manifest_sink_writes_verified_contract_and_evidence_ref(tmp_path: Path) -> None:
    thumb = tmp_path / "thumbnails" / "fire.jpg"
    thumb.parent.mkdir()
    thumb.write_bytes(b"fake-jpeg")
    sink = ObjectManifestSink(tmp_path, stream_id="cam-01", run_id="run-1")

    manifest = sink.write(_alert(thumbnail_path="thumbnails/fire.jpg"))

    path = tmp_path / "manifests" / f"{manifest['manifest_id']}.json"
    index = tmp_path / "object_manifest.index.jsonl"
    assert path.exists()
    assert index.exists()
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == manifest
    assert manifest["schema_version"] == "object_manifest.v1"
    assert manifest["stream_id"] == "cam-01"
    assert manifest["alert_id"] == manifest["records"][0]["alert_id"]
    assert manifest["event_id"] == manifest["records"][0]["event_id"]
    assert manifest["evidence_refs"][0]["schema_version"] == "evidence_ref.v1"
    assert manifest["evidence_refs"][0]["uri"] == thumb.resolve().as_uri()
    assert manifest["evidence_refs"][0]["sha256"]
    assert manifest["evidence_refs"][0]["metadata"]["relative_path"] == "thumbnails/fire.jpg"
    assert manifest["records"][0]["schema_version"] == "verified_alert.v1"
    assert manifest["records"][0]["evidence_refs"][0]["uri"] == thumb.resolve().as_uri()
    index_row = json.loads(index.read_text(encoding="utf-8"))
    assert index_row["schema_version"] == "object_manifest_index.v1"
    assert index_row["manifest_ref"]["kind"] == "metadata_manifest"


def test_manifest_sink_is_idempotent_for_same_alert_and_index(
    tmp_path: Path,
) -> None:
    thumb = tmp_path / "thumbnails" / "fire.jpg"
    thumb.parent.mkdir()
    thumb.write_bytes(b"fake-jpeg")
    sink = ObjectManifestSink(tmp_path, stream_id="cam-01")

    first = sink.write(_alert(thumbnail_path="thumbnails/fire.jpg"))
    manifest = sink.write(_alert(thumbnail_path="thumbnails/fire.jpg"))

    assert manifest == first
    assert manifest["manifest_id"] == first["manifest_id"]
    assert len(manifest["records"]) == 1
    assert len(manifest["evidence_refs"]) == 1
    assert (
        len((tmp_path / "object_manifest.index.jsonl").read_text(encoding="utf-8").splitlines())
        == 1
    )
