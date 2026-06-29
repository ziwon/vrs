"""Canonical VRS contract adapters.

These helpers keep the existing lightweight dataclasses stable while producing
versioned records suitable for DeepStream metadata export, event transport, and
object-storage manifests.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .schemas import CandidateAlert, Detection, Frame, VerifiedAlert


DETECTION_SCHEMA_VERSION = "detection.v1"
CANDIDATE_ALERT_SCHEMA_VERSION = "candidate_alert.v1"
VERIFIED_ALERT_SCHEMA_VERSION = "verified_alert.v1"
EVIDENCE_REF_SCHEMA_VERSION = "evidence_ref.v1"
STREAM_SCHEMA_VERSION = "stream.v1"
OBJECT_MANIFEST_SCHEMA_VERSION = "object_manifest.v1"


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def stable_id(prefix: str, *parts: Any) -> str:
    normalized = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def stable_stream_time(pts_s: float | None) -> str | None:
    return f"stream:{float(pts_s):.6f}" if pts_s is not None else None


def evidence_ref_v1(
    *,
    uri: str,
    kind: str,
    media_type: str,
    created_at: str | None = None,
    sha256: str | None = None,
    size_bytes: int | None = None,
    retention: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an object-storage-ready evidence reference."""

    out: dict[str, Any] = {
        "schema_version": EVIDENCE_REF_SCHEMA_VERSION,
        "uri": uri,
        "kind": kind,
        "media_type": media_type,
        "created_at": created_at or now_utc_iso(),
    }
    if sha256 is not None:
        out["sha256"] = sha256
    if size_bytes is not None:
        out["size_bytes"] = int(size_bytes)
    if retention:
        out["retention"] = dict(retention)
    if metadata:
        out["metadata"] = dict(metadata)
    return out


def detection_v1(
    detection: Detection,
    *,
    stream_id: str | None = None,
    clip_id: str | None = None,
    frame: Frame | None = None,
    frame_index: int | None = None,
    pts_s: float | None = None,
    detector_id: str | None = None,
    detection_id: str | None = None,
    source_id: str | None = None,
    source_runtime: str = "python",
    evidence_refs: list[dict[str, Any]] | None = None,
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Adapt the current ``Detection`` dataclass to ``detection.v1``."""

    if frame is not None:
        frame_index = frame.index
        pts_s = frame.pts_s

    resolved_detection_id = detection_id or stable_id(
        "det",
        source_runtime,
        stream_id,
        clip_id,
        frame_index,
        pts_s,
        detection.class_name,
        detection.track_id,
        ",".join(f"{float(x):.3f}" for x in detection.xyxy),
    )
    out: dict[str, Any] = {
        "schema_version": DETECTION_SCHEMA_VERSION,
        "record_type": "detection",
        "detection_id": resolved_detection_id,
        "idempotency_key": resolved_detection_id,
        "class_name": detection.class_name,
        "score": float(detection.score),
        "bbox_xyxy": [float(x) for x in detection.xyxy],
        "raw_label": detection.raw_label,
        "track_id": detection.track_id,
        "source_runtime": source_runtime,
        "observed_at": observed_at or stable_stream_time(pts_s) or now_utc_iso(),
        "evidence_refs": list(evidence_refs or []),
    }
    if stream_id is not None:
        out["stream_id"] = stream_id
    if source_id is not None:
        out["source_id"] = source_id
    if clip_id is not None:
        out["clip_id"] = clip_id
    if frame_index is not None:
        out["frame_index"] = int(frame_index)
    if pts_s is not None:
        out["pts_s"] = float(pts_s)
    if detector_id is not None:
        out["detector_id"] = detector_id
    return out


def candidate_alert_v1(
    alert: CandidateAlert,
    *,
    stream_id: str | None = None,
    source_id: str | None = None,
    policy_id: str | None = None,
    alert_id: str | None = None,
    event_id: str | None = None,
    source_runtime: str = "python",
    evidence_refs: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Adapt ``CandidateAlert`` to the canonical event-state output."""

    resolved_event_id = event_id or stable_id(
        "evt",
        stream_id,
        source_id,
        policy_id,
        alert.class_name,
        alert.track_id,
        alert.start_pts_s,
        alert.peak_pts_s,
        alert.peak_frame_index,
    )
    resolved_alert_id = alert_id or stable_id("cand", resolved_event_id, alert.severity)
    out: dict[str, Any] = {
        "schema_version": CANDIDATE_ALERT_SCHEMA_VERSION,
        "record_type": "candidate_alert",
        "alert_id": resolved_alert_id,
        "event_id": resolved_event_id,
        "idempotency_key": resolved_alert_id,
        "class_name": alert.class_name,
        "severity": alert.severity,
        "start_pts_s": float(alert.start_pts_s),
        "peak_pts_s": float(alert.peak_pts_s),
        "peak_frame_index": int(alert.peak_frame_index),
        "track_id": alert.track_id,
        "source_runtime": source_runtime,
        "created_at": created_at or stable_stream_time(alert.peak_pts_s) or now_utc_iso(),
        "peak_detections": [
            detection_v1(
                det,
                stream_id=stream_id,
                source_id=source_id,
                frame_index=alert.peak_frame_index,
                pts_s=alert.peak_pts_s,
                source_runtime=source_runtime,
            )
            for det in alert.peak_detections
        ],
        "keyframe_pts": [float(x) for x in alert.keyframe_pts],
        "num_keyframes": len(alert.keyframes),
        "evidence_refs": list(evidence_refs or []),
    }
    if stream_id is not None:
        out["stream_id"] = stream_id
    if source_id is not None:
        out["source_id"] = source_id
    if policy_id is not None:
        out["policy_id"] = policy_id
    return out


def verified_alert_v1(
    alert: VerifiedAlert,
    *,
    stream_id: str | None = None,
    source_id: str | None = None,
    policy_id: str | None = None,
    alert_id: str | None = None,
    candidate_alert_id: str | None = None,
    event_id: str | None = None,
    verifier_id: str | None = None,
    verification_id: str | None = None,
    verification_attempt: int = 1,
    verdict_version: str = "1",
    model_id: str | None = None,
    model_version: str | None = None,
    prompt_id: str | None = None,
    prompt_version: str | None = None,
    verifier_metadata: dict[str, Any] | None = None,
    source_runtime: str = "python",
    evidence_refs: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Adapt ``VerifiedAlert`` to the canonical verifier output."""

    bbox = alert.bbox_xywh_norm or alert.candidate.detector_bbox_xywh_norm()
    candidate = candidate_alert_v1(
        alert.candidate,
        stream_id=stream_id,
        source_id=source_id,
        policy_id=policy_id,
        alert_id=candidate_alert_id,
        event_id=event_id,
        source_runtime=source_runtime,
    )
    candidate_alert_id = str(candidate["alert_id"])
    resolved_event_id = str(candidate["event_id"])
    resolved_verification_id = verification_id or stable_id(
        "verify",
        candidate_alert_id,
        verifier_id or "default-verifier",
        verification_attempt,
        model_id,
        model_version,
        prompt_id,
        prompt_version,
    )
    resolved_alert_id = alert_id or stable_id(
        "ver",
        candidate_alert_id,
        verifier_id or "default-verifier",
        source_runtime,
    )
    out: dict[str, Any] = {
        "schema_version": VERIFIED_ALERT_SCHEMA_VERSION,
        "record_type": "verified_alert",
        "alert_id": resolved_alert_id,
        "event_id": resolved_event_id,
        "candidate_alert_id": candidate_alert_id,
        "verification_id": resolved_verification_id,
        "verification_attempt": int(verification_attempt),
        "verdict_version": verdict_version,
        "idempotency_key": resolved_alert_id,
        "candidate": candidate,
        "true_alert": bool(alert.true_alert),
        "confidence": float(alert.confidence),
        "false_negative_class": alert.false_negative_class,
        "rationale": alert.rationale,
        "bbox_xywh_norm": [float(x) for x in bbox] if bbox is not None else None,
        "trajectory_xy_norm": [[float(x), float(y)] for (x, y) in alert.trajectory_xy_norm],
        "verifier_raw": alert.verifier_raw,
        "verifier_json_valid": alert.verifier_json_valid,
        "thumbnail_path": alert.thumbnail_path,
        "incident_id": alert.incident_id,
        "incident_stream_ids": list(alert.incident_stream_ids),
        "incident_primary_stream_id": alert.incident_primary_stream_id,
        "verifier_id": verifier_id,
        "model_id": model_id,
        "model_version": model_version,
        "prompt_id": prompt_id,
        "prompt_version": prompt_version,
        "verifier_metadata": dict(verifier_metadata or {}),
        "source_runtime": source_runtime,
        "created_at": created_at or stable_stream_time(alert.candidate.peak_pts_s) or now_utc_iso(),
        "evidence_refs": list(evidence_refs or []),
    }
    if stream_id is not None:
        out["stream_id"] = stream_id
    if source_id is not None:
        out["source_id"] = source_id
    if policy_id is not None:
        out["policy_id"] = policy_id
    return out


def stream_v1(
    *,
    stream_id: str,
    source_uri: str,
    name: str | None = None,
    roi_polygon: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical stream source record."""

    out: dict[str, Any] = {
        "schema_version": STREAM_SCHEMA_VERSION,
        "record_type": "stream",
        "stream_id": stream_id,
        "source_uri": source_uri,
    }
    if name is not None:
        out["name"] = name
    if roi_polygon is not None:
        out["roi_polygon"] = list(roi_polygon)
    if metadata:
        out["metadata"] = dict(metadata)
    return out


def object_manifest_v1(
    *,
    manifest_id: str,
    run_id: str | None = None,
    stream_id: str | None = None,
    event_id: str | None = None,
    alert_id: str | None = None,
    evidence_refs: list[dict[str, Any]] | None = None,
    record_refs: list[dict[str, Any]] | None = None,
    records: list[dict[str, Any]] | None = None,
    created_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a manifest describing evidence objects and linked contracts."""

    out: dict[str, Any] = {
        "schema_version": OBJECT_MANIFEST_SCHEMA_VERSION,
        "record_type": "object_manifest",
        "manifest_id": manifest_id,
        "created_at": created_at or now_utc_iso(),
        "evidence_refs": list(evidence_refs or []),
        "record_refs": list(record_refs or []),
        "records": list(records or []),
    }
    if run_id is not None:
        out["run_id"] = run_id
    if stream_id is not None:
        out["stream_id"] = stream_id
    if event_id is not None:
        out["event_id"] = event_id
    if alert_id is not None:
        out["alert_id"] = alert_id
    if metadata:
        out["metadata"] = dict(metadata)
    return out
