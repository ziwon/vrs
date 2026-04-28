"""Audit signing tests for tamper-evident alert JSONL output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from vrs.audit import GENESIS_HASH, AuditConfig, verify_jsonl
from vrs.schemas import CandidateAlert, Detection, VerifiedAlert
from vrs.sinks import JsonlSink


def _alert(class_name: str = "fire", peak: float = 1.0) -> VerifiedAlert:
    return VerifiedAlert(
        candidate=CandidateAlert(
            class_name=class_name,
            severity="critical",
            start_pts_s=0.0,
            peak_pts_s=peak,
            peak_frame_index=int(peak * 4),
            peak_detections=[
                Detection(class_name=class_name, score=0.9, xyxy=(1.0, 2.0, 3.0, 4.0))
            ],
            keyframes=[np.zeros((4, 4, 3), dtype=np.uint8)],
            keyframe_pts=[peak],
        ),
        true_alert=True,
        confidence=0.95,
        false_negative_class=None,
        rationale=f"{class_name} present",
    )


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_records(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_jsonl_sink_unsigned_output_remains_backward_compatible(tmp_path: Path):
    path = tmp_path / "alerts.jsonl"

    with JsonlSink(path) as sink:
        sink.write(_alert())

    record = _read_records(path)[0]
    assert "record_hash" not in record
    assert "prev_hash" not in record
    assert "schema_version" not in record


def test_signed_hmac_chain_verifies(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VRS_AUDIT_HMAC_KEY", "test-secret")
    path = tmp_path / "alerts.jsonl"
    audit = {
        "enabled": True,
        "mode": "hmac_sha256",
        "key_id": "local-dev-key",
        "key_env": "VRS_AUDIT_HMAC_KEY",
    }

    with JsonlSink(path, audit=audit) as sink:
        sink.write(_alert("fire", 1.0))
        sink.write(_alert("smoke", 2.0))

    records = _read_records(path)
    assert records[0]["prev_hash"] == GENESIS_HASH
    assert records[1]["prev_hash"] == records[0]["record_hash"]
    assert records[0]["key_id"] == "local-dev-key"

    result = verify_jsonl(path, mode="hmac_sha256", key_env="VRS_AUDIT_HMAC_KEY")
    assert result.valid
    assert result.checked_records == 2


def test_modified_line_is_detected(tmp_path: Path, monkeypatch):
    path = _signed_log(tmp_path, monkeypatch)
    records = _read_records(path)
    records[0]["confidence"] = 0.01
    _write_records(path, records)

    result = verify_jsonl(path, mode="hmac_sha256", key_env="VRS_AUDIT_HMAC_KEY")

    assert not result.valid
    assert any("record_hash mismatch" in error for error in result.errors)


def test_deleted_line_is_detected(tmp_path: Path, monkeypatch):
    path = _signed_log(tmp_path, monkeypatch)
    records = _read_records(path)
    _write_records(path, [records[0], records[2]])

    result = verify_jsonl(path, mode="hmac_sha256", key_env="VRS_AUDIT_HMAC_KEY")

    assert not result.valid
    assert any("prev_hash does not match" in error for error in result.errors)


def test_reordered_lines_are_detected(tmp_path: Path, monkeypatch):
    path = _signed_log(tmp_path, monkeypatch)
    records = _read_records(path)
    _write_records(path, [records[1], records[0], records[2]])

    result = verify_jsonl(path, mode="hmac_sha256", key_env="VRS_AUDIT_HMAC_KEY")

    assert not result.valid
    assert any("prev_hash does not match" in error for error in result.errors)


def test_wrong_hmac_key_is_detected(tmp_path: Path, monkeypatch):
    path = _signed_log(tmp_path, monkeypatch)
    monkeypatch.setenv("VRS_AUDIT_HMAC_KEY", "wrong-secret")

    result = verify_jsonl(path, mode="hmac_sha256", key_env="VRS_AUDIT_HMAC_KEY")

    assert not result.valid
    assert any("record_hash mismatch" in error for error in result.errors)


def test_verification_cli(tmp_path: Path, monkeypatch):
    path = _signed_log(tmp_path, monkeypatch)
    env = {**os.environ, "VRS_AUDIT_HMAC_KEY": "test-secret"}

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vrs.audit",
            "--log",
            str(path),
            "--mode",
            "hmac_sha256",
            "--key-env",
            "VRS_AUDIT_HMAC_KEY",
        ],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0
    assert "OK: verified 3 signed record" in proc.stdout


def test_sha256_mode_does_not_require_key(tmp_path: Path):
    path = tmp_path / "alerts.jsonl"
    with JsonlSink(path, audit=AuditConfig(enabled=True, mode="sha256")) as sink:
        sink.write(_alert())

    assert verify_jsonl(path, mode="sha256").valid


def _signed_log(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("VRS_AUDIT_HMAC_KEY", "test-secret")
    path = tmp_path / "alerts.jsonl"
    with JsonlSink(
        path,
        audit={
            "enabled": True,
            "mode": "hmac_sha256",
            "key_id": "local-dev-key",
            "key_env": "VRS_AUDIT_HMAC_KEY",
        },
    ) as sink:
        sink.write(_alert("fire", 1.0))
        sink.write(_alert("smoke", 2.0))
        sink.write(_alert("weapon", 3.0))
    return path
