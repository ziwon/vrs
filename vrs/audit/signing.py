"""Hash-chain and HMAC helpers for tamper-evident alert JSONL records."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "vrs.alert.v1"
GENESIS_HASH = "0" * 64
SUPPORTED_MODES = {"sha256", "hmac_sha256"}


@dataclass(frozen=True)
class AuditConfig:
    """Runtime configuration for optional audit signing."""

    enabled: bool = False
    mode: str = "hmac_sha256"
    key_id: str | None = None
    key_env: str | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> AuditConfig:
        raw = raw or {}
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=str(raw.get("mode", "hmac_sha256")),
            key_id=raw.get("key_id"),
            key_env=raw.get("key_env"),
        )


@dataclass
class VerificationResult:
    valid: bool
    checked_records: int = 0
    unsigned_records: int = 0
    errors: list[str] = field(default_factory=list)


def _canonical_bytes(record: dict[str, Any]) -> bytes:
    return json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _resolve_key(config: AuditConfig, *, explicit_key: str | None = None) -> bytes | None:
    key = explicit_key
    if key is None and config.key_env:
        key = os.environ.get(config.key_env)
    if key is None:
        return None
    return key.encode("utf-8")


def _digest(record_without_hash: dict[str, Any], *, mode: str, key: bytes | None) -> str:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"unsupported audit mode: {mode}")
    payload = _canonical_bytes(record_without_hash)
    if mode == "sha256":
        return hashlib.sha256(payload).hexdigest()
    if key is None:
        raise ValueError("hmac_sha256 audit mode requires a key")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def _last_record_hash(path: Path) -> str:
    if not path.exists() or path.stat().st_size == 0:
        return GENESIS_HASH

    last_line = ""
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last_line = line
    if not last_line:
        return GENESIS_HASH

    try:
        record = json.loads(last_line)
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: cannot continue audit chain after invalid JSON: {e}") from e

    record_hash = record.get("record_hash")
    if isinstance(record_hash, str) and len(record_hash) == 64:
        return record_hash
    return GENESIS_HASH


class AuditSigner:
    """Adds audit metadata and a chained digest to alert JSON records."""

    def __init__(self, config: AuditConfig, *, prev_hash: str = GENESIS_HASH):
        if not config.enabled:
            raise ValueError("AuditSigner requires enabled audit config")
        if config.mode not in SUPPORTED_MODES:
            raise ValueError(f"unsupported audit mode: {config.mode}")
        self.config = config
        self.prev_hash = prev_hash
        self._key = _resolve_key(config)
        if config.mode == "hmac_sha256" and self._key is None:
            raise ValueError(f"audit mode hmac_sha256 requires key_env {config.key_env!r}")

    @classmethod
    def for_path(cls, path: str | Path, config: AuditConfig) -> AuditSigner:
        return cls(config, prev_hash=_last_record_hash(Path(path)))

    def sign(self, record: dict[str, Any]) -> dict[str, Any]:
        signed = dict(record)
        signed.update(
            schema_version=AUDIT_SCHEMA_VERSION,
            prev_hash=self.prev_hash,
            audit_mode=self.config.mode,
        )
        if self.config.key_id:
            signed["key_id"] = self.config.key_id

        record_hash = _digest(signed, mode=self.config.mode, key=self._key)
        signed["record_hash"] = record_hash
        self.prev_hash = record_hash
        return signed


def verify_jsonl(
    path: str | Path,
    *,
    mode: str | None = None,
    key: str | None = None,
    key_env: str | None = None,
    allow_unsigned: bool = False,
) -> VerificationResult:
    """Verify a signed JSONL alert log.

    The chain starts at ``GENESIS_HASH``. Each signed record must point at the
    previous signed record's hash, so middle deletion and reordering are caught.
    """

    path = Path(path)
    errors: list[str] = []
    checked = 0
    unsigned = 0
    prev_hash = GENESIS_HASH

    key_bytes = None
    if key is not None:
        key_bytes = key.encode("utf-8")
    elif key_env:
        env_key = os.environ.get(key_env)
        if env_key is not None:
            key_bytes = env_key.encode("utf-8")

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as e:
        return VerificationResult(False, errors=[f"{path}: {e}"])

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {line_no}: invalid JSON: {e}")
            continue

        record_hash = record.get("record_hash")
        if not isinstance(record_hash, str):
            if allow_unsigned:
                unsigned += 1
                continue
            errors.append(f"line {line_no}: missing record_hash")
            continue

        record_mode = record.get("audit_mode", mode or "sha256")
        if mode is not None and record_mode != mode:
            errors.append(f"line {line_no}: audit_mode {record_mode!r} != expected {mode!r}")
            continue
        if record_mode not in SUPPORTED_MODES:
            errors.append(f"line {line_no}: unsupported audit_mode {record_mode!r}")
            continue
        if record.get("prev_hash") != prev_hash:
            errors.append(f"line {line_no}: prev_hash does not match previous record")
            continue
        if record_mode == "hmac_sha256" and key_bytes is None:
            errors.append("hmac_sha256 verification requires --key or --key-env")
            continue

        unsigned_hash = dict(record)
        unsigned_hash.pop("record_hash", None)
        expected_hash = _digest(unsigned_hash, mode=record_mode, key=key_bytes)
        if not hmac.compare_digest(record_hash, expected_hash):
            errors.append(f"line {line_no}: record_hash mismatch")
            continue

        checked += 1
        prev_hash = record_hash

    return VerificationResult(
        valid=not errors,
        checked_records=checked,
        unsigned_records=unsigned,
        errors=errors,
    )
