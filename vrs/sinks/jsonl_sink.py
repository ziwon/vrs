"""Append-only JSONL writer for VerifiedAlert records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import IO

from ..audit import AuditConfig, AuditSigner
from ..schemas import VerifiedAlert


class JsonlSink:
    def __init__(self, path: str | Path, audit: dict | AuditConfig | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp: IO[str] | None = None
        self.audit_config = (
            audit if isinstance(audit, AuditConfig) else AuditConfig.from_mapping(audit)
        )
        self._audit_signer: AuditSigner | None = None

    def __enter__(self) -> JsonlSink:
        if self.audit_config.enabled:
            self._audit_signer = AuditSigner.for_path(self.path, self.audit_config)
        self._fp = self.path.open("a", encoding="utf-8")
        return self

    def write(self, alert: VerifiedAlert) -> None:
        assert self._fp is not None, "JsonlSink used outside of a with-block"
        record = alert.to_json()
        if self._audit_signer is not None:
            record = self._audit_signer.sign(record)
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()

    def __exit__(self, *exc) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
