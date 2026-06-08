"""Filesystem-backed access to VRS run artifacts.

This module intentionally stays stdlib-only so importing the web layer does not
load GPU/model dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class AlertFile:
    path: Path
    stream_id: str | None


@dataclass(frozen=True)
class RunInfo:
    name: str
    layout: str
    streams: list[str]
    alert_count: int
    updated_at: float | None


@dataclass(frozen=True)
class JsonlError:
    path: str
    line: int
    error: str


@dataclass(frozen=True)
class AlertReadResult:
    alerts: list[dict[str, Any]]
    errors: list[JsonlError]
    total: int


class UnsafePathError(ValueError):
    """Raised when a requested artifact path escapes the configured runs root."""


class RunArtifactStore:
    def __init__(self, runs_root: Path | str = "runs") -> None:
        self.runs_root = Path(runs_root).resolve()

    def list_runs(self) -> list[RunInfo]:
        if not self.runs_root.exists():
            return []
        runs: list[RunInfo] = []
        for path in sorted(self.runs_root.iterdir(), key=lambda item: item.name):
            if path.is_dir() and _safe_name(path.name):
                runs.append(self.describe_run(path.name))
        return runs

    def describe_run(self, run_name: str) -> RunInfo:
        run_dir = self.run_dir(run_name)
        files = self.alert_files(run_name)
        streams = sorted({file.stream_id for file in files if file.stream_id is not None})
        updated_at = max((file.path.stat().st_mtime for file in files), default=None)
        if any(file.stream_id is None for file in files) and streams:
            layout = "mixed"
        elif streams:
            layout = "multi"
        elif (run_dir / "alerts.jsonl").exists():
            layout = "single"
        else:
            layout = "empty"
        return RunInfo(
            name=run_name,
            layout=layout,
            streams=streams,
            alert_count=sum(count_jsonl_records(file.path) for file in files),
            updated_at=updated_at,
        )

    def run_dir(self, run_name: str) -> Path:
        if not _safe_name(run_name):
            raise UnsafePathError("invalid run name")
        return self._resolve_inside(self.runs_root / run_name, must_exist=False)

    def alert_files(self, run_name: str) -> list[AlertFile]:
        run_dir = self.run_dir(run_name)
        if not run_dir.exists() or not run_dir.is_dir():
            return []
        files: list[AlertFile] = []
        direct = run_dir / "alerts.jsonl"
        if direct.is_file():
            files.append(AlertFile(path=direct, stream_id=None))
        for child in sorted(run_dir.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or not _safe_name(child.name):
                continue
            stream_alerts = child / "alerts.jsonl"
            if stream_alerts.is_file():
                files.append(AlertFile(path=stream_alerts, stream_id=child.name))
        return files

    def read_alerts(
        self,
        run_name: str,
        *,
        class_name: str | None = None,
        severity: str | None = None,
        true_alert: bool | None = None,
        stream_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        since_line: int | None = None,
    ) -> AlertReadResult:
        alerts: list[dict[str, Any]] = []
        errors: list[JsonlError] = []
        for alert_file in self.alert_files(run_name):
            if stream_id is not None and alert_file.stream_id != stream_id:
                continue
            records, file_errors = read_jsonl_records(alert_file.path)
            errors.extend(file_errors)
            for line_no, record in records:
                if since_line is not None and line_no <= since_line:
                    continue
                effective_stream = record.get("stream_id") or alert_file.stream_id
                if class_name is not None and record.get("class_name") != class_name:
                    continue
                if severity is not None and record.get("severity") != severity:
                    continue
                if true_alert is not None and record.get("true_alert") is not true_alert:
                    continue
                if stream_id is not None and effective_stream != stream_id:
                    continue
                enriched = dict(record)
                if effective_stream is not None:
                    enriched["stream_id"] = effective_stream
                enriched["_line"] = line_no
                enriched["_alert_id"] = _alert_id(run_name, alert_file.stream_id, line_no)
                thumb = record.get("thumbnail_path")
                if isinstance(thumb, str) and thumb:
                    enriched["thumbnail_url"] = self.thumbnail_url(
                        run_name, thumb, stream_id=alert_file.stream_id
                    )
                alerts.append(enriched)
        alerts.sort(key=lambda item: (str(item.get("stream_id") or ""), int(item.get("_line") or 0)))
        total = len(alerts)
        window = alerts[max(offset, 0) :]
        if limit >= 0:
            window = window[:limit]
        return AlertReadResult(alerts=window, errors=errors, total=total)

    def thumbnail_path(
        self, run_name: str, thumbnail_path: str, *, stream_id: str | None = None
    ) -> Path:
        base = self.run_dir(run_name)
        if stream_id is not None:
            if not _safe_name(stream_id):
                raise UnsafePathError("invalid stream id")
            base = self._resolve_inside(base / stream_id, must_exist=False)
        rel = Path(thumbnail_path)
        if rel.is_absolute() or ".." in rel.parts:
            raise UnsafePathError("invalid thumbnail path")
        resolved = self._resolve_inside(base / rel, must_exist=True)
        if resolved.suffix.lower() not in IMAGE_SUFFIXES:
            raise UnsafePathError("unsupported thumbnail type")
        return resolved

    def thumbnail_url(
        self, run_name: str, thumbnail_path: str, *, stream_id: str | None = None
    ) -> str:
        path = quote(thumbnail_path, safe="/")
        url = f"/api/runs/{quote(run_name)}/thumbnails/{path}"
        if stream_id is not None:
            url += f"?stream_id={quote(stream_id)}"
        return url

    def _resolve_inside(self, path: Path, *, must_exist: bool) -> Path:
        resolved = path.resolve(strict=must_exist)
        try:
            resolved.relative_to(self.runs_root)
        except ValueError as exc:
            raise UnsafePathError("path escapes runs root") from exc
        return resolved


def iter_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records, _ = read_jsonl_records(path)
    return records


def read_jsonl_records(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[JsonlError]]:
    records: list[tuple[int, dict[str, Any]]] = []
    errors: list[JsonlError] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(JsonlError(path=str(path), line=line_no, error=exc.msg))
                continue
            if not isinstance(value, dict):
                errors.append(JsonlError(path=str(path), line=line_no, error="expected JSON object"))
                continue
            records.append((line_no, value))
    return records, errors


def count_jsonl_records(path: Path) -> int:
    return len(iter_jsonl_records(path))


def _safe_name(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and value not in {".", ".."}


def _alert_id(run_name: str, stream_id: str | None, line_no: int) -> str:
    parts = [run_name]
    if stream_id is not None:
        parts.append(stream_id)
    parts.append(str(line_no))
    return ":".join(parts)
