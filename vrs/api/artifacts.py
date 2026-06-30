"""Filesystem-backed access to VRS run artifacts.

This module intentionally stays stdlib-only so importing the API layer does not
load GPU/model dependencies.
"""

from __future__ import annotations

import json
import re
from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
TAIL_CURSOR_RE = re.compile(r"^[A-Za-z0-9_-]*={0,2}$")


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
    next_cursor: str | None = None


class UnsafePathError(ValueError):
    """Raised when a requested artifact path escapes the configured runs root."""


class RunArtifactStore:
    def __init__(self, runs_root: Path | str = "runs") -> None:
        self.runs_root = Path(runs_root).resolve()
        self._jsonl_cache = JsonlCache()

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
            alert_count=sum(self.count_jsonl_records(file.path) for file in files),
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
            records, file_errors = self.read_jsonl_records(alert_file.path)
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
                enriched = self._add_record_metadata(enriched, alert_file, line_no)
                enriched["_line"] = line_no
                enriched["_alert_id"] = _alert_id(run_name, alert_file.stream_id, line_no)
                thumb = record.get("thumbnail_path")
                if isinstance(thumb, str) and thumb:
                    enriched["thumbnail_url"] = self.thumbnail_url(
                        run_name, thumb, stream_id=alert_file.stream_id
                    )
                alerts.append(enriched)
        alerts.sort(
            key=lambda item: (str(item.get("stream_id") or ""), int(item.get("_line") or 0))
        )
        total = len(alerts)
        window = alerts[max(offset, 0) :]
        if limit >= 0:
            window = window[:limit]
        return AlertReadResult(alerts=window, errors=errors, total=total)

    def tail_alerts(
        self,
        run_name: str,
        *,
        cursor: str | None = None,
        since_line: int | None = None,
        stream_id: str | None = None,
        limit: int = 100,
        mode: str = "poll",
    ) -> AlertReadResult:
        positions = decode_tail_cursor(cursor)
        alerts: list[dict[str, Any]] = []
        errors: list[JsonlError] = []
        file_max_lines: dict[str, int] = {}
        for alert_file in self.alert_files(run_name):
            if stream_id is not None and alert_file.stream_id != stream_id:
                continue
            key = _cursor_key(alert_file)
            floor = positions.get(key, since_line or 0)
            records, file_errors = self.read_jsonl_records(alert_file.path)
            errors.extend(file_errors)
            file_max_lines[key] = max((line_no for line_no, _ in records), default=floor)
            for line_no, record in records:
                if line_no <= floor:
                    continue
                effective_stream = record.get("stream_id") or alert_file.stream_id
                if stream_id is not None and effective_stream != stream_id:
                    continue
                enriched = dict(record)
                if effective_stream is not None:
                    enriched["stream_id"] = effective_stream
                enriched = self._add_record_metadata(enriched, alert_file, line_no)
                enriched["_line"] = line_no
                enriched["_alert_id"] = _alert_id(run_name, alert_file.stream_id, line_no)
                thumb = record.get("thumbnail_path")
                if isinstance(thumb, str) and thumb:
                    enriched["thumbnail_url"] = self.thumbnail_url(
                        run_name, thumb, stream_id=alert_file.stream_id
                    )
                alerts.append(enriched)

        if mode == "latest":
            alerts.sort(key=_alert_sort_key, reverse=True)
        else:
            alerts.sort(
                key=lambda item: (str(item.get("stream_id") or ""), int(item.get("_line") or 0))
            )
        total = len(alerts)
        window = alerts[:limit] if limit >= 0 else alerts
        next_positions = dict(positions)
        if mode == "latest":
            for key, line_no in file_max_lines.items():
                next_positions[key] = max(next_positions.get(key, 0), line_no)
        else:
            for alert in window:
                key = str(alert.get("_cursor_key") or "")
                line_no = int(alert.get("_line") or 0)
                if key:
                    next_positions[key] = max(next_positions.get(key, 0), line_no)
        return AlertReadResult(
            alerts=window,
            errors=errors,
            total=total,
            next_cursor=encode_tail_cursor(next_positions),
        )

    def read_jsonl_records(
        self, path: Path
    ) -> tuple[list[tuple[int, dict[str, Any]]], list[JsonlError]]:
        return self._jsonl_cache.read(path)

    def count_jsonl_records(self, path: Path) -> int:
        records, _ = self.read_jsonl_records(path)
        return len(records)

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

    @staticmethod
    def _add_record_metadata(
        record: dict[str, Any], alert_file: AlertFile, line_no: int
    ) -> dict[str, Any]:
        enriched = dict(record)
        if "ts" not in enriched:
            for key in ("created_at", "written_at"):
                if isinstance(enriched.get(key), str):
                    enriched["ts"] = enriched[key]
                    break
        enriched.setdefault("written_at", _format_timestamp(alert_file.path.stat().st_mtime))
        if "latency_ms" not in enriched:
            latency = _latency_ms(enriched)
            if latency is not None:
                enriched["latency_ms"] = latency
        enriched["_cursor_key"] = _cursor_key(alert_file)
        return enriched


@dataclass(frozen=True)
class JsonlCacheEntry:
    size: int
    mtime_ns: int
    records: list[tuple[int, dict[str, Any]]]
    errors: list[JsonlError]


class JsonlCache:
    def __init__(self) -> None:
        self._entries: dict[Path, JsonlCacheEntry] = {}

    def read(self, path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[JsonlError]]:
        stat = path.stat()
        cached = self._entries.get(path)
        if (
            cached is not None
            and cached.size == stat.st_size
            and cached.mtime_ns == stat.st_mtime_ns
        ):
            return cached.records, cached.errors
        records, errors = _parse_jsonl_records(path)
        self._entries[path] = JsonlCacheEntry(
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
            records=records,
            errors=errors,
        )
        return records, errors


def iter_jsonl_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records, _ = read_jsonl_records(path)
    return records


def read_jsonl_records(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[JsonlError]]:
    return _parse_jsonl_records(path)


def _parse_jsonl_records(
    path: Path,
) -> tuple[list[tuple[int, dict[str, Any]]], list[JsonlError]]:
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
                errors.append(
                    JsonlError(path=str(path), line=line_no, error="expected JSON object")
                )
                continue
            records.append((line_no, value))
    return records, errors


def count_jsonl_records(path: Path) -> int:
    return len(iter_jsonl_records(path))


def encode_tail_cursor(positions: dict[str, int]) -> str:
    payload = {key: int(value) for key, value in sorted(positions.items()) if int(value) > 0}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_tail_cursor(cursor: str | None) -> dict[str, int]:
    if not cursor:
        return {}
    if not TAIL_CURSOR_RE.fullmatch(cursor):
        raise UnsafePathError("invalid tail cursor")
    padded = cursor + "=" * (-len(cursor) % 4)
    try:
        decoded = urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        value = json.loads(decoded)
    except (BinasciiError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise UnsafePathError("invalid tail cursor") from exc
    if not isinstance(value, dict):
        raise UnsafePathError("invalid tail cursor")
    positions: dict[str, int] = {}
    for key, line_no in value.items():
        if not isinstance(key, str) or not _safe_name(key):
            raise UnsafePathError("invalid tail cursor")
        if not isinstance(line_no, int) or isinstance(line_no, bool) or line_no < 0:
            raise UnsafePathError("invalid tail cursor")
        positions[key] = line_no
    return positions


def _cursor_key(alert_file: AlertFile) -> str:
    return alert_file.stream_id or "__root__"


def _format_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, UTC).isoformat().replace("+00:00", "Z")


def _latency_ms(record: dict[str, Any]) -> int | None:
    written = _parse_timestamp(record.get("written_at") or record.get("created_at"))
    event = _parse_timestamp(record.get("ts"))
    if written is None or event is None:
        return None
    return max(0, round((written - event).total_seconds() * 1000))


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _alert_sort_key(alert: dict[str, Any]) -> tuple[float, float, str, int]:
    timestamp = _parse_timestamp(alert.get("ts"))
    timestamp_value = timestamp.timestamp() if timestamp is not None else 0.0
    pts = _as_float(alert.get("peak_pts_s"), default=0.0)
    return (
        timestamp_value,
        pts,
        str(alert.get("stream_id") or ""),
        int(alert.get("_line") or 0),
    )


def _as_float(value: object, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_name(value: str) -> bool:
    return bool(value) and "/" not in value and "\\" not in value and value not in {".", ".."}


def _alert_id(run_name: str, stream_id: str | None, line_no: int) -> str:
    parts = [run_name]
    if stream_id is not None:
        parts.append(stream_id)
    parts.append(str(line_no))
    return ":".join(parts)
