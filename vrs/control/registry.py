"""Stream registry and health primitives for the control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from .static_assignment import StreamInput


class HealthStatus(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class StreamRegistryEntry:
    stream: StreamInput
    enabled: bool = True
    assigned_worker_id: str | None = None
    created_at: str = field(default_factory=lambda: _now())
    updated_at: str = field(default_factory=lambda: _now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream": self.stream.to_contract(),
            "enabled": self.enabled,
            "assigned_worker_id": self.assigned_worker_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class StreamRegistry:
    """In-memory stream registry for static control-plane rendering."""

    def __init__(self) -> None:
        self._entries: dict[str, StreamRegistryEntry] = {}

    def upsert(
        self,
        stream: StreamInput,
        *,
        enabled: bool = True,
        assigned_worker_id: str | None = None,
    ) -> StreamRegistryEntry:
        existing = self._entries.get(stream.stream_id)
        entry = StreamRegistryEntry(
            stream=stream,
            enabled=enabled,
            assigned_worker_id=assigned_worker_id,
            created_at=existing.created_at if existing else _now(),
            updated_at=_now(),
        )
        self._entries[stream.stream_id] = entry
        return entry

    def disable(self, stream_id: str) -> StreamRegistryEntry:
        entry = self.get(stream_id)
        return self.upsert(
            entry.stream,
            enabled=False,
            assigned_worker_id=entry.assigned_worker_id,
        )

    def get(self, stream_id: str) -> StreamRegistryEntry:
        try:
            return self._entries[stream_id]
        except KeyError as exc:
            raise KeyError(f"unknown stream: {stream_id}") from exc

    def list(self, *, enabled_only: bool = False) -> list[StreamRegistryEntry]:
        entries = sorted(self._entries.values(), key=lambda entry: entry.stream.stream_id)
        if enabled_only:
            entries = [entry for entry in entries if entry.enabled]
        return entries


@dataclass(frozen=True)
class QueuePressure:
    queue: str
    stream_id: str
    size: int
    dropped: int
    status: HealthStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "stream_id": self.stream_id,
            "size": self.size,
            "dropped": self.dropped,
            "status": self.status.value,
        }


@dataclass(frozen=True)
class WorkerHealth:
    worker_id: str
    gpu_role: str
    status: HealthStatus
    checked_at: str
    queue_pressure: list[QueuePressure] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "gpu_role": self.gpu_role,
            "status": self.status.value,
            "checked_at": self.checked_at,
            "queue_pressure": [item.to_dict() for item in self.queue_pressure],
            "details": dict(self.details),
        }


def summarize_queue_pressure(
    queue_stats: dict[str, Any],
    *,
    degraded_drop_threshold: int = 1,
    unhealthy_drop_threshold: int = 100,
) -> list[QueuePressure]:
    """Convert existing MultiStreamPipeline.queue_stats() output to health rows."""

    out: list[QueuePressure] = []
    for queue_name, payload in sorted(queue_stats.items()):
        if not isinstance(payload, dict):
            continue
        if "by_stream" in payload and isinstance(payload["by_stream"], dict):
            for stream_id, stats in sorted(payload["by_stream"].items()):
                out.append(
                    _pressure_row(
                        queue=queue_name,
                        stream_id=str(stream_id),
                        stats=stats,
                        degraded_drop_threshold=degraded_drop_threshold,
                        unhealthy_drop_threshold=unhealthy_drop_threshold,
                    )
                )
        else:
            out.append(
                _pressure_row(
                    queue=queue_name,
                    stream_id="all",
                    stats=payload,
                    degraded_drop_threshold=degraded_drop_threshold,
                    unhealthy_drop_threshold=unhealthy_drop_threshold,
                )
            )
    return out


def worker_health_from_queue_stats(
    *,
    worker_id: str,
    gpu_role: str,
    queue_stats: dict[str, Any],
    details: dict[str, Any] | None = None,
) -> WorkerHealth:
    pressure = summarize_queue_pressure(queue_stats)
    statuses = [item.status for item in pressure]
    if any(status == HealthStatus.UNHEALTHY for status in statuses):
        status = HealthStatus.UNHEALTHY
    elif any(status == HealthStatus.DEGRADED for status in statuses):
        status = HealthStatus.DEGRADED
    elif statuses:
        status = HealthStatus.HEALTHY
    else:
        status = HealthStatus.UNKNOWN
    return WorkerHealth(
        worker_id=worker_id,
        gpu_role=gpu_role,
        status=status,
        checked_at=_now(),
        queue_pressure=pressure,
        details=dict(details or {}),
    )


def _pressure_row(
    *,
    queue: str,
    stream_id: str,
    stats: Any,
    degraded_drop_threshold: int,
    unhealthy_drop_threshold: int,
) -> QueuePressure:
    stats = stats if isinstance(stats, dict) else {}
    size = int(stats.get("size", 0))
    dropped = int(stats.get("dropped", 0))
    if dropped >= unhealthy_drop_threshold:
        status = HealthStatus.UNHEALTHY
    elif dropped >= degraded_drop_threshold:
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY
    return QueuePressure(
        queue=queue,
        stream_id=stream_id,
        size=size,
        dropped=dropped,
        status=status,
    )


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
