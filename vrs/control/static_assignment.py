"""Static stream assignment primitives.

This is intentionally small: it renders deterministic worker inputs before a
future scheduler/operator exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import stream_v1


@dataclass(frozen=True)
class StreamInput:
    stream_id: str
    source_uri: str
    name: str | None = None
    roi_polygon: list[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> StreamInput:
        source = (
            data.get("source_uri") or data.get("source") or data.get("rtsp") or data.get("video")
        )
        if not data.get("id") and not data.get("stream_id"):
            raise ValueError("stream entry must define id or stream_id")
        if not source:
            raise ValueError("stream entry must define source_uri, source, rtsp, or video")
        return cls(
            stream_id=str(data.get("stream_id") or data["id"]),
            source_uri=str(source),
            name=str(data["name"]) if data.get("name") is not None else None,
            roi_polygon=data.get("roi_polygon"),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_contract(self) -> dict[str, Any]:
        return stream_v1(
            stream_id=self.stream_id,
            source_uri=self.source_uri,
            name=self.name,
            roi_polygon=self.roi_polygon,
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class StreamAssignment:
    worker_id: str
    stream: StreamInput


@dataclass(frozen=True)
class WorkerConfig:
    worker_id: str
    gpu_role: str
    streams: list[dict[str, Any]]
    transport: dict[str, Any]
    object_store: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "gpu_role": self.gpu_role,
            "streams": self.streams,
            "transport": self.transport,
            "object_store": self.object_store,
        }


def assign_streams_static(
    streams: list[StreamInput],
    *,
    worker_count: int,
    worker_prefix: str = "deepstream",
    max_streams_per_worker: int | None = None,
) -> list[StreamAssignment]:
    if worker_count <= 0:
        raise ValueError("worker_count must be > 0")
    if max_streams_per_worker is not None and max_streams_per_worker <= 0:
        raise ValueError("max_streams_per_worker must be > 0")
    capacity = worker_count * max_streams_per_worker if max_streams_per_worker else None
    if capacity is not None and len(streams) > capacity:
        raise ValueError(
            f"{len(streams)} streams exceed static capacity {capacity} "
            f"({worker_count} workers x {max_streams_per_worker})"
        )

    assignments = []
    for idx, stream in enumerate(streams):
        worker_idx = idx % worker_count
        assignments.append(
            StreamAssignment(
                worker_id=f"{worker_prefix}-{worker_idx}",
                stream=stream,
            )
        )
    return assignments


def render_worker_configs(
    assignments: list[StreamAssignment],
    *,
    gpu_role: str = "deepstream",
    transport: dict[str, Any] | None = None,
    object_store: dict[str, Any] | None = None,
) -> list[WorkerConfig]:
    by_worker: dict[str, list[dict[str, Any]]] = {}
    for assignment in assignments:
        by_worker.setdefault(assignment.worker_id, []).append(assignment.stream.to_contract())
    return [
        WorkerConfig(
            worker_id=worker_id,
            gpu_role=gpu_role,
            streams=streams,
            transport=dict(transport or {"type": "redis-streams"}),
            object_store=dict(object_store or {"type": "local-filesystem"}),
        )
        for worker_id, streams in sorted(by_worker.items())
    ]
