"""Dependency-free DeepStream metadata adapter.

DeepStream Python bindings are not imported here. Production exporters can map
NvDsObjectMeta/NvDsFrameMeta fields into this dataclass and then publish the
resulting detection.v1 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..contracts import detection_v1
from ..schemas import Detection


@dataclass(frozen=True)
class DeepStreamDetectionMetadata:
    stream_id: str
    frame_index: int
    pts_s: float
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    clip_id: str | None = None
    raw_label: str = ""
    track_id: int | None = None
    detector_id: str | None = None
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> DeepStreamDetectionMetadata:
        bbox = data.get("bbox_xyxy")
        if bbox is None:
            bbox = data.get("xyxy")
        if bbox is None and {"left", "top", "width", "height"}.issubset(data):
            left = float(data["left"])
            top = float(data["top"])
            bbox = (
                left,
                top,
                left + float(data["width"]),
                top + float(data["height"]),
            )
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("DeepStream metadata needs bbox_xyxy or left/top/width/height")
        evidence_refs = data.get("evidence_refs") or []
        if not isinstance(evidence_refs, list):
            raise ValueError("evidence_refs must be a list")
        return cls(
            stream_id=str(data["stream_id"]),
            clip_id=str(data["clip_id"]) if data.get("clip_id") is not None else None,
            frame_index=int(data["frame_index"]),
            pts_s=float(data.get("pts_s", 0.0)),
            class_name=str(data["class_name"]),
            confidence=float(data.get("confidence", data.get("score"))),
            bbox_xyxy=tuple(float(x) for x in bbox),
            raw_label=str(data.get("raw_label", "")),
            track_id=int(data["track_id"]) if data.get("track_id") is not None else None,
            detector_id=str(data["detector_id"]) if data.get("detector_id") is not None else None,
            evidence_refs=evidence_refs,
        )


def detection_from_deepstream(meta: DeepStreamDetectionMetadata) -> dict[str, Any]:
    """Convert exported DeepStream object metadata to ``detection.v1``."""

    det = Detection(
        class_name=meta.class_name,
        score=float(meta.confidence),
        xyxy=tuple(float(x) for x in meta.bbox_xyxy),
        raw_label=meta.raw_label,
        track_id=meta.track_id,
    )
    return detection_v1(
        det,
        stream_id=meta.stream_id,
        clip_id=meta.clip_id,
        frame_index=meta.frame_index,
        pts_s=meta.pts_s,
        detector_id=meta.detector_id,
        source_runtime="deepstream",
        evidence_refs=meta.evidence_refs,
    )
