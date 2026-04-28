"""Data contracts that flow through the cascade.

Frame ──► Detection (per object, YOLOE) ──► CandidateAlert (event-state) ──► VerifiedAlert (Cosmos)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Frame:
    """A single decoded frame from the ingest stage."""

    index: int  # frame index at the target FPS
    pts_s: float  # presentation timestamp in seconds (stream time)
    image: np.ndarray  # HxWx3 BGR uint8 (OpenCV native)


@dataclass
class Detection:
    """One object reported by the YOLOE fast path."""

    class_name: str  # name from the watch policy
    score: float
    xyxy: tuple[float, float, float, float]  # absolute pixels in the source frame
    raw_label: str = ""  # the YOLOE prompt that fired (for debugging)
    track_id: int | None = None  # assigned by the tracker; None if untracked


@dataclass
class CandidateAlert:
    """A class-level event raised once temporal persistence is met."""

    class_name: str
    severity: str
    start_pts_s: float
    peak_pts_s: float
    peak_frame_index: int
    peak_detections: list[Detection]  # detections on the peak frame
    keyframes: list[np.ndarray] = field(default_factory=list)  # BGR uint8
    keyframe_pts: list[float] = field(default_factory=list)
    track_id: int | None = None  # tracker id of the peak detection, or None if untracked

    def summary(self) -> dict[str, Any]:
        return {
            "class_name": self.class_name,
            "severity": self.severity,
            "start_pts_s": self.start_pts_s,
            "peak_pts_s": self.peak_pts_s,
            "peak_frame_index": self.peak_frame_index,
            "track_id": self.track_id,
            "peak_detections": [
                {
                    "score": float(d.score),
                    "xyxy": [float(x) for x in d.xyxy],
                    "raw_label": d.raw_label,
                    "track_id": d.track_id,
                }
                for d in self.peak_detections
            ],
            "num_keyframes": len(self.keyframes),
        }

    def detector_bbox_xywh_norm(self) -> tuple[float, float, float, float] | None:
        """Return the highest-score peak detector box as normalized x/y/w/h."""
        if not self.peak_detections or not self.keyframes:
            return None
        height, width = self.keyframes[-1].shape[:2]
        if width <= 0 or height <= 0:
            return None
        det = max(self.peak_detections, key=lambda item: item.score)
        x1, y1, x2, y2 = det.xyxy
        x1 = _clamp(float(x1) / width)
        y1 = _clamp(float(y1) / height)
        x2 = _clamp(float(x2) / width)
        y2 = _clamp(float(y2) / height)
        return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


@dataclass
class VerifiedAlert:
    """The final record written to the sink."""

    candidate: CandidateAlert
    true_alert: bool
    confidence: float  # 0.0 .. 1.0
    false_negative_class: str | None  # detector missed this listed event
    rationale: str  # one short sentence
    bbox_xywh_norm: tuple[float, float, float, float] | None = None  # Cosmos bbox
    trajectory_xy_norm: list[tuple[float, float]] = field(default_factory=list)
    verifier_raw: str = ""
    thumbnail_path: str | None = None

    def to_json(self) -> dict[str, Any]:
        out = self.candidate.summary()
        bbox = self.bbox_xywh_norm or self.candidate.detector_bbox_xywh_norm()
        out.update(
            true_alert=bool(self.true_alert),
            confidence=float(self.confidence),
            false_negative_class=self.false_negative_class,
            rationale=self.rationale,
            bbox_xywh_norm=([float(x) for x in bbox] if bbox is not None else None),
            trajectory_xy_norm=[[float(x), float(y)] for (x, y) in self.trajectory_xy_norm],
            verifier_raw=self.verifier_raw,
            thumbnail_path=self.thumbnail_path,
        )
        return out


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
