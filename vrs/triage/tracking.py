"""Per-stream object tracking on top of the YOLOE fast path.

Why: without tracking, one persistent fire raises one alert per cooldown
window — an operator sees N duplicates for one physical occurrence. With a
``track_id``, cooldown is keyed on ``(class, track_id)`` so the same
physical object produces one alert; two *different* simultaneous fires
each get their own alert.

Two implementations are shipped:

* ``NullTracker`` — passes detections through unchanged (``track_id`` stays
  ``None``). This preserves the pre-tracking behavior when tracking is
  disabled, without any code-path branching elsewhere in the pipeline.
* ``SimpleIoUTracker`` — greedy IoU association with configurable gap
  tolerance. Pure Python, no new deps, fully unit-testable. Handles the
  common case well; a ByteTrack / OC-SORT backend can plug into the same
  ``Tracker`` protocol when the ultralytics tracker API is imported
  lazily (that backend will live alongside this one — not a replacement).

Tracker state is *per-stream*: the multi-stream pipeline constructs one
tracker per ``stream_id`` so track IDs don't collide across cameras.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..schemas import Detection


class Tracker(Protocol):
    """Mutates ``track_id`` on the detections it is handed and returns them.

    Implementations should be stateful — the same instance sees every frame
    of one stream — and must cope with an empty detection list.
    """

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Detection]: ...


# ──────────────────────────────────────────────────────────────────────
# pass-through
# ──────────────────────────────────────────────────────────────────────


class NullTracker:
    """No-op tracker — keeps ``track_id=None`` on every detection."""

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Detection]:
        return list(detections)


# ──────────────────────────────────────────────────────────────────────
# simple IoU tracker — greedy, per-class association
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Track:
    track_id: int
    class_name: str
    last_xyxy: tuple[float, float, float, float]
    last_seen_idx: int


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


class SimpleIoUTracker:
    """Greedy IoU-based tracker.

    Each frame: for every class present in detections, walk detections in
    descending score and try to match each to the best unmatched track of
    the same class (IoU ≥ ``iou_threshold``). Unmatched detections spawn
    new tracks. Tracks unseen for more than ``max_gap_frames`` frames are
    retired so their IDs don't keep getting matched against stale boxes.

    Class-segregated matching avoids a jittery smoke box stealing a fire's
    track id just because their bboxes happen to overlap.
    """

    def __init__(self, iou_threshold: float = 0.3, max_gap_frames: int = 5):
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_gap_frames < 1:
            raise ValueError("max_gap_frames must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.max_gap_frames = int(max_gap_frames)
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    def update(self, detections: Sequence[Detection], frame_index: int) -> list[Detection]:
        out: list[Detection] = []
        # group by class
        by_cls: dict[str, list[Detection]] = {}
        for d in detections:
            by_cls.setdefault(d.class_name, []).append(d)

        for cls, cls_dets in by_cls.items():
            # Match highest-score detections first — they have the best chance
            # of being the real object, so they deserve first crack at the
            # existing tracks.
            cls_dets_sorted = sorted(cls_dets, key=lambda d: -d.score)
            active_tracks = [t for t in self._tracks.values() if t.class_name == cls]
            matched_track_ids: set[int] = set()

            for det in cls_dets_sorted:
                best_iou = 0.0
                best_tid: int | None = None
                for t in active_tracks:
                    if t.track_id in matched_track_ids:
                        continue
                    iou = _iou(det.xyxy, t.last_xyxy)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_tid = t.track_id
                if best_tid is not None:
                    matched_track_ids.add(best_tid)
                    track = self._tracks[best_tid]
                    track.last_xyxy = det.xyxy
                    track.last_seen_idx = frame_index
                    det.track_id = best_tid
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = _Track(
                        track_id=tid,
                        class_name=cls,
                        last_xyxy=det.xyxy,
                        last_seen_idx=frame_index,
                    )
                    det.track_id = tid
                out.append(det)

        # retire stale tracks so their IDs don't drift back onto unrelated
        # objects that happen to overlap a decayed bbox
        expired = [
            tid
            for tid, t in self._tracks.items()
            if frame_index - t.last_seen_idx > self.max_gap_frames
        ]
        for tid in expired:
            del self._tracks[tid]

        return out


# ──────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────


def build_tracker(cfg: dict | None) -> Tracker:
    """Construct a tracker from a YAML config block.

    Accepted shape::

        tracker:
          backend: simple_iou       # simple_iou | none (default: none)
          iou_threshold: 0.3
          max_gap_frames: 5

    A missing block, ``cfg is None``, or ``backend: none`` yields a
    ``NullTracker`` — behavior stays identical to pre-tracking builds.
    """
    if not cfg:
        return NullTracker()
    backend = str(cfg.get("backend", "none")).lower()
    if backend in ("none", "null", ""):
        return NullTracker()
    if backend in ("simple_iou", "iou", "simple"):
        return SimpleIoUTracker(
            iou_threshold=float(cfg.get("iou_threshold", 0.3)),
            max_gap_frames=int(cfg.get("max_gap_frames", 5)),
        )
    raise ValueError(f"unknown tracker backend: {backend!r}")
