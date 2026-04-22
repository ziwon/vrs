"""Annotated mp4 writer.

Overlays we draw:
  * status banner per active alert (TRUE / FALSE / FN)
  * YOLOE detection boxes (white) on every frame the detector hit
  * Cosmos-returned bbox (red) and trajectory polyline (yellow) for the duration
    of an alert hold window — gives a strong visual audit trail

When a ``FaceDetector`` is supplied, faces are Gaussian-blurred on the
frame copy *before* any overlay is drawn. The detector and verifier
upstream still see unblurred pixels — the verifier needs faces visible
to reason about events like "a person has collapsed" — but everything
written to disk is blurred. This matches GDPR / K-GDPR expectations for
CCTV retention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..privacy import FaceDetector, NullFaceDetector, blur_faces
from ..schemas import Detection, Frame, VerifiedAlert


_SEVERITY_COLOR = {
    "info": (200, 200, 200),
    "low": (200, 200, 0),
    "medium": (0, 215, 255),
    "high": (0, 165, 255),
    "critical": (0, 0, 255),
}


@dataclass
class _ActiveBanner:
    text: str
    color: Tuple[int, int, int]
    expires_at_pts_s: float
    bbox_xywh_norm: Optional[Tuple[float, float, float, float]] = None
    trajectory_xy_norm: List[Tuple[float, float]] = field(default_factory=list)


class VideoAnnotator:
    def __init__(
        self,
        path: str | Path,
        fps: float,
        banner_hold_s: float = 4.0,
        *,
        face_detector: Optional[FaceDetector] = None,
        blur_kernel: int = 31,
        blur_margin_pct: float = 0.15,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.banner_hold_s = float(banner_hold_s)
        self.face_detector: FaceDetector = face_detector or NullFaceDetector()
        self.blur_kernel = int(blur_kernel)
        self.blur_margin_pct = float(blur_margin_pct)

        self._writer: Optional[cv2.VideoWriter] = None
        self._size: Optional[Tuple[int, int]] = None
        self._active: Dict[str, _ActiveBanner] = {}    # keyed by class name

    # ---- lifecycle --------------------------------------------------

    def _lazy_open(self, shape) -> None:
        if self._writer is not None:
            return
        h, w = shape[:2]
        self._size = (w, h)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open video writer at {self.path}")

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    # ---- public API -------------------------------------------------

    def note_alert(self, alert: VerifiedAlert) -> None:
        if alert.true_alert:
            color = _SEVERITY_COLOR.get(alert.candidate.severity, (0, 0, 255))
            status = "TRUE ALERT"
        else:
            color = (0, 165, 255)
            status = "FALSE ALARM"
        text = (
            f"[{status}] {alert.candidate.class_name.upper()}  "
            f"sev={alert.candidate.severity}  conf={alert.confidence:.2f}"
        )
        if alert.false_negative_class:
            text += f"   (missed: {alert.false_negative_class})"
        self._active[alert.candidate.class_name] = _ActiveBanner(
            text=text,
            color=color,
            expires_at_pts_s=alert.candidate.peak_pts_s + self.banner_hold_s,
            bbox_xywh_norm=alert.bbox_xywh_norm,
            trajectory_xy_norm=list(alert.trajectory_xy_norm),
        )

    def write(self, frame: Frame, detections: Optional[List[Detection]] = None) -> None:
        img = frame.image.copy()
        self._lazy_open(img.shape)

        # Privacy pass runs on raw pixels *before* any overlay is drawn —
        # otherwise the banners/timestamps would get blurred too. A
        # NullFaceDetector returns [] and blur_faces short-circuits, so
        # the cost of a privacy-disabled run is one Python function call.
        faces = self.face_detector(img)
        if faces:
            blur_faces(img, faces, kernel=self.blur_kernel,
                       margin_pct=self.blur_margin_pct)

        # detector boxes (subtle white) — gives operator instant feedback
        if detections:
            for d in detections:
                x1, y1, x2, y2 = (int(v) for v in d.xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
                cv2.putText(
                    img, f"{d.class_name} {d.score:.2f}",
                    (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA,
                )

        # active banners: text + Cosmos bbox + Cosmos trajectory
        h, w = img.shape[:2]
        for cls in list(self._active.keys()):
            if frame.pts_s > self._active[cls].expires_at_pts_s:
                del self._active[cls]
        y_text = 30
        for banner in self._active.values():
            self._draw_banner(img, banner.text, banner.color, y_text)
            y_text += 40
            if banner.bbox_xywh_norm is not None:
                bx, by, bw, bh = banner.bbox_xywh_norm
                x1, y1 = int(bx * w), int(by * h)
                x2, y2 = int((bx + bw) * w), int((by + bh) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), banner.color, 2)
            if len(banner.trajectory_xy_norm) >= 2:
                pts = np.array(
                    [(int(x * w), int(y * h)) for x, y in banner.trajectory_xy_norm],
                    dtype=np.int32,
                )
                cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        cv2.putText(
            img, f"t={frame.pts_s:.2f}s  f#{frame.index}",
            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
        self._writer.write(img)  # type: ignore[union-attr]

    # ---- helpers ----------------------------------------------------

    @staticmethod
    def _draw_banner(img: np.ndarray, text: str, color, y: int) -> None:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = 10
        cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 8, y + 8), color, 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
