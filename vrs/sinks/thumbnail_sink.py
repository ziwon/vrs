"""Event thumbnail writer.

Production runs should retain compact event evidence by default: one image per
verified alert, linked from ``alerts.jsonl``. Full annotated MP4s are still
useful for demos/debugging, but they are expensive to store, harder to review at
scale, and carry more privacy risk.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ..privacy import FaceDetector, NullFaceDetector, blur_faces
from ..schemas import Detection, VerifiedAlert


_SEVERITY_COLOR = {
    "info": (200, 200, 200),
    "low": (200, 200, 0),
    "medium": (0, 215, 255),
    "high": (0, 165, 255),
    "critical": (0, 0, 255),
}


def _safe_name(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return out or "event"


class EventThumbnailSink:
    """Writes one annotated image per alert and stores its relative path."""

    def __init__(
        self,
        root_dir: str | Path,
        dir_name: str = "thumbnails",
        *,
        ext: str = "jpg",
        quality: int = 90,
        face_detector: Optional[FaceDetector] = None,
        blur_kernel: int = 31,
        blur_margin_pct: float = 0.15,
    ):
        self.root_dir = Path(root_dir)
        self.dir_name = dir_name.strip("/\\") or "thumbnails"
        self.path = self.root_dir / self.dir_name
        self.path.mkdir(parents=True, exist_ok=True)
        self.ext = ext.lower().lstrip(".") or "jpg"
        if self.ext == "jpeg":
            self.ext = "jpg"
        self.quality = max(1, min(100, int(quality)))
        self.face_detector: FaceDetector = face_detector or NullFaceDetector()
        self.blur_kernel = int(blur_kernel)
        self.blur_margin_pct = float(blur_margin_pct)

    def write(self, alert: VerifiedAlert) -> Optional[str]:
        frame = self._pick_keyframe(alert)
        if frame is None:
            return None

        img = frame.copy()
        faces = self.face_detector(img)
        if faces:
            blur_faces(
                img,
                faces,
                kernel=self.blur_kernel,
                margin_pct=self.blur_margin_pct,
            )
        self._draw_overlays(img, alert)

        rel = self._relative_name(alert)
        out_path = self.root_dir / rel
        params = self._encode_params()
        import cv2  # lazy: keep JsonlSink-only tests import-light
        ok = cv2.imwrite(str(out_path), img, params)
        if not ok:
            raise RuntimeError(f"failed to write event thumbnail: {out_path}")
        alert.thumbnail_path = rel.as_posix()
        return alert.thumbnail_path

    def _relative_name(self, alert: VerifiedAlert) -> Path:
        cls = _safe_name(alert.candidate.class_name)
        verdict = "true" if alert.true_alert else "false"
        track = "none" if alert.candidate.track_id is None else str(alert.candidate.track_id)
        pts_ms = int(round(alert.candidate.peak_pts_s * 1000.0))
        name = (
            f"{alert.candidate.peak_frame_index:08d}_"
            f"{pts_ms:010d}ms_{cls}_track-{track}_{verdict}.{self.ext}"
        )
        return Path(self.dir_name) / name

    def _encode_params(self) -> List[int]:
        import cv2  # lazy
        if self.ext == "jpg":
            return [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        if self.ext == "png":
            return [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
        return []

    @staticmethod
    def _pick_keyframe(alert: VerifiedAlert) -> Optional[np.ndarray]:
        frames = alert.candidate.keyframes
        if not frames:
            return None
        pts = alert.candidate.keyframe_pts
        if not pts or len(pts) != len(frames):
            return frames[-1]
        peak = alert.candidate.peak_pts_s
        idx = min(range(len(frames)), key=lambda i: abs(float(pts[i]) - peak))
        return frames[idx]

    def _draw_overlays(self, img: np.ndarray, alert: VerifiedAlert) -> None:
        import cv2  # lazy
        color = _SEVERITY_COLOR.get(alert.candidate.severity, (0, 0, 255))
        if not alert.true_alert:
            color = (0, 165, 255)
        for det in alert.candidate.peak_detections:
            self._draw_detection(img, det)

        h, w = img.shape[:2]
        if alert.bbox_xywh_norm is not None:
            bx, by, bw, bh = alert.bbox_xywh_norm
            x1, y1 = int(bx * w), int(by * h)
            x2, y2 = int((bx + bw) * w), int((by + bh) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if len(alert.trajectory_xy_norm) >= 2:
            pts = np.array(
                [(int(x * w), int(y * h)) for x, y in alert.trajectory_xy_norm],
                dtype=np.int32,
            )
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)

        status = "TRUE" if alert.true_alert else "FALSE"
        label = (
            f"{status} {alert.candidate.class_name} "
            f"sev={alert.candidate.severity} conf={alert.confidence:.2f}"
        )
        self._draw_banner(img, label, color)

    @staticmethod
    def _draw_detection(img: np.ndarray, det: Detection) -> None:
        import cv2  # lazy
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
        cv2.putText(
            img,
            f"{det.class_name} {det.score:.2f}",
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_banner(img: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
        import cv2  # lazy
        x, y = 10, 30
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 8, y + 8), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 4, y - th - 8), (x + tw + 8, y + 8), color, 2)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
