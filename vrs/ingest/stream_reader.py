"""Frame iterator for RTSP streams and mp4 files.

Decoding via OpenCV/FFmpeg. Drops frames to hit ``target_fps`` (typical CCTV is
15-30 fps; we run the cascade at 4 fps to match Cosmos-Reason2-2B's training
distribution and to keep the GPU lightly loaded).

Optional ROI polygon mask; pixels outside are zeroed (frame size unchanged so
detector input shape is stable).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import cv2
import numpy as np

from ..schemas import Frame


class StreamReader:
    def __init__(
        self,
        source: str,
        target_fps: float = 4.0,
        roi_polygon: list[tuple[float, float]] | None = None,
        reconnect_s: float = 2.0,
    ):
        self.source = source
        self.target_fps = float(target_fps)
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        self.reconnect_s = reconnect_s

        self._cap: cv2.VideoCapture | None = None
        self._native_fps: float = 0.0
        self._is_live = self._looks_like_live(source)

    @staticmethod
    def _looks_like_live(s: str) -> bool:
        return s.startswith(("rtsp://", "rtmp://", "http://", "https://")) or s.isdigit()

    def _open(self) -> None:
        if self.source.isdigit():
            self._cap = cv2.VideoCapture(int(self.source))
        else:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        self._native_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 25.0)

    def _apply_roi(self, img: np.ndarray) -> np.ndarray:
        if self.roi_polygon is None:
            return img
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.roi_polygon], 255)
        return cv2.bitwise_and(img, img, mask=mask)

    def __iter__(self) -> Iterator[Frame]:
        self._open()
        assert self._cap is not None

        native_fps = max(self._native_fps, 1.0)
        step = max(1, round(native_fps / max(self.target_fps, 0.1)))

        src_idx = 0
        out_idx = 0
        t_start = time.monotonic()

        while True:
            ok, frame = self._cap.read()
            if not ok:
                if self._is_live:
                    self._cap.release()
                    time.sleep(self.reconnect_s)
                    try:
                        self._open()
                        continue
                    except RuntimeError:
                        break
                break  # file EOF

            if src_idx % step != 0:
                src_idx += 1
                continue

            frame = self._apply_roi(frame)
            pts_s = time.monotonic() - t_start if self._is_live else src_idx / native_fps
            yield Frame(index=out_idx, pts_s=pts_s, image=frame)
            out_idx += 1
            src_idx += 1

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def native_fps(self) -> float:
        return self._native_fps
