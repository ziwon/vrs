"""YuNet face detector via ``cv2.FaceDetectorYN``.

YuNet is small (~300 KB), fast on CPU, and ships with OpenCV ≥ 4.8 — so
privacy blurring costs us no new heavy dependency on top of what the
pipeline already needs for decode. The model weights are a separate ONNX
file that operators download once per deployment (tiny; fine to cache in
a volume).

Download::

    curl -L -o face_detection_yunet_2023mar.onnx \\
        https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

The 2023-March checkpoint is the current stable; newer revisions drop in
without code changes.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .detectors import FaceBox

logger = logging.getLogger(__name__)


class YuNetFaceDetector:
    """``FaceDetector`` backed by OpenCV's bundled YuNet runtime.

    The input size is dynamic — OpenCV accepts arbitrary ``(w, h)`` but we
    resize the long edge to ``input_size`` for speed and then map boxes
    back to the original resolution. This keeps detection cost roughly
    constant regardless of source frame resolution.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        input_size: int = 320,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        import cv2

        if model_path is None:
            raise ValueError(
                "YuNet backend requires `privacy.model` — download "
                "face_detection_yunet_2023mar.onnx (see module docstring)."
            )
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found: {model_path}. "
                "Check the path or download the ONNX file per the module docstring."
            )
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "cv2.FaceDetectorYN is not available in this OpenCV build. "
                "Upgrade to opencv-python >= 4.8."
            )

        self.input_size = int(input_size)
        self.score_threshold = float(score_threshold)
        # OpenCV's constructor takes (model, config, input_size, score_threshold,
        # nms_threshold, top_k). config is unused for YuNet.
        self._detector = cv2.FaceDetectorYN.create(
            str(model_path),
            "",
            (self.input_size, self.input_size),
            self.score_threshold,
            float(nms_threshold),
            int(top_k),
        )

    def __call__(self, bgr: np.ndarray) -> list[FaceBox]:
        if bgr.size == 0:
            return []
        h, w = bgr.shape[:2]

        # Resize long edge to input_size; preserve aspect ratio so faces
        # don't get squashed into false positives.
        scale = self.input_size / max(h, w)
        if scale < 1.0:
            import cv2

            new_w = round(w * scale)
            new_h = round(h * scale)
            resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = bgr
            new_w, new_h = w, h

        self._detector.setInputSize((new_w, new_h))
        ok, faces = self._detector.detect(resized)
        if not ok or faces is None or len(faces) == 0:
            return []

        inv_scale = 1.0 / scale if scale < 1.0 else 1.0

        out: list[FaceBox] = []
        for row in faces:
            # YuNet layout: [x, y, w, h, landmark xs/ys, score]. We only
            # need the bbox; landmarks are ignored because the blur is
            # an axis-aligned rectangle anyway.
            x, y, fw, fh = row[0], row[1], row[2], row[3]
            score = float(row[-1])
            if score < self.score_threshold:
                continue
            ax = round(max(0.0, x * inv_scale))
            ay = round(max(0.0, y * inv_scale))
            aw = round(fw * inv_scale)
            ah = round(fh * inv_scale)
            # clamp to image bounds
            aw = max(1, min(aw, w - ax))
            ah = max(1, min(ah, h - ay))
            out.append((ax, ay, aw, ah))
        return out
