"""Face detector Protocol + ``NullFaceDetector`` pass-through + factory.

Split out from the YuNet implementation so the Protocol and the factory
can be imported without pulling in OpenCV's ``FaceDetectorYN``
dependency (only ultralytics CI hosts have the ONNX runtime hooked up).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# (x, y, w, h) in absolute pixels — OpenCV-style bbox
FaceBox = Tuple[int, int, int, int]


@runtime_checkable
class FaceDetector(Protocol):
    """Takes a BGR HxWx3 uint8 image, returns a list of face boxes."""
    def __call__(self, bgr: np.ndarray) -> List[FaceBox]: ...


class NullFaceDetector:
    """No-op detector — used when privacy is disabled. Kept as an explicit
    type so pipeline code can treat it like any other backend without a
    branch on ``None``."""
    def __call__(self, bgr: np.ndarray) -> List[FaceBox]:  # noqa: ARG002
        return []


def build_face_detector(cfg: Optional[dict]) -> FaceDetector:
    """Construct a face detector from a YAML block, or ``NullFaceDetector``
    when disabled / misconfigured (a misconfigured detector should never
    take the pipeline down — the blur pass silently becomes a no-op and
    the pipeline keeps running).
    """
    if not cfg or not cfg.get("enabled", False):
        return NullFaceDetector()
    backend = str(cfg.get("backend", "yunet")).lower()
    if backend in ("none", "null", ""):
        return NullFaceDetector()
    if backend == "yunet":
        from .yunet import YuNetFaceDetector
        try:
            return YuNetFaceDetector(
                model_path=cfg.get("model"),
                input_size=int(cfg.get("input_size", 320)),
                score_threshold=float(cfg.get("score_threshold", 0.6)),
                nms_threshold=float(cfg.get("nms_threshold", 0.3)),
                top_k=int(cfg.get("top_k", 5000)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "YuNet face detector setup failed (%s); face blurring "
                "will be disabled for this run. Check `privacy.model` "
                "points at a valid YuNet ONNX file.", e,
            )
            return NullFaceDetector()
    raise ValueError(f"unknown face detector backend: {backend!r}")
