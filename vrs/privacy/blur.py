"""In-place Gaussian blur of face ROIs.

Separate module so the blur pass has a clean seam — a test can swap a
fake detector in, feed a known image, and assert exactly which pixels
get blurred without exercising the full annotator.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .detectors import FaceBox


def blur_faces(
    bgr: np.ndarray,
    faces: Iterable[FaceBox],
    *,
    kernel: int = 31,
    margin_pct: float = 0.15,
) -> np.ndarray:
    """Gaussian-blur every face ROI in ``bgr``. Mutates ``bgr``.

    Args:
        bgr: the frame to blur. Caller is responsible for copying if the
            original must be preserved.
        faces: (x, y, w, h) boxes in absolute pixels.
        kernel: Gaussian kernel size (must be odd; we bump even values up).
        margin_pct: fractional margin added around each face before blur
            so edges (hair, chin, ear) are covered instead of leaving a
            sharp square of identifiable pixels.

    Returns ``bgr`` (same object) for call-chain convenience.
    """
    if kernel < 1:
        raise ValueError("kernel must be >= 1")
    if kernel % 2 == 0:
        kernel += 1     # cv2.GaussianBlur requires odd kernel
    if margin_pct < 0:
        raise ValueError("margin_pct must be >= 0")

    # Lazy import so privacy package is import-safe on CPU-only hosts.
    import cv2

    h, w = bgr.shape[:2]
    for (x, y, fw, fh) in faces:
        mx = int(fw * margin_pct)
        my = int(fh * margin_pct)
        x1 = max(0, x - mx)
        y1 = max(0, y - my)
        x2 = min(w, x + fw + mx)
        y2 = min(h, y + fh + my)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = bgr[y1:y2, x1:x2]
        bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), 0)
    return bgr
