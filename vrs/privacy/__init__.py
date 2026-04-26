"""Privacy primitives — face detection + blurring for written artifacts.

Scope: only retained operator artifacts (event thumbnails and optional
annotated MP4). The detector/verifier pipeline sees unaltered frames (the
verifier needs faces visible to reason about events like "a person has
collapsed"), but anything *written to disk* for human review goes through a
face-blur pass. This matches GDPR / K-GDPR expectations for CCTV retention.

Factory:

    faceblur:
      enabled: true
      backend: yunet            # yunet | none (default)
      model: face_detection_yunet_2023mar.onnx    # optional — see docstring
      input_size: 320
      score_threshold: 0.6
      blur_kernel: 31           # odd; larger = more blur
      margin_pct: 0.15          # expand each face bbox this fraction to cover
                                # partially-occluded edges (hairline, chin)

See ``yunet.py`` for the implementation. The Protocol below is small on
purpose — new backends (RetinaFace, MediaPipe) only need to expose
``__call__(bgr) -> List[Tuple]``.
"""

from __future__ import annotations

from .blur import blur_faces
from .detectors import (
    FaceBox,
    FaceDetector,
    NullFaceDetector,
    build_face_detector,
)

__all__ = [
    "FaceBox",
    "FaceDetector",
    "NullFaceDetector",
    "blur_faces",
    "build_face_detector",
]
