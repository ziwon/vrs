"""Detector backend Protocol + factory.

Mirrors ``vrs/runtime/backends.py`` on the slow-path side: the pipeline
never imports a concrete detector directly, it goes through
``build_detector(cfg, policy, backend)``. This lets us swap in a
TRT-exported engine (produced via Ultralytics export or NVIDIA TAO)
without touching the pipeline code or the event-state queue.

Backends shipped:

* ``ultralytics`` — the pre-existing ``YOLOEDetector`` backed by
  ``ultralytics.YOLOE``. Default and well-tested.
* ``tensorrt``    — ``TensorRTYOLOEDetector`` skeleton. Loads a
  pre-exported TensorRT engine (``.engine`` file) and runs it via
  the Ultralytics ``YOLO`` wrapper, which natively accepts an engine
  path for its ``predict`` API. Prompt handling / text-PE loading is
  identical to the ultralytics path — the difference is purely the
  execution provider.
"""
from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from ..policy import WatchPolicy
from ..schemas import Detection, Frame


@runtime_checkable
class Detector(Protocol):
    """Minimum surface the pipeline needs from a detector."""
    def __call__(self, frame: Frame) -> List[Detection]: ...
    def batch(self, frames: List[Frame]) -> List[List[Detection]]: ...


# ──────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────

_KNOWN_BACKENDS = {"ultralytics", "tensorrt"}


def build_detector(cfg, policy: WatchPolicy, backend: str = "ultralytics") -> Detector:
    """Construct the detector backend named by ``backend``.

    Lazy-imports the backend module so hosts without a TRT runtime can
    still run the ultralytics default unchanged.
    """
    name = (backend or "ultralytics").lower()
    if name == "ultralytics":
        from .yoloe_detector import YOLOEDetector
        return YOLOEDetector(cfg, policy)
    if name == "tensorrt":
        from .tensorrt_detector import TensorRTYOLOEDetector
        return TensorRTYOLOEDetector(cfg, policy)
    raise ValueError(
        f"unknown detector backend: {backend!r}. Valid: {sorted(_KNOWN_BACKENDS)}"
    )
