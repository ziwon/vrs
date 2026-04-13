"""Sinks — jsonl writer and (optional) annotated-video writer.

``VideoAnnotator`` pulls cv2; it's lazy-loaded via PEP 562 so CPU-only tests
that only need ``JsonlSink`` don't force an OpenCV install.
"""
from .jsonl_sink import JsonlSink

__all__ = ["JsonlSink", "VideoAnnotator"]


def __getattr__(name):
    if name == "VideoAnnotator":
        from .video_annotator import VideoAnnotator
        return VideoAnnotator
    raise AttributeError(f"module 'vrs.sinks' has no attribute {name!r}")
