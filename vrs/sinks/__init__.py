"""Sinks — jsonl writer, event thumbnails, and optional annotated-video writer.

``EventThumbnailSink`` and ``VideoAnnotator`` pull cv2 lazily; they are
lazy-loaded via PEP 562 so CPU-only tests that only need ``JsonlSink`` don't
force an OpenCV install.
"""

from .jsonl_sink import JsonlSink

__all__ = ["EventThumbnailSink", "JsonlSink", "VideoAnnotator"]


def __getattr__(name):
    if name == "EventThumbnailSink":
        from .thumbnail_sink import EventThumbnailSink

        return EventThumbnailSink
    if name == "VideoAnnotator":
        from .video_annotator import VideoAnnotator

        return VideoAnnotator
    raise AttributeError(f"module 'vrs.sinks' has no attribute {name!r}")
