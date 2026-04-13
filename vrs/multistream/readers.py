"""Pluggable frame readers for multi-stream decode.

Two backends:

  * OpenCVReader (default) — FFmpeg CPU decode. Works everywhere.
  * NvDecReader (optional) — ``cv2.cudacodec.VideoReader``. Uses NVDEC for
    hardware-accelerated H.264/H.265 decode and delivers frames via
    ``cv2.cuda_GpuMat`` (zero-copy on GPU until we convert to BGR for
    YOLOE / Cosmos). Requires OpenCV built with CUDA; gracefully falls back
    to OpenCVReader when unavailable.

A ``Reader`` is an iterable of ``Frame``. Both backends honor ``target_fps``
(nominally 4, to match Cosmos-Reason2-2B's training distribution) and
optional ``roi_polygon``.

──────────────────────────────────────────────────────────────────────────
DeepStream 8.0 path (future):
Replacing OpenCVReader with a pyds pipeline (``nvurisrcbin → nvstreammux →
nvvideoconvert → appsink``) yields true zero-copy NVMM surfaces that can
feed a TRT YOLOE engine via ``nvinfer`` without ever touching host memory.
We keep the Reader interface small so that integration is additive — just
implement a new reader that yields ``Frame`` and register it below.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple

import cv2
import numpy as np

from ..schemas import Frame


class Reader(ABC):
    """Abstract frame-yielding iterator."""
    native_fps: float = 0.0

    @abstractmethod
    def __iter__(self) -> Iterator[Frame]:
        ...


# ──────────────────────────────────────────────────────────────────────
# OpenCV (CPU FFmpeg) — always available
# ──────────────────────────────────────────────────────────────────────

class OpenCVReader(Reader):
    def __init__(
        self,
        source: str,
        target_fps: float = 4.0,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
        reconnect_s: float = 2.0,
    ):
        self.source = source
        self.target_fps = float(target_fps)
        self.roi = (
            np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        )
        self.reconnect_s = reconnect_s
        self._is_live = source.startswith(("rtsp://", "rtmp://", "http://", "https://")) or source.isdigit()

    def _open(self) -> cv2.VideoCapture:
        cap = (
            cv2.VideoCapture(int(self.source))
            if self.source.isdigit()
            else cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        )
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.source}")
        self.native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        return cap

    def _mask(self, img: np.ndarray) -> np.ndarray:
        if self.roi is None:
            return img
        m = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [self.roi], 255)
        return cv2.bitwise_and(img, img, mask=m)

    def __iter__(self) -> Iterator[Frame]:
        cap = self._open()
        step = max(1, int(round(max(self.native_fps, 1.0) / max(self.target_fps, 0.1))))
        src_idx = 0
        out_idx = 0
        t0 = time.monotonic()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    if self._is_live:
                        cap.release()
                        time.sleep(self.reconnect_s)
                        try:
                            cap = self._open()
                            continue
                        except RuntimeError:
                            break
                    break

                if src_idx % step != 0:
                    src_idx += 1
                    continue

                frame = self._mask(frame)
                pts_s = (
                    time.monotonic() - t0
                    if self._is_live
                    else src_idx / max(self.native_fps, 1.0)
                )
                yield Frame(index=out_idx, pts_s=pts_s, image=frame)
                out_idx += 1
                src_idx += 1
        finally:
            cap.release()


# ──────────────────────────────────────────────────────────────────────
# NVDEC via cv2.cudacodec — optional, zero-copy decode on GPU
# ──────────────────────────────────────────────────────────────────────

def _has_cudacodec() -> bool:
    """True when OpenCV was built with CUDA and cudacodec is usable."""
    try:
        cuda_mod = getattr(cv2, "cuda", None)
        codec_mod = getattr(cv2, "cudacodec", None)
        if cuda_mod is None or codec_mod is None:
            return False
        return cuda_mod.getCudaEnabledDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


class NvDecReader(Reader):
    """Hardware-accelerated decode via NVDEC.

    Delivers frames as BGR numpy arrays (we download from the GPU surface
    right before handing off to the rest of the pipeline). For a pure-GPU
    path, feed the GpuMat directly into a TRT YOLOE engine — that's the
    future DeepStream-style optimization.
    """

    def __init__(
        self,
        source: str,
        target_fps: float = 4.0,
        roi_polygon: Optional[List[Tuple[float, float]]] = None,
    ):
        if not _has_cudacodec():
            raise RuntimeError(
                "cv2.cudacodec not available — rebuild OpenCV with CUDA, or use OpenCVReader"
            )
        self.source = source
        self.target_fps = float(target_fps)
        self.roi = (
            np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        )

    def __iter__(self) -> Iterator[Frame]:
        reader = cv2.cudacodec.createVideoReader(self.source)  # type: ignore[attr-defined]
        fmt = reader.format() if hasattr(reader, "format") else None
        self.native_fps = float(getattr(fmt, "fps", 25.0) or 25.0)
        step = max(1, int(round(self.native_fps / max(self.target_fps, 0.1))))
        src_idx = 0
        out_idx = 0
        t0 = time.monotonic()
        while True:
            ok, gpu_frame = reader.nextFrame()
            if not ok or gpu_frame is None:
                break
            if src_idx % step != 0:
                src_idx += 1
                continue

            # download to host — replace this with a GPU path when we TRT-ify YOLOE
            host_rgba = gpu_frame.download()
            if host_rgba.shape[-1] == 4:
                bgr = cv2.cvtColor(host_rgba, cv2.COLOR_RGBA2BGR)
            else:
                bgr = host_rgba

            if self.roi is not None:
                m = np.zeros(bgr.shape[:2], dtype=np.uint8)
                cv2.fillPoly(m, [self.roi], 255)
                bgr = cv2.bitwise_and(bgr, bgr, mask=m)

            pts_s = time.monotonic() - t0  # cudacodec doesn't expose PTS cheaply
            yield Frame(index=out_idx, pts_s=pts_s, image=bgr)
            out_idx += 1
            src_idx += 1


# ──────────────────────────────────────────────────────────────────────
# factory
# ──────────────────────────────────────────────────────────────────────

def build_reader(
    backend: str,
    source: str,
    target_fps: float,
    roi_polygon: Optional[List[Tuple[float, float]]] = None,
) -> Reader:
    backend = (backend or "opencv").lower()
    if backend == "opencv":
        return OpenCVReader(source, target_fps, roi_polygon)
    if backend in ("nvdec", "cudacodec"):
        if not _has_cudacodec():
            # graceful fallback, logged via a print so operators see it
            print(f"[WARN] nvdec requested but cv2.cudacodec unavailable — falling back to opencv for {source}")
            return OpenCVReader(source, target_fps, roi_polygon)
        return NvDecReader(source, target_fps, roi_polygon)
    raise ValueError(f"unknown decoder backend: {backend!r}")
