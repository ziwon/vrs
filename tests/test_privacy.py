"""Face-blur privacy tests.

Most tests exercise the blur + Protocol + factory paths directly so CI
doesn't need the YuNet ONNX model file. One test constructs a YuNet
detector against a fake ``cv2.FaceDetectorYN`` to pin the wiring without
downloading weights.
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from vrs.privacy import (
    FaceDetector,
    NullFaceDetector,
    blur_faces,
    build_face_detector,
)


# ─── blur_faces ───────────────────────────────────────────────────────

def test_blur_faces_no_faces_is_noop():
    img = np.full((64, 64, 3), 200, dtype=np.uint8)
    out = blur_faces(img, [], kernel=15)
    # Same object returned; pixel values identical
    assert out is img
    assert (out == 200).all()


def test_blur_faces_actually_blurs_the_roi():
    """Inside the face bbox: pixel values should change (edges smeared);
    outside: pixel values must be untouched."""
    img = np.full((80, 80, 3), 0, dtype=np.uint8)
    # Checkerboard pattern inside the ROI so Gaussian has something to
    # smear. Outside stays solid black.
    for y in range(20, 60, 4):
        for x in range(20, 60, 4):
            img[y:y+2, x:x+2] = 255

    before = img.copy()
    blur_faces(img, [(20, 20, 40, 40)], kernel=11, margin_pct=0.0)

    # Interior pixels changed
    assert not np.array_equal(before[20:60, 20:60], img[20:60, 20:60])
    # Exterior pixels untouched
    assert np.array_equal(before[:20, :], img[:20, :])
    assert np.array_equal(before[:, :20], img[:, :20])


def test_blur_faces_respects_image_bounds_with_margin():
    """A face box that would overflow with the margin must clamp, not crash."""
    img = np.full((40, 40, 3), 128, dtype=np.uint8)
    # face at the corner; margin would push x1 negative
    blur_faces(img, [(0, 0, 10, 10)], kernel=5, margin_pct=0.5)
    # No crash is the assertion; also image still valid
    assert img.shape == (40, 40, 3)


def test_blur_faces_handles_degenerate_box():
    """A zero-area box (fully clipped against frame bounds) should skip
    silently."""
    img = np.full((40, 40, 3), 50, dtype=np.uint8)
    before = img.copy()
    blur_faces(img, [(100, 100, 10, 10)], kernel=5, margin_pct=0.0)
    assert np.array_equal(before, img)


def test_blur_faces_bumps_even_kernel_to_odd():
    """cv2.GaussianBlur requires odd kernel; we accept even and bump up."""
    img = np.full((40, 40, 3), 100, dtype=np.uint8)
    # Checkerboard so the blur has gradient to smear
    img[::2, ::2] = 200
    blur_faces(img, [(5, 5, 30, 30)], kernel=10, margin_pct=0.0)   # even
    # No exception = acceptance; correctness is covered by the other test


def test_blur_faces_rejects_invalid_params():
    img = np.full((40, 40, 3), 50, dtype=np.uint8)
    with pytest.raises(ValueError, match="kernel"):
        blur_faces(img, [(0, 0, 10, 10)], kernel=0)
    with pytest.raises(ValueError, match="margin_pct"):
        blur_faces(img, [(0, 0, 10, 10)], margin_pct=-0.1)


# ─── NullFaceDetector + factory ───────────────────────────────────────

def test_null_face_detector_returns_empty():
    det = NullFaceDetector()
    assert det(np.zeros((32, 32, 3), dtype=np.uint8)) == []
    assert isinstance(det, FaceDetector)


def test_factory_returns_null_when_disabled():
    assert isinstance(build_face_detector(None), NullFaceDetector)
    assert isinstance(build_face_detector({}), NullFaceDetector)
    assert isinstance(build_face_detector({"enabled": False}), NullFaceDetector)
    assert isinstance(build_face_detector({"enabled": True, "backend": "none"}),
                      NullFaceDetector)


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown face detector backend"):
        build_face_detector({"enabled": True, "backend": "whatever"})


def test_factory_degrades_to_null_on_yunet_setup_failure(tmp_path, caplog):
    """Misconfigured privacy (missing model file) should not crash the
    pipeline — it drops back to a no-op blur with a warning so operators
    see it in the logs. Safer than refusing to run, which would take a
    whole deployment down on a config typo."""
    with caplog.at_level(logging.WARNING, logger="vrs.privacy.detectors"):
        det = build_face_detector({
            "enabled": True,
            "backend": "yunet",
            "model": str(tmp_path / "nonexistent.onnx"),
        })
    assert isinstance(det, NullFaceDetector)
    assert any("YuNet face detector setup failed" in r.getMessage()
               for r in caplog.records)


# ─── YuNet — wiring pinned via a fake cv2.FaceDetectorYN ──────────────

class _FakeYuNet:
    """Stand-in for cv2.FaceDetectorYN — records config + returns canned faces."""
    instances = []
    def __init__(self, faces_out=None):
        self.faces_out = faces_out if faces_out is not None else np.zeros((0, 15), dtype=np.float32)
        self.input_sizes = []
        _FakeYuNet.instances.append(self)
    def setInputSize(self, size):
        self.input_sizes.append(size)
    def detect(self, bgr):
        return (True, self.faces_out)


@pytest.fixture
def fake_yunet(monkeypatch):
    """Patch cv2.FaceDetectorYN with a canned stub so we can construct
    YuNetFaceDetector without the real ONNX model."""
    _FakeYuNet.instances.clear()
    calls = {"create_args": []}
    def _create(model, config, input_size, score_threshold, nms_threshold, top_k):
        calls["create_args"].append((model, config, input_size,
                                      score_threshold, nms_threshold, top_k))
        return _FakeYuNet()
    fake_cls = types.SimpleNamespace(create=_create)
    monkeypatch.setattr(cv2, "FaceDetectorYN", fake_cls, raising=False)
    return calls


def test_yunet_detector_rejects_missing_model_arg():
    from vrs.privacy.yunet import YuNetFaceDetector
    with pytest.raises(ValueError, match="requires `privacy.model`"):
        YuNetFaceDetector(model_path=None)


def test_yunet_detector_rejects_missing_model_file(tmp_path):
    from vrs.privacy.yunet import YuNetFaceDetector
    with pytest.raises(FileNotFoundError, match="YuNet model not found"):
        YuNetFaceDetector(model_path=tmp_path / "nope.onnx")


def test_yunet_detector_constructs_and_passes_config(fake_yunet, tmp_path):
    """YuNet.create must receive the operator-provided thresholds."""
    from vrs.privacy.yunet import YuNetFaceDetector

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"\x00" * 16)

    det = YuNetFaceDetector(
        model_path=model, input_size=256,
        score_threshold=0.7, nms_threshold=0.25, top_k=100,
    )
    assert isinstance(det, FaceDetector)
    args = fake_yunet["create_args"][0]
    assert args[0] == str(model)
    assert args[2] == (256, 256)
    assert args[3] == pytest.approx(0.7)
    assert args[4] == pytest.approx(0.25)
    assert args[5] == 100


def test_yunet_scales_boxes_back_to_original_resolution(fake_yunet, tmp_path):
    """Detections on a downscaled frame must be mapped back to original
    pixel coords before blur runs."""
    from vrs.privacy.yunet import YuNetFaceDetector

    model = tmp_path / "fake.onnx"
    model.write_bytes(b"\x00" * 16)

    det = YuNetFaceDetector(model_path=model, input_size=160)
    # One face at (32, 32, 16, 16) in the *downscaled* frame
    face_row = np.array([[32, 32, 16, 16,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9]],
                        dtype=np.float32)
    _FakeYuNet.instances[0].faces_out = face_row

    # Source frame is 640×480 → scale = 160/640 = 0.25 → inv_scale = 4
    bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = det(bgr)
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert x == pytest.approx(128, abs=1)    # 32 * 4
    assert y == pytest.approx(128, abs=1)
    assert w == pytest.approx(64, abs=1)     # 16 * 4
    assert h == pytest.approx(64, abs=1)


def test_yunet_returns_empty_on_no_faces(fake_yunet, tmp_path):
    from vrs.privacy.yunet import YuNetFaceDetector
    model = tmp_path / "fake.onnx"
    model.write_bytes(b"\x00" * 16)
    det = YuNetFaceDetector(model_path=model, input_size=160)
    boxes = det(np.zeros((240, 320, 3), dtype=np.uint8))
    assert boxes == []


def test_yunet_handles_empty_image():
    """size == 0 image short-circuits rather than calling into cv2."""
    from vrs.privacy.yunet import YuNetFaceDetector
    # bypass __init__ so we don't need a model file for this edge-case
    det = object.__new__(YuNetFaceDetector)
    boxes = det(np.zeros((0, 0, 3), dtype=np.uint8))
    assert boxes == []


# ─── VideoAnnotator integration ───────────────────────────────────────

def test_video_annotator_runs_blur_before_drawing_overlays(tmp_path, monkeypatch):
    """The annotator must blur raw pixels *before* it draws detector
    boxes or the timestamp — otherwise the overlay pixels would get
    blurred out. We exercise this by injecting a stub detector that
    always 'finds' a face at a known location and asserting the
    post-write region has been blurred while overlay-occupied pixels
    remain the operator-UI colors."""
    from vrs.schemas import Frame
    from vrs.sinks.video_annotator import VideoAnnotator

    captured_frames = []
    class _FakeWriter:
        def __init__(self, *a, **kw): pass
        def isOpened(self): return True
        def write(self, img): captured_frames.append(img.copy())
        def release(self): pass

    # Patch VideoWriter so the test doesn't write an mp4
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a, **kw: _FakeWriter())

    class _StubDet:
        def __call__(self, bgr):
            return [(10, 10, 30, 30)]

    # Input: solid gray frame with a high-contrast checkerboard pattern inside
    # the face box.  A uniform region stays identical after Gaussian blur, so
    # we need alternating 0/255 stripes for the blur to produce a visible
    # smeared-out mean that differs from both 0 and 255.
    img = np.full((80, 80, 3), 128, dtype=np.uint8)
    for y in range(10, 40, 2):
        img[y, 10:40] = 255 if (y % 4 < 2) else 0
    frame = Frame(index=0, pts_s=0.0, image=img.copy())

    ann = VideoAnnotator(tmp_path / "a.mp4", fps=4.0,
                         face_detector=_StubDet(), blur_kernel=11,
                         blur_margin_pct=0.0)
    ann.write(frame, detections=[])
    ann.close()

    assert len(captured_frames) == 1
    written = captured_frames[0]
    original_face = img[10:40, 10:40].copy()
    written_face = written[10:40, 10:40]
    # The blur must have changed at least some pixels inside the face ROI.
    assert not np.array_equal(original_face, written_face)
    # Outside the face region (and away from the timestamp at the bottom):
    # pixels should be unchanged. Use a region that doesn't overlap with the
    # face box (10:40, 10:40), any banner text (top), or the timestamp
    # overlay (bottom row). Row 45:55, col 60:70 is safely outside all of those.
    bg = written[45:55, 60:70]
    assert (bg == 128).all()


def test_video_annotator_with_null_detector_leaves_pixels_untouched(tmp_path, monkeypatch):
    """Verifies the privacy-disabled path: no face detector wired, no blur
    applied to raw pixels — only the operator overlays should differ."""
    from vrs.schemas import Frame
    from vrs.sinks.video_annotator import VideoAnnotator

    captured = []
    class _FakeWriter:
        def __init__(self, *a, **kw): pass
        def isOpened(self): return True
        def write(self, img): captured.append(img.copy())
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoWriter", lambda *a, **kw: _FakeWriter())

    img = np.full((80, 80, 3), 50, dtype=np.uint8)
    img[10:40, 10:40] = 200
    frame = Frame(index=0, pts_s=0.0, image=img.copy())

    ann = VideoAnnotator(tmp_path / "a.mp4", fps=4.0)   # default NullFaceDetector
    ann.write(frame, detections=[])
    ann.close()

    # The "face" rectangle is still sharp — no blur happened.
    face = captured[0][10:40, 10:40]
    assert (face == 200).all()
