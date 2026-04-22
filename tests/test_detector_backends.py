"""Tests for the detector backend abstraction.

No GPU needed. The ultralytics backend itself (``YOLOEDetector``) already
has coverage through the full pipeline tests; these tests pin the
Protocol + factory + TRT-skeleton input validation so the seam is safe
to extend.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.triage import Detector, build_detector
from vrs.triage.backends import _KNOWN_BACKENDS
from vrs.triage.yoloe_detector import YOLOEConfig


def _policy() -> WatchPolicy:
    return WatchPolicy([
        WatchItem(name="fire", detector_prompts=["fire", "flame"],
                  verifier_prompt="flames", severity="critical",
                  min_score=0.3, min_persist_frames=2),
    ])


def _fake_ultralytics_yolo(records: dict):
    fake = types.ModuleType("ultralytics")
    class _FakeYOLO:
        def __init__(self, path, task=None):
            records["init_args"] = {"path": path, "task": task}
        def predict(self, images, **kwargs):
            records["predict_kwargs"] = kwargs
            return []                       # empty results → empty detections
    fake.YOLO = _FakeYOLO
    return fake


# ─── factory ──────────────────────────────────────────────────────────

def test_build_detector_rejects_unknown():
    with pytest.raises(ValueError, match="unknown detector backend"):
        build_detector(YOLOEConfig(), _policy(), backend="banana")


def test_known_backends_set_matches_factory():
    """Silent drift between advertised backend names and factory branches
    is a real risk — this test catches it."""
    assert _KNOWN_BACKENDS == {"ultralytics", "tensorrt"}


# ─── tensorrt backend — input validation ─────────────────────────────

def test_tensorrt_backend_rejects_non_engine_path():
    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    cfg = YOLOEConfig(model="yoloe-11l-seg.pt")
    with pytest.raises(ValueError, match=".engine"):
        TensorRTYOLOEDetector(cfg, _policy())


def test_tensorrt_backend_rejects_missing_file(tmp_path: Path):
    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    cfg = YOLOEConfig(model=str(tmp_path / "nope.engine"))
    with pytest.raises(FileNotFoundError, match="TRT engine not found"):
        TensorRTYOLOEDetector(cfg, _policy())


def test_tensorrt_backend_loads_engine_with_ultralytics_yolo(monkeypatch, tmp_path: Path):
    records: dict = {}
    fake = _fake_ultralytics_yolo(records)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    engine = tmp_path / "yoloe.engine"
    engine.write_bytes(b"\x00" * 16)      # non-empty placeholder

    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    cfg = YOLOEConfig(model=str(engine), imgsz=640, conf_floor=0.20, iou=0.50)
    det = TensorRTYOLOEDetector(cfg, _policy())

    # Protocol conformance — the verifier-style Protocol contract.
    assert isinstance(det, Detector)

    # The ultralytics YOLO wrapper was initialized with the .engine path.
    assert records["init_args"]["path"] == str(engine)
    assert records["init_args"]["task"] == "detect"


def test_tensorrt_backend_batch_forwards_predict_kwargs(monkeypatch, tmp_path: Path):
    records: dict = {}
    fake = _fake_ultralytics_yolo(records)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    engine = tmp_path / "yoloe.engine"
    engine.write_bytes(b"\x00" * 16)

    from vrs.schemas import Frame
    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    import numpy as np

    cfg = YOLOEConfig(model=str(engine), imgsz=512, conf_floor=0.25, iou=0.40,
                      device="cuda")
    det = TensorRTYOLOEDetector(cfg, _policy())
    frame = Frame(index=0, pts_s=0.0, image=np.zeros((32, 32, 3), dtype=np.uint8))
    out = det.batch([frame])

    assert out == [[]]                    # empty upstream → empty detections
    kwargs = records["predict_kwargs"]
    # precision flags are deliberately NOT forwarded: precision is baked into
    # the engine at export time and ultralytics' engine loader rejects half=.
    assert "half" not in kwargs
    assert kwargs["imgsz"] == 512
    assert kwargs["conf"] == pytest.approx(0.25)
    assert kwargs["iou"] == pytest.approx(0.40)
    assert kwargs["device"] == "cuda"
    assert kwargs["verbose"] is False


def test_tensorrt_backend_empty_frames_short_circuits(monkeypatch, tmp_path: Path):
    records: dict = {}
    fake = _fake_ultralytics_yolo(records)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    engine = tmp_path / "yoloe.engine"
    engine.write_bytes(b"\x00" * 16)

    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    det = TensorRTYOLOEDetector(YOLOEConfig(model=str(engine)), _policy())
    assert det.batch([]) == []
    assert "predict_kwargs" not in records    # predict was never called


def test_factory_routes_tensorrt_to_trt_detector(monkeypatch, tmp_path: Path):
    records: dict = {}
    fake = _fake_ultralytics_yolo(records)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    engine = tmp_path / "yoloe.engine"
    engine.write_bytes(b"\x00" * 16)

    cfg = YOLOEConfig(model=str(engine))
    det = build_detector(cfg, _policy(), backend="tensorrt")
    from vrs.triage.tensorrt_detector import TensorRTYOLOEDetector
    assert isinstance(det, TensorRTYOLOEDetector)
