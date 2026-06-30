import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_COMPARE_SCRIPT = Path("scripts/compare_yoloe_raw_tensors.py")
_COMPARE_SPEC = importlib.util.spec_from_file_location("compare_yoloe_raw_tensors", _COMPARE_SCRIPT)
assert _COMPARE_SPEC is not None
assert _COMPARE_SPEC.loader is not None
_COMPARE = importlib.util.module_from_spec(_COMPARE_SPEC)
sys.modules[_COMPARE_SPEC.name] = _COMPARE
_COMPARE_SPEC.loader.exec_module(_COMPARE)

_DUMP_SCRIPT = Path("scripts/dump_yoloe_pytorch_raw.py")
_DUMP_SPEC = importlib.util.spec_from_file_location("dump_yoloe_pytorch_raw", _DUMP_SCRIPT)
assert _DUMP_SPEC is not None
assert _DUMP_SPEC.loader is not None
_DUMP = importlib.util.module_from_spec(_DUMP_SPEC)
sys.modules[_DUMP_SPEC.name] = _DUMP
_DUMP_SPEC.loader.exec_module(_DUMP)

_TRTEXEC_SCRIPT = Path("scripts/convert_trtexec_output.py")
_TRTEXEC_SPEC = importlib.util.spec_from_file_location("convert_trtexec_output", _TRTEXEC_SCRIPT)
assert _TRTEXEC_SPEC is not None
assert _TRTEXEC_SPEC.loader is not None
_TRTEXEC = importlib.util.module_from_spec(_TRTEXEC_SPEC)
sys.modules[_TRTEXEC_SPEC.name] = _TRTEXEC
_TRTEXEC_SPEC.loader.exec_module(_TRTEXEC)

_RAW_RGB_SCRIPT = Path("scripts/convert_raw_rgb_to_tensor.py")
_RAW_RGB_SPEC = importlib.util.spec_from_file_location("convert_raw_rgb_to_tensor", _RAW_RGB_SCRIPT)
assert _RAW_RGB_SPEC is not None
assert _RAW_RGB_SPEC.loader is not None
_RAW_RGB = importlib.util.module_from_spec(_RAW_RGB_SPEC)
sys.modules[_RAW_RGB_SPEC.name] = _RAW_RGB
_RAW_RGB_SPEC.loader.exec_module(_RAW_RGB)

_RAW_DETECTIONS_SCRIPT = Path("scripts/export_yoloe_raw_detections.py")
_RAW_DETECTIONS_SPEC = importlib.util.spec_from_file_location(
    "export_yoloe_raw_detections", _RAW_DETECTIONS_SCRIPT
)
assert _RAW_DETECTIONS_SPEC is not None
assert _RAW_DETECTIONS_SPEC.loader is not None
_RAW_DETECTIONS = importlib.util.module_from_spec(_RAW_DETECTIONS_SPEC)
sys.modules[_RAW_DETECTIONS_SPEC.name] = _RAW_DETECTIONS
_RAW_DETECTIONS_SPEC.loader.exec_module(_RAW_DETECTIONS)


def test_normalize_yoloe_output_strips_batch_and_transposes_anchor_first() -> None:
    batch = np.zeros((1, 48, 8400), dtype=np.float32)
    assert _COMPARE.normalize_yoloe_output(batch).shape == (48, 8400)

    anchor_first = np.zeros((8400, 48), dtype=np.float32)
    assert _COMPARE.normalize_yoloe_output(anchor_first).shape == (48, 8400)


def test_compare_arrays_reports_delta_metrics() -> None:
    left = np.array([[1.0, 2.0]], dtype=np.float32)
    right = np.array([[1.0, 4.0]], dtype=np.float32)

    report = _COMPARE.compare_arrays(left, right)

    assert report["count"] == 2
    assert report["max_abs_delta"] == pytest.approx(2.0)
    assert report["mean_abs_delta"] == pytest.approx(1.0)
    assert report["cosine_similarity"] is not None
    assert report["per_channel"][0]["channel"] == 0
    assert report["per_channel"][0]["max_abs_delta"] == pytest.approx(2.0)
    assert report["top_abs_deltas"][0]["anchor"] == 1
    assert report["top_abs_deltas"][0]["abs_delta"] == pytest.approx(2.0)


def test_build_anchor_filter_keeps_high_score_anchors_from_either_side() -> None:
    left = np.zeros((6, 3), dtype=np.float32)
    right = np.zeros((6, 3), dtype=np.float32)
    left[4, 1] = 0.2
    right[5, 2] = 0.3

    mask = _COMPARE.build_anchor_filter(
        left,
        right,
        score_start=4,
        score_end=None,
        min_score=0.1,
        class_count=2,
    )

    assert mask.tolist() == [False, True, True]


def test_load_dump_reads_metadata_relative_binary(tmp_path: Path) -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    arr.tofile(tmp_path / "tensor.f32")
    (tmp_path / "tensor.json").write_text(
        json.dumps({"binary": "tensor.f32", "dims": [2, 3]}),
        encoding="utf-8",
    )

    loaded = _COMPARE.load_dump(tmp_path / "tensor.json")

    assert loaded["array"].shape == (2, 3)
    assert np.array_equal(loaded["array"], arr)


def test_letterbox_bgr_records_padding() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas, meta = _DUMP.letterbox_bgr(image, size=640, pad_value=114)

    assert canvas.shape == (640, 640, 3)
    assert meta["pad_x"] == 0
    assert meta["pad_y"] == 140
    assert meta["scale"] == 1.0


def test_convert_trtexec_helpers_select_tensor_and_parse_dims() -> None:
    payload = [
        {"name": "output1", "dimensions": "1x2", "values": [0, 1]},
        {"name": "output0", "dimensions": "1x2x3", "values": list(range(6))},
    ]

    tensor = _TRTEXEC.select_tensor(payload, "output0")

    assert tensor["values"] == list(range(6))
    assert _TRTEXEC.parse_dims(tensor["dimensions"]) == [1, 2, 3]


def test_convert_raw_rgb_to_tensor_outputs_chw_float32(tmp_path: Path) -> None:
    raw = np.array(
        [
            [[0, 127, 255], [255, 127, 0]],
            [[10, 20, 30], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )
    path = tmp_path / "frame.rgb"
    raw.tofile(path)

    tensor = _RAW_RGB.raw_rgb_to_chw_float32(path, width=2, height=2)

    assert tensor.shape == (1, 3, 2, 2)
    assert tensor.dtype == np.float32
    assert tensor[0, 0, 0, 1] == pytest.approx(1.0)
    assert tensor[0, 2, 0, 0] == pytest.approx(1.0)


def test_export_raw_detections_reads_raw_rgb_as_bgr_frame(tmp_path: Path) -> None:
    rgb = np.array([[[1, 2, 3]]], dtype=np.uint8)
    path = tmp_path / "frame.rgb"
    rgb.tofile(path)

    frame = _RAW_DETECTIONS.read_raw_rgb_frame(str(path), width=1, height=1)

    assert frame.shape == (1, 1, 3)
    assert frame[0, 0].tolist() == [3, 2, 1]
