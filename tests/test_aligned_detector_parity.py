import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path("scripts/compare_aligned_detector_parity.py")
_SPEC = importlib.util.spec_from_file_location("compare_aligned_detector_parity", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

Record = _MODULE.Record
compare = _MODULE.compare
map_record = _MODULE.map_record
transform_record = _MODULE.transform_record


def test_aligned_parity_maps_prompt_and_transforms_bbox() -> None:
    python = [
        Record(
            class_name="smoke",
            raw_label="smoke",
            score=0.5,
            bbox_xyxy=(10.0, 20.0, 50.0, 60.0),
            frame_index=1,
            pts_s=1.0,
        )
    ]
    candidate = Record(
        class_name="billowing smoke",
        raw_label="billowing smoke",
        score=0.7,
        bbox_xyxy=(20.0, 140.0, 100.0, 220.0),
        frame_index=24,
        pts_s=1.02,
    )

    mapped = map_record(candidate, {"billowing smoke": "smoke"})
    transformed = transform_record(mapped, sx=0.5, sy=0.5, ox=0.0, oy=-100.0)
    report = compare(
        python,
        [transformed],
        time_tolerance_s=0.06,
        iou_threshold=0.5,
    )

    assert report["totals"]["matched"] == 1
    assert report["matched_by_class"] == {"smoke": 1}
    assert report["bbox"]["min"] == 1.0
