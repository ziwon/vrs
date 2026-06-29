import json
from pathlib import Path

import numpy as np

from vrs.eval.detection_export import detections_to_contracts, write_detection_jsonl
from vrs.schemas import Detection, Frame


def test_detections_to_contracts_includes_clip_and_frame_context() -> None:
    frame = Frame(index=3, pts_s=0.75, image=np.zeros((8, 8, 3), dtype=np.uint8))

    records = detections_to_contracts(
        [Detection(class_name="fire", score=0.9, xyxy=(1, 2, 3, 4))],
        frame=frame,
        stream_id="cam-1",
        clip_id="clip-a",
        detector_id="python-yoloe",
    )

    assert records[0]["schema_version"] == "detection.v1"
    assert records[0]["source_runtime"] == "python"
    assert records[0]["stream_id"] == "cam-1"
    assert records[0]["clip_id"] == "clip-a"
    assert records[0]["frame_index"] == 3
    assert records[0]["pts_s"] == 0.75
    assert records[0]["detector_id"] == "python-yoloe"


def test_write_detection_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "detections.jsonl"
    write_detection_jsonl(path, [{"schema_version": "detection.v1", "class_name": "fire"}])

    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"] == "detection.v1"
