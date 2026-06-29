import json
from pathlib import Path

from vrs.deepstream.worker import (
    convert_metadata_file,
    convert_metadata_payload,
    write_detection_jsonl,
)


def test_convert_metadata_payload_accepts_deepstream_rect_fields() -> None:
    records = convert_metadata_payload(
        {
            "stream_id": "cam-1",
            "clip_id": "clip-a",
            "frame_index": 10,
            "pts_s": 2.5,
            "class_name": "fire",
            "confidence": 0.9,
            "left": 1,
            "top": 2,
            "width": 10,
            "height": 20,
            "track_id": 7,
        }
    )

    assert records[0]["schema_version"] == "detection.v1"
    assert records[0]["source_runtime"] == "deepstream"
    assert records[0]["bbox_xyxy"] == [1.0, 2.0, 11.0, 22.0]
    assert records[0]["track_id"] == 7


def test_convert_metadata_file_accepts_jsonl_and_writes_detection_jsonl(tmp_path: Path) -> None:
    source = tmp_path / "metadata.jsonl"
    source.write_text(
        json.dumps(
            {
                "stream_id": "cam-1",
                "frame_index": 1,
                "class_name": "smoke",
                "score": 0.8,
                "bbox_xyxy": [0, 1, 2, 3],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "detections.jsonl"

    records = convert_metadata_file(source)
    write_detection_jsonl(out, records)

    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema_version"] == "detection.v1"
    assert written["class_name"] == "smoke"
