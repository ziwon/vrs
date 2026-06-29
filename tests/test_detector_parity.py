import json
from pathlib import Path

import pytest

from vrs.eval.detector_parity import (
    DetectionRecord,
    RuntimeSummary,
    bbox_iou_xyxy,
    compare_detector_outputs,
    load_detection_records,
    write_parity_report,
)


def _det(
    cls: str,
    score: float,
    bbox: tuple[float, float, float, float],
    *,
    frame_index: int = 1,
    runtime: str = "python",
) -> DetectionRecord:
    return DetectionRecord(
        class_name=cls,
        score=score,
        bbox_xyxy=bbox,
        frame_index=frame_index,
        stream_id="cam-1",
        clip_id="clip-a",
        source_runtime=runtime,
    )


def test_bbox_iou_xyxy() -> None:
    assert bbox_iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert bbox_iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_compare_detector_outputs_records_required_parity_fields() -> None:
    python = [
        _det("fire", 0.9, (0, 0, 10, 10)),
        _det("smoke", 0.8, (20, 20, 40, 40)),
    ]
    candidate = [
        _det("flame", 0.85, (1, 1, 11, 11), runtime="deepstream"),
        _det("person", 0.7, (50, 50, 70, 70), runtime="deepstream"),
    ]

    report = compare_detector_outputs(
        python_records=python,
        candidate_records=[
            DetectionRecord(
                class_name="fire" if det.class_name == "flame" else det.class_name,
                score=det.score,
                bbox_xyxy=det.bbox_xyxy,
                frame_index=det.frame_index,
                stream_id=det.stream_id,
                clip_id=det.clip_id,
                source_runtime=det.source_runtime,
            )
            for det in candidate
        ],
        class_mapping={"flame": "fire"},
        iou_threshold=0.5,
        python_runtime=RuntimeSummary(
            latency_ms={"p50": 6.0},
            throughput_fps=120.0,
            queue_drops=0,
            gpu_memory={"peak_gb": 1.2},
        ),
        candidate_runtime=RuntimeSummary(
            latency_ms={"p50": 3.0},
            throughput_fps=240.0,
            queue_drops=1,
            gpu_memory={"peak_gb": 0.9},
        ),
    )

    assert report["schema_version"] == "vrs.eval.detector_parity.v1"
    assert report["class_mapping"] == {"flame": "fire"}
    assert report["totals"] == {
        "python_count": 2,
        "candidate_count": 2,
        "matched": 1,
        "unmatched_python": 1,
        "unmatched_candidate": 1,
    }
    assert report["bbox"]["mean_iou"] is not None
    assert report["confidence"]["mean_abs_delta"] == pytest.approx(0.05)
    assert report["runtime"]["candidate"]["queue_drops"] == 1
    assert report["runtime"]["candidate"]["gpu_memory"]["peak_gb"] == 0.9
    assert report["per_class"]["smoke"]["unmatched_python"] == 1
    assert report["per_class"]["person"]["unmatched_candidate"] == 1


def test_load_detection_records_accepts_jsonl_and_class_mapping(tmp_path: Path) -> None:
    path = tmp_path / "detections.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "detection.v1",
                "class_name": "flame",
                "score": 0.9,
                "bbox_xyxy": [0, 0, 1, 1],
                "frame_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_detection_records(path, class_mapping={"flame": "fire"})

    assert records[0].class_name == "fire"
    assert records[0].bbox_xyxy == (0.0, 0.0, 1.0, 1.0)


def test_write_parity_report_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "parity.json"

    write_parity_report(out, {"schema_version": "vrs.eval.detector_parity.v1"})

    assert json.loads(out.read_text(encoding="utf-8"))["schema_version"].endswith(".v1")
