"""Helpers for exporting Python detector output as detection.v1 records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schemas import Detection, Frame


def detections_to_contracts(
    detections: list[Detection],
    *,
    frame: Frame,
    stream_id: str,
    clip_id: str | None = None,
    detector_id: str | None = None,
    source_runtime: str = "python",
) -> list[dict[str, Any]]:
    return [
        det.to_contract(
            stream_id=stream_id,
            clip_id=clip_id,
            frame=frame,
            detector_id=detector_id,
            source_runtime=source_runtime,
        )
        for det in detections
    ]


def write_detection_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
