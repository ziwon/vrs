"""Minimal DeepStream metadata exporter boundary.

This worker does not run GStreamer or DeepStream. It converts metadata exported
by a DeepStream probe/plugin into canonical detection.v1 JSONL so the rest of
VRS can be exercised before the full DeepStream process is wired.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .adapter import DeepStreamDetectionMetadata, detection_from_deepstream


def convert_metadata_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        raw_items = payload["detections"]
    elif isinstance(payload, dict):
        raw_items = [payload]
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raise ValueError("metadata payload must be an object, array, or object with detections")
    return [
        detection_from_deepstream(DeepStreamDetectionMetadata.from_mapping(dict(item)))
        for item in raw_items
    ]


def convert_metadata_file(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        records = []
        for line in text.splitlines():
            if line.strip():
                records.extend(convert_metadata_payload(json.loads(line)))
        return records
    return convert_metadata_payload(json.loads(text))


def write_detection_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="Convert DeepStream metadata export to detection.v1 JSONL"
    )
    ap.add_argument("--input", required=True, help="DeepStream metadata JSON or JSONL")
    ap.add_argument("--out", required=True, help="Output detection.v1 JSONL path")
    args = ap.parse_args(argv)

    records = convert_metadata_file(args.input)
    write_detection_jsonl(args.out, records)
    print(f"written {args.out}: {len(records)} detection.v1 records")


if __name__ == "__main__":
    main(sys.argv[1:])
