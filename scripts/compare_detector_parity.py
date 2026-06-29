"""Compare Python detector output with DeepStream/TensorRT detector output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vrs.eval.detector_parity import (
    compare_detector_outputs,
    load_detection_records,
    load_runtime_summary,
    write_parity_report,
)


def _load_class_mapping(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    path = Path(raw)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--class-map must be a JSON object or path to one")
    return {str(k): str(v) for k, v in payload.items()}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare canonical detection.v1 outputs from two detector runtimes"
    )
    ap.add_argument("--python-detections", required=True, help="JSON/JSONL detection.v1 records")
    ap.add_argument(
        "--candidate-detections",
        required=True,
        help="DeepStream/TensorRT JSON/JSONL detection.v1 records",
    )
    ap.add_argument("--out", default="runs/parity/detector_parity.json")
    ap.add_argument(
        "--class-map",
        help="JSON object or path mapping candidate runtime class names to canonical class names",
    )
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--python-runtime", help="Optional JSON runtime summary")
    ap.add_argument("--candidate-runtime", help="Optional JSON runtime summary")
    args = ap.parse_args()

    class_mapping = _load_class_mapping(args.class_map)
    report = compare_detector_outputs(
        python_records=load_detection_records(args.python_detections),
        candidate_records=load_detection_records(
            args.candidate_detections,
            class_mapping=class_mapping,
        ),
        class_mapping=class_mapping,
        iou_threshold=args.iou_threshold,
        python_runtime=load_runtime_summary(args.python_runtime),
        candidate_runtime=load_runtime_summary(args.candidate_runtime),
    )
    write_parity_report(args.out, report)
    totals = report["totals"]
    print(
        f"written {args.out}: matched={totals['matched']} "
        f"unmatched_python={totals['unmatched_python']} "
        f"unmatched_candidate={totals['unmatched_candidate']}"
    )


if __name__ == "__main__":
    main()
