"""Compare detector JSONL when runtimes use different frame rates or coordinates."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vrs.eval.detector_parity import bbox_iou_xyxy
from vrs.policy import load_watch_policy


@dataclass(frozen=True)
class Record:
    class_name: str
    raw_label: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_index: int | None
    pts_s: float | None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compare detection.v1 outputs by event, timestamp tolerance, IoU, and optional "
            "candidate bbox scaling. Useful for Python-vs-DeepStream parity where one side "
            "runs every frame in muxer coordinates."
        )
    )
    ap.add_argument("--python-detections", required=True)
    ap.add_argument("--candidate-detections", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="configs/policies/safety.yaml")
    ap.add_argument("--time-tolerance-s", type=float, default=0.06)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--candidate-scale-x", type=float, default=1.0)
    ap.add_argument("--candidate-scale-y", type=float, default=1.0)
    ap.add_argument("--candidate-offset-x", type=float, default=0.0)
    ap.add_argument("--candidate-offset-y", type=float, default=0.0)
    args = ap.parse_args()

    policy = load_watch_policy(args.policy)
    prompt_to_event = {
        prompt: policy.event_for_prompt_index(i)
        for i, prompt in enumerate(policy.yoloe_vocabulary())
    }

    python_records = load_records(args.python_detections)
    candidate_records = [
        transform_record(
            map_record(det, prompt_to_event),
            args.candidate_scale_x,
            args.candidate_scale_y,
            args.candidate_offset_x,
            args.candidate_offset_y,
        )
        for det in load_records(args.candidate_detections)
    ]
    report = compare(
        python_records,
        candidate_records,
        time_tolerance_s=args.time_tolerance_s,
        iou_threshold=args.iou_threshold,
    )
    report["class_mapping"] = prompt_to_event
    report["candidate_transform"] = {
        "scale_x": args.candidate_scale_x,
        "scale_y": args.candidate_scale_y,
        "offset_x": args.candidate_offset_x,
        "offset_y": args.candidate_offset_y,
    }
    report["time_tolerance_s"] = args.time_tolerance_s
    report["iou_threshold"] = args.iou_threshold

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    totals = report["totals"]
    print(
        f"written {out}: matched={totals['matched']} "
        f"unmatched_python={totals['unmatched_python']} "
        f"unmatched_candidate={totals['unmatched_candidate']}"
    )


def load_records(path: str | Path) -> list[Record]:
    rows: list[Record] = []
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return rows
    for item in _payloads(text, Path(path).suffix):
        bbox = item["bbox_xyxy"]
        rows.append(
            Record(
                class_name=str(item["class_name"]),
                raw_label=str(item.get("raw_label") or item["class_name"]),
                score=float(item["score"]),
                bbox_xyxy=tuple(float(v) for v in bbox),
                frame_index=_optional_int(item.get("frame_index")),
                pts_s=_optional_float(item.get("pts_s")),
            )
        )
    return rows


def _payloads(text: str, suffix: str) -> Iterable[dict[str, Any]]:
    if suffix == ".jsonl":
        for line in text.splitlines():
            if line.strip():
                yield json.loads(line)
        return
    payload = json.loads(text)
    if isinstance(payload, list):
        yield from payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        yield from payload["records"]
    elif isinstance(payload, dict) and isinstance(payload.get("detections"), list):
        yield from payload["detections"]
    elif isinstance(payload, dict):
        yield payload
    else:
        raise ValueError("expected JSON object, array, or JSONL detections")


def map_record(record: Record, prompt_to_event: dict[str, str]) -> Record:
    class_name = prompt_to_event.get(record.class_name, record.class_name)
    return Record(
        class_name=class_name,
        raw_label=record.raw_label,
        score=record.score,
        bbox_xyxy=record.bbox_xyxy,
        frame_index=record.frame_index,
        pts_s=record.pts_s,
    )


def transform_record(record: Record, sx: float, sy: float, ox: float, oy: float) -> Record:
    x1, y1, x2, y2 = record.bbox_xyxy
    return Record(
        class_name=record.class_name,
        raw_label=record.raw_label,
        score=record.score,
        bbox_xyxy=((x1 + ox) * sx, (y1 + oy) * sy, (x2 + ox) * sx, (y2 + oy) * sy),
        frame_index=record.frame_index,
        pts_s=record.pts_s,
    )


def compare(
    python_records: list[Record],
    candidate_records: list[Record],
    *,
    time_tolerance_s: float,
    iou_threshold: float,
) -> dict[str, Any]:
    candidate_by_class: dict[str, list[Record]] = defaultdict(list)
    for det in candidate_records:
        candidate_by_class[det.class_name].append(det)

    matches: list[dict[str, Any]] = []
    unmatched_python: list[Record] = []
    used: set[tuple[str, int]] = set()

    for py_det in python_records:
        best: tuple[float, float, int, Record] | None = None
        for idx, cand in enumerate(candidate_by_class.get(py_det.class_name, [])):
            key = (py_det.class_name, idx)
            if key in used:
                continue
            if py_det.pts_s is None or cand.pts_s is None:
                continue
            dt = abs(py_det.pts_s - cand.pts_s)
            if dt > time_tolerance_s:
                continue
            iou = bbox_iou_xyxy(py_det.bbox_xyxy, cand.bbox_xyxy)
            if iou < iou_threshold:
                continue
            score = (iou, -dt)
            if best is None or score > (best[0], best[1]):
                best = (iou, -dt, idx, cand)
        if best is None:
            unmatched_python.append(py_det)
            continue
        iou, neg_dt, idx, cand = best
        used.add((py_det.class_name, idx))
        matches.append(
            {
                "class_name": py_det.class_name,
                "python": _record_payload(py_det),
                "candidate": _record_payload(cand),
                "iou": iou,
                "dt_s": -neg_dt,
                "score_delta": cand.score - py_det.score,
            }
        )

    unmatched_candidate: list[Record] = []
    for class_name, records in candidate_by_class.items():
        for idx, det in enumerate(records):
            if (class_name, idx) not in used:
                unmatched_candidate.append(det)

    return {
        "schema_version": "vrs.eval.aligned_detector_parity.v1",
        "totals": {
            "python_count": len(python_records),
            "candidate_count": len(candidate_records),
            "matched": len(matches),
            "unmatched_python": len(unmatched_python),
            "unmatched_candidate": len(unmatched_candidate),
        },
        "python_by_class": dict(Counter(r.class_name for r in python_records)),
        "candidate_by_class": dict(Counter(r.class_name for r in candidate_records)),
        "matched_by_class": dict(Counter(m["class_name"] for m in matches)),
        "bbox": _summary([m["iou"] for m in matches]),
        "time_delta_s": _summary([m["dt_s"] for m in matches]),
        "confidence_delta": _summary([m["score_delta"] for m in matches]),
        "matches": matches,
    }


def _record_payload(record: Record) -> dict[str, Any]:
    return {
        "class_name": record.class_name,
        "raw_label": record.raw_label,
        "score": record.score,
        "bbox_xyxy": list(record.bbox_xyxy),
        "frame_index": record.frame_index,
        "pts_s": record.pts_s,
    }


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "median": None, "max": None}
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


if __name__ == "__main__":
    main()
