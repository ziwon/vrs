"""Detector parity reports for Python and DeepStream/TensorRT outputs."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "vrs.eval.detector_parity.v1"


@dataclass(frozen=True)
class DetectionRecord:
    class_name: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_index: int | None = None
    stream_id: str | None = None
    clip_id: str | None = None
    pts_s: float | None = None
    source_runtime: str | None = None

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        class_mapping: Mapping[str, str] | None = None,
    ) -> DetectionRecord:
        bbox = data.get("bbox_xyxy")
        if bbox is None:
            bbox = data.get("xyxy")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError("detection record must contain bbox_xyxy with four values")
        class_name = str(data["class_name"])
        if class_mapping:
            class_name = class_mapping.get(class_name, class_name)
        return cls(
            class_name=class_name,
            score=float(data["score"]),
            bbox_xyxy=tuple(float(x) for x in bbox),
            frame_index=_optional_int(data.get("frame_index")),
            stream_id=_optional_str(data.get("stream_id")),
            clip_id=_optional_str(data.get("clip_id") or data.get("video_id")),
            pts_s=_optional_float(data.get("pts_s")),
            source_runtime=_optional_str(data.get("source_runtime")),
        )

    @property
    def group_key(self) -> tuple[str | None, str | None, int | None, str]:
        return (self.clip_id, self.stream_id, self.frame_index, self.class_name)


@dataclass(frozen=True)
class RuntimeSummary:
    latency_ms: dict[str, Any] = field(default_factory=dict)
    throughput_fps: float | None = None
    queue_drops: int | None = None
    gpu_memory: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> RuntimeSummary:
        if not data:
            return cls()
        queue_drops = data.get("queue_drops")
        return cls(
            latency_ms=dict(data.get("latency_ms") or {}),
            throughput_fps=_optional_float(data.get("throughput_fps")),
            queue_drops=int(queue_drops) if queue_drops is not None else None,
            gpu_memory=dict(data.get("gpu_memory") or data.get("vram_gb") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "throughput_fps": self.throughput_fps,
            "queue_drops": self.queue_drops,
            "gpu_memory": self.gpu_memory,
        }


def load_detection_records(
    path: str | Path,
    *,
    class_mapping: Mapping[str, str] | None = None,
) -> list[DetectionRecord]:
    """Load detection.v1 records from JSON array, JSON object, or JSONL."""

    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        payloads = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        if isinstance(payload, list):
            payloads = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            payloads = payload["records"]
        elif isinstance(payload, dict) and isinstance(payload.get("detections"), list):
            payloads = payload["detections"]
        elif isinstance(payload, dict):
            payloads = [payload]
        else:
            raise ValueError(f"{path}: expected JSON object, array, or JSONL detections")
    return [DetectionRecord.from_mapping(item, class_mapping=class_mapping) for item in payloads]


def compare_detector_outputs(
    *,
    python_records: Sequence[DetectionRecord],
    candidate_records: Sequence[DetectionRecord],
    class_mapping: Mapping[str, str] | None = None,
    iou_threshold: float = 0.5,
    python_runtime: RuntimeSummary | None = None,
    candidate_runtime: RuntimeSummary | None = None,
) -> dict[str, Any]:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    candidate_by_key: dict[
        tuple[str | None, str | None, int | None, str], list[DetectionRecord]
    ] = defaultdict(list)
    for det in candidate_records:
        candidate_by_key[det.group_key].append(det)

    matches: list[dict[str, Any]] = []
    unmatched_python: list[DetectionRecord] = []
    used_candidate: set[tuple[tuple[str | None, str | None, int | None, str], int]] = set()

    for py_det in python_records:
        candidates = candidate_by_key.get(py_det.group_key, [])
        best_idx = None
        best_iou = -1.0
        for idx, cand_det in enumerate(candidates):
            used_key = (py_det.group_key, idx)
            if used_key in used_candidate:
                continue
            iou = bbox_iou_xyxy(py_det.bbox_xyxy, cand_det.bbox_xyxy)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is None or best_iou < iou_threshold:
            unmatched_python.append(py_det)
            continue
        used_candidate.add((py_det.group_key, best_idx))
        cand_det = candidates[best_idx]
        matches.append(_match_payload(py_det, cand_det, best_iou))

    unmatched_candidate = []
    for key, candidates in candidate_by_key.items():
        for idx, cand_det in enumerate(candidates):
            if (key, idx) not in used_candidate:
                unmatched_candidate.append(cand_det)

    return {
        "schema_version": SCHEMA_VERSION,
        "iou_threshold": iou_threshold,
        "class_mapping": dict(class_mapping or {}),
        "totals": {
            "python_count": len(python_records),
            "candidate_count": len(candidate_records),
            "matched": len(matches),
            "unmatched_python": len(unmatched_python),
            "unmatched_candidate": len(unmatched_candidate),
        },
        "bbox": _summarize_bbox(matches),
        "confidence": _summarize_confidence(matches),
        "per_class": _summarize_per_class(matches, unmatched_python, unmatched_candidate),
        "runtime": {
            "python": (python_runtime or RuntimeSummary()).to_dict(),
            "candidate": (candidate_runtime or RuntimeSummary()).to_dict(),
        },
        "matches": matches,
    }


def bbox_iou_xyxy(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_runtime_summary(path: str | Path | None) -> RuntimeSummary:
    if path is None:
        return RuntimeSummary()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected runtime summary object")
    return RuntimeSummary.from_mapping(payload)


def write_parity_report(path: str | Path, report: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _match_payload(
    python_det: DetectionRecord,
    candidate_det: DetectionRecord,
    iou: float,
) -> dict[str, Any]:
    bbox_delta = [float(candidate_det.bbox_xyxy[i] - python_det.bbox_xyxy[i]) for i in range(4)]
    return {
        "clip_id": python_det.clip_id,
        "stream_id": python_det.stream_id,
        "frame_index": python_det.frame_index,
        "class_name": python_det.class_name,
        "iou": round(iou, 6),
        "python_score": python_det.score,
        "candidate_score": candidate_det.score,
        "confidence_delta": candidate_det.score - python_det.score,
        "bbox_delta_xyxy": bbox_delta,
        "bbox_abs_delta_mean_px": statistics.fmean(abs(x) for x in bbox_delta),
    }


def _summarize_bbox(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ious = [float(m["iou"]) for m in matches]
    deltas = [float(m["bbox_abs_delta_mean_px"]) for m in matches]
    return {
        "mean_iou": _mean(ious),
        "p50_iou": _percentile(ious, 0.50),
        "p95_abs_delta_px": _percentile(deltas, 0.95),
        "mean_abs_delta_px": _mean(deltas),
    }


def _summarize_confidence(matches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deltas = [float(m["confidence_delta"]) for m in matches]
    abs_deltas = [abs(x) for x in deltas]
    return {
        "mean_delta": _mean(deltas),
        "mean_abs_delta": _mean(abs_deltas),
        "p95_abs_delta": _percentile(abs_deltas, 0.95),
    }


def _summarize_per_class(
    matches: Sequence[Mapping[str, Any]],
    unmatched_python: Sequence[DetectionRecord],
    unmatched_candidate: Sequence[DetectionRecord],
) -> dict[str, Any]:
    classes = {
        *(str(m["class_name"]) for m in matches),
        *(d.class_name for d in unmatched_python),
        *(d.class_name for d in unmatched_candidate),
    }
    out = {}
    for cls in sorted(classes):
        cls_matches = [m for m in matches if m["class_name"] == cls]
        cls_py_unmatched = [d for d in unmatched_python if d.class_name == cls]
        cls_cand_unmatched = [d for d in unmatched_candidate if d.class_name == cls]
        out[cls] = {
            "matched": len(cls_matches),
            "unmatched_python": len(cls_py_unmatched),
            "unmatched_candidate": len(cls_cand_unmatched),
            "mean_iou": _mean([float(m["iou"]) for m in cls_matches]),
            "mean_abs_confidence_delta": _mean(
                [abs(float(m["confidence_delta"])) for m in cls_matches]
            ),
        }
    return out


def _mean(values: Sequence[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(float(ordered[0]), 6)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return round(float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac), 6)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None


def _optional_str(value: Any) -> str | None:
    return str(value) if value is not None else None
