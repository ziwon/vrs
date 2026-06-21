"""Evaluation harness — iterate dataset → run cascade → score.

Pipeline-agnostic: the caller supplies a ``pipeline_factory`` that maps a
per-video output directory to something with ``.run(source_path)``. Production
code passes a closure over ``vrs.pipeline.build_pipeline``; tests inject a
stub so scoring can be exercised without GPUs or video decode.

After each run the harness reads ``<video_out>/alerts.jsonl`` and scores it
against the item's ground-truth events. The final ``HarnessResult`` holds
both the per-video breakdown (useful for spotting which clips drag metrics
down) and the aggregate (headline numbers).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..schemas import Detection, Frame
from ..triage.backends import Detector
from .datasets.base import Dataset
from .metrics import aggregate_scores, score_alerts_against_truth
from .schemas import RunScore

logger = logging.getLogger(__name__)


PipelineFactory = Callable[[Path], Any]  # (out_dir) -> something with .run(source)
EvalMode = Literal["full_cascade", "detector_only"]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass
class HarnessResult:
    aggregate: RunScore
    per_video: list[tuple[Path, RunScore]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "aggregate": self.aggregate.to_dict(),
            "per_video": [{"video": str(p), "score": s.to_dict()} for p, s in self.per_video],
        }


def config_for_eval_mode(config: dict[str, Any], mode: EvalMode) -> dict[str, Any]:
    """Return a copy of ``config`` adjusted for the requested eval mode."""
    if mode not in ("full_cascade", "detector_only"):
        raise ValueError(f"unknown eval mode: {mode!r}")

    cfg = deepcopy(config)
    if mode == "detector_only":
        verifier = cfg.setdefault("verifier", {})
        verifier["enabled"] = False
    return cfg


def dataset_items_are_images(dataset: Dataset) -> bool:
    """Return true when every item points at a still-image source.

    This lets CLI adapters other than D-Fire use the direct detector-only image
    path without hard-coding adapter names. Empty datasets are treated as image
    compatible because there is no media item that requires video decoding.
    """
    return all(item.video_path.suffix.lower() in IMAGE_SUFFIXES for item in dataset)


def evaluate(
    dataset: Dataset,
    pipeline_factory: PipelineFactory,
    out_dir: str | Path,
    *,
    tolerance_s: float = 1.0,
    bbox_iou_threshold: float | None = None,
    alerts_filename: str = "alerts.jsonl",
    classes: Iterable[str] | None = None,
) -> HarnessResult:
    """Run the cascade over every item in ``dataset`` and return scored results.

    Args:
        dataset: anything that iterates ``EvalItem`` (e.g. ``LabeledDirDataset``).
        pipeline_factory: ``out_dir -> pipeline``. The harness calls
            ``pipeline.run(str(item.video_path))`` for each item.
        out_dir: parent dir; each item gets its own subdir keyed by
            ``video_path.stem``.
        tolerance_s: temporal slack around ground-truth event windows.
        alerts_filename: name of the JSONL the pipeline writes in its out dir.
        classes: restrict scoring to these class names (``None`` = union of
            classes seen in alerts / events).
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    classes_set = set(classes) if classes is not None else None
    per_video: list[tuple[Path, RunScore]] = []

    for item in dataset:
        video_out = root / item.video_path.stem
        video_out.mkdir(parents=True, exist_ok=True)

        pipeline = pipeline_factory(video_out)
        try:
            pipeline.run(str(item.video_path))
        except Exception as e:
            logger.error("pipeline run failed on %s: %s", item.video_path, e)

        alerts = _load_alerts(video_out / alerts_filename)
        score = score_alerts_against_truth(
            alerts,
            item.events,
            tolerance_s=tolerance_s,
            bbox_iou_threshold=bbox_iou_threshold,
            classes=classes_set,
        )
        score.detector_latencies_ms.extend(
            float(v) for v in getattr(pipeline, "detector_latencies_ms", [])
        )
        score.verifier_latencies_ms.extend(
            float(v) for v in getattr(pipeline, "verifier_latencies_ms", [])
        )
        score.verifier_tokens_per_second.extend(
            float(v) for v in getattr(pipeline, "verifier_tokens_per_second", [])
        )
        per_video.append((item.video_path, score))
        logger.info(
            "scored %s — alerts=%d events=%d per_class=%s flip_rate=%.3f",
            item.video_path.name,
            score.n_alerts_total,
            score.n_events,
            {k: v.to_dict() for k, v in score.per_class.items()},
            score.flip_rate,
        )

    return HarnessResult(
        aggregate=aggregate_scores(s for _, s in per_video),
        per_video=per_video,
    )


def evaluate_detector_only_images(
    dataset: Dataset,
    detector: Detector,
    out_dir: str | Path,
    *,
    bbox_iou_threshold: float | None = None,
    classes: Iterable[str] | None = None,
    alerts_filename: str = "alerts.jsonl",
) -> HarnessResult:
    """Evaluate image-labeled data by running a detector directly on images.

    This bypasses the full cascade (tracker, event-state queue, verifier, sinks)
    and is intended for datasets such as D-Fire where each item is a single
    image with YOLO-style box labels.
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)

    classes_set = set(classes) if classes is not None else None
    per_video: list[tuple[Path, RunScore]] = []

    for item in dataset:
        video_out = root / item.video_path.stem
        video_out.mkdir(parents=True, exist_ok=True)

        image = _load_image(item.video_path)
        frame = Frame(index=0, pts_s=0.0, image=image)

        detector_t0 = time.perf_counter()
        try:
            detections = detector(frame)
        except Exception as e:
            logger.error("detector run failed on %s: %s", item.video_path, e)
            detections = []
        detector_latency_ms = (time.perf_counter() - detector_t0) * 1000.0

        alerts = _detections_to_alerts(detections, image)
        _write_alerts(video_out / alerts_filename, alerts)
        score = score_alerts_against_truth(
            alerts,
            item.events,
            tolerance_s=0.0,
            bbox_iou_threshold=bbox_iou_threshold,
            classes=classes_set,
        )
        score.detector_latencies_ms.append(detector_latency_ms)
        per_video.append((item.video_path, score))
        logger.info(
            "scored %s — detections=%d alerts=%d events=%d detector_ms=%.1f per_class=%s flip_rate=%.3f",
            item.video_path.name,
            len(detections or []),
            score.n_alerts_total,
            score.n_events,
            detector_latency_ms,
            {k: v.to_dict() for k, v in score.per_class.items()},
            score.flip_rate,
        )

    return HarnessResult(
        aggregate=aggregate_scores(s for _, s in per_video),
        per_video=per_video,
    )


def _load_alerts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError as e:
                logger.warning("skipping malformed alert line in %s: %s", path, e)
    return out


def _load_image(path: Path):
    import cv2

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return image


def _detections_to_alerts(detections: list[Detection], image) -> list[dict]:
    height, width = image.shape[:2]
    out: list[dict] = []
    for det in sorted(
        detections or [],
        key=lambda d: (d.class_name, -float(d.score), tuple(float(v) for v in d.xyxy)),
    ):
        out.append(
            {
                "class_name": det.class_name,
                "peak_pts_s": 0.0,
                "true_alert": True,
                "false_negative_class": None,
                "confidence": float(det.score),
                "bbox_xywh_norm": list(_xyxy_to_norm(det.xyxy, width=width, height=height)),
                "raw_label": det.raw_label,
            }
        )
    return out


def _xyxy_to_norm(
    xyxy: tuple[float, float, float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = _clamp01(x1 / width)
    y1 = _clamp01(y1 / height)
    x2 = _clamp01(x2 / width)
    y2 = _clamp01(y2 / height)
    return (x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _write_alerts(path: Path, alerts: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for alert in alerts:
            f.write(json.dumps(alert) + "\n")
