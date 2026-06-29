"""Evaluation harness.

Measures cascade quality against labeled datasets so every other tuning
knob (thresholds, prompts, verifier model, tracking) stops being guesswork.

Current scope:
  * Dataclasses for ground-truth events and eval results (``schemas``).
  * Per-class P/R/F1 + verifier-flip rate + FN-flag rate (``metrics``).
  * Stable, versioned ``EvalReport`` JSON contract (``report``).
  * Harness that iterates a dataset → runs the cascade → scores
    (``harness``), with labeled-directory, D-Fire, Le2i, and UCF-Crime/UCA
    adapters under ``datasets``.
  * Regression gate that compares two reports and exits non-zero on F1
    drops beyond a tolerance (``ci``, run as ``python -m vrs.eval.ci``).
  * Detector runtime parity reports for Python versus DeepStream/TensorRT
    canonical detection outputs (``detector_parity``).

Upcoming:
  * UP-Fall multimodal dataset coverage if needed for non-RGB fall validation.
"""

from __future__ import annotations

from .detection_export import detections_to_contracts, write_detection_jsonl
from .detector_parity import (
    DetectionRecord,
    RuntimeSummary,
    compare_detector_outputs,
    load_detection_records,
    write_parity_report,
)
from .harness import (
    EvalMode,
    HarnessResult,
    config_for_eval_mode,
    dataset_items_are_images,
    evaluate,
    evaluate_detector_only_images,
)
from .metrics import aggregate_scores, bbox_iou_xywh_norm, score_alerts_against_truth
from .report import (
    SCHEMA_VERSION,
    EvalReport,
    PerVideoReport,
    ReportClassMetrics,
    ReportLatency,
    ReportMetrics,
    ReportModel,
    ReportModels,
    ReportQualitySignals,
    ReportRun,
    ReportRuntime,
)
from .schemas import ClassMetrics, EvalItem, GroundTruthEvent, RunScore

# Note: ``ci`` is intentionally not re-exported here — it is a tool module
# with a ``__main__`` entry point, so ``python -m vrs.eval.ci`` works
# without triggering a runpy double-import warning. Use
# ``from vrs.eval.ci import compare_reports`` if you need the function.

__all__ = [
    "SCHEMA_VERSION",
    "ClassMetrics",
    "DetectionRecord",
    "EvalItem",
    "EvalMode",
    "EvalReport",
    "GroundTruthEvent",
    "HarnessResult",
    "PerVideoReport",
    "ReportClassMetrics",
    "ReportLatency",
    "ReportMetrics",
    "ReportModel",
    "ReportModels",
    "ReportQualitySignals",
    "ReportRun",
    "ReportRuntime",
    "RunScore",
    "RuntimeSummary",
    "aggregate_scores",
    "bbox_iou_xywh_norm",
    "compare_detector_outputs",
    "config_for_eval_mode",
    "dataset_items_are_images",
    "detections_to_contracts",
    "evaluate",
    "evaluate_detector_only_images",
    "load_detection_records",
    "score_alerts_against_truth",
    "write_detection_jsonl",
    "write_parity_report",
]
