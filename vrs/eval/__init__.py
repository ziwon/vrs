"""Evaluation harness.

Measures cascade quality against labeled datasets so every other tuning
knob (thresholds, prompts, verifier model, tracking) stops being guesswork.

Current scope:
  * Dataclasses for ground-truth events and eval results (``schemas``).
  * Per-class P/R/F1 + verifier-flip rate + FN-flag rate (``metrics``),
    including event-level, bbox-level, and image-level scoring.
  * Stable, versioned ``EvalReport`` JSON contract (``report``).
  * Harness that iterates a dataset → runs the cascade → scores
    (``harness``), with labeled-directory and D-Fire adapters under
    ``datasets``.
  * Regression gate that compares two reports and exits non-zero on F1
    drops beyond a tolerance (``ci``, run as ``python -m vrs.eval.ci``).

Upcoming:
  * ``datasets/le2i.py`` and other real public datasets.
"""

from __future__ import annotations

from .harness import BBoxScoringMode, EvalMode, HarnessResult, config_for_eval_mode, evaluate
from .metrics import (
    aggregate_scores,
    score_alerts_against_truth,
    score_detections_against_truth,
    score_image_level_against_truth,
)
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
    "BBoxScoringMode",
    "ClassMetrics",
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
    "aggregate_scores",
    "config_for_eval_mode",
    "evaluate",
    "score_alerts_against_truth",
    "score_detections_against_truth",
    "score_image_level_against_truth",
]
