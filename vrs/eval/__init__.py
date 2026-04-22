"""Evaluation harness.

Measures cascade quality against labeled datasets so every other tuning
knob (thresholds, prompts, verifier model, tracking) stops being guesswork.

Current scope (Step 1.A):
  * Dataclasses for ground-truth events and eval results (``schemas``).
  * Per-class P/R/F1 + verifier-flip rate + FN-flag rate (``metrics``).
  * One concrete dataset adapter: a directory of mp4 files with sidecar
    JSON labels (``datasets.labeled_dir``).

Upcoming:
  * ``harness.py``  — iterate dataset → run cascade → score.
  * ``datasets/dfire.py`` and ``datasets/le2i.py`` — real public datasets.
  * ``ci.py``       — regression gate: fail on F1 delta < -0.02.
"""
from __future__ import annotations

from .harness import HarnessResult, evaluate
from .metrics import aggregate_scores, score_alerts_against_truth
from .schemas import ClassMetrics, EvalItem, GroundTruthEvent, RunScore

# Note: ``ci`` is intentionally not re-exported here — it is a tool module
# with a ``__main__`` entry point, so ``python -m vrs.eval.ci`` works
# without triggering a runpy double-import warning. Use
# ``from vrs.eval.ci import compare_reports`` if you need the function.

__all__ = [
    "ClassMetrics",
    "EvalItem",
    "GroundTruthEvent",
    "HarnessResult",
    "RunScore",
    "aggregate_scores",
    "evaluate",
    "score_alerts_against_truth",
]
