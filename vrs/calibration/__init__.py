"""Closed-loop threshold self-calibration.

The verifier's ``true_alert`` / ``false_alarm`` verdict is a ground-truth-
like signal for every detector hit. Aggregated over a rolling window per
``(stream_id, class_name)``, it tells us whether the current YOLOE
``min_score`` is too low (flip rate creeping up) or too high (flip rate
near zero while the alert rate is below target).

Stage A ships a ``Calibrator`` that:

* Accumulates a rolling window of verdicts.
* Computes flip_rate + alerts/hour.
* Emits a *suggested* new ``min_score`` to
  ``runs/<name>/calibration_suggestions.jsonl``.

Stage B can promote suggestions into an operator-reviewable threshold export
with per-stream/class cooldowns and an append-only audit log. The running
detector policy remains immutable; operators can apply or roll back from the
exported snapshot deliberately.
"""

from __future__ import annotations

from .applier import CalibrationApplier
from .calibrator import Calibrator, build_calibrator
from .schemas import Suggestion, WindowEntry
from .sink import CalibrationSink
from .suggester import suggest

__all__ = [
    "CalibrationApplier",
    "CalibrationSink",
    "Calibrator",
    "Suggestion",
    "WindowEntry",
    "build_calibrator",
    "suggest",
]
