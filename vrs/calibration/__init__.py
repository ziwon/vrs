"""Closed-loop threshold self-calibration — Stage A (log-only).

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
* Never edits the policy. A human reviews and applies the suggestion.

Stage B (autonomous apply with caps + per-class cool-downs) is future
work. The same stateless ``suggest()`` function will drive both stages, so
an operator can move classes off "log only" per-class when they trust the
loop.
"""

from __future__ import annotations

from .calibrator import Calibrator, build_calibrator
from .schemas import Suggestion, WindowEntry
from .sink import CalibrationSink
from .suggester import suggest

__all__ = [
    "CalibrationSink",
    "Calibrator",
    "Suggestion",
    "WindowEntry",
    "build_calibrator",
    "suggest",
]
