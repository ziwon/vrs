"""Data contracts for the calibration loop."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class WindowEntry:
    """One verifier verdict recorded against a ``(stream_id, class)``."""
    ts_monotonic: float    # time.monotonic() when the verdict landed
    was_flipped: bool      # verifier said false_alarm
    had_fn_flag: bool      # verifier reported a false-negative class


@dataclass
class Suggestion:
    """One calibration suggestion written to the JSONL sink.

    Stage A: advisory only — the ``current_min_score`` is what YOLOE is
    using today, ``suggested_min_score`` is what the calibrator thinks it
    should become. An operator (or a future Stage-B loop) applies it.
    """
    ts: str                           # ISO-8601 wall clock
    stream_id: str
    class_name: str
    current_min_score: float
    suggested_min_score: float
    direction: str                    # "tighten" | "loosen"
    reason: str
    flip_rate: float
    fn_flag_rate: float
    n_alerts: int
    alerts_per_hour: Optional[float]

    @property
    def delta(self) -> float:
        return self.suggested_min_score - self.current_min_score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "stream_id": self.stream_id,
            "class_name": self.class_name,
            "current_min_score": float(self.current_min_score),
            "suggested_min_score": float(self.suggested_min_score),
            "delta": float(self.delta),
            "direction": self.direction,
            "reason": self.reason,
            "flip_rate": float(self.flip_rate),
            "fn_flag_rate": float(self.fn_flag_rate),
            "n_alerts": int(self.n_alerts),
            "alerts_per_hour": (
                float(self.alerts_per_hour)
                if self.alerts_per_hour is not None else None
            ),
        }
