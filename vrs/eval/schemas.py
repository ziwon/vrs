"""Data contracts for the eval harness.

Ground truth is event-level (class name + time window in the source media).
Image datasets such as D-Fire use a degenerate ``0.0`` to ``0.0`` event window
and can optionally attach normalized YOLO boxes for bbox-level scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class GroundTruthEvent:
    """One labeled event in a source video or image.

    ``bbox_xywh_norm`` uses normalized VRS alert coordinates:
    ``(x_min, y_min, width, height)``.
    """

    class_name: str
    start_s: float
    end_s: float
    bbox_xywh_norm: tuple[float, float, float, float] | None = None


@dataclass
class EvalItem:
    """One media item + its ground-truth events (what a Dataset iterator yields)."""

    video_path: Path
    events: list[GroundTruthEvent] = field(default_factory=list)


@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


@dataclass
class RunScore:
    """Result of scoring one pipeline run against one dataset item (or the
    aggregate across many items — totals accumulate naturally)."""

    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    n_alerts_total: int = 0  # all alerts, regardless of verifier verdict
    n_alerts_true: int = 0  # alerts where verifier said true_alert=True
    n_fn_flagged: int = 0  # alerts where verifier reported a false negative
    n_events: int = 0  # ground-truth events counted

    @property
    def flip_rate(self) -> float:
        """Fraction of alerts the verifier flipped to false_alarm.

        This is the key self-signal the improvements doc names: healthy cascades
        stabilize around 5-15 %. Trending up -> detector drift; trending down ->
        verifier is rubber-stamping.
        """
        if not self.n_alerts_total:
            return 0.0
        return (self.n_alerts_total - self.n_alerts_true) / self.n_alerts_total

    @property
    def fn_flag_rate(self) -> float:
        if not self.n_alerts_total:
            return 0.0
        return self.n_fn_flagged / self.n_alerts_total

    def overall(self) -> ClassMetrics:
        agg = ClassMetrics()
        for cm in self.per_class.values():
            agg.tp += cm.tp
            agg.fp += cm.fp
            agg.fn += cm.fn
        return agg

    def to_dict(self) -> dict:
        return {
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "overall": self.overall().to_dict(),
            "n_alerts_total": self.n_alerts_total,
            "n_alerts_true": self.n_alerts_true,
            "n_fn_flagged": self.n_fn_flagged,
            "n_events": self.n_events,
            "flip_rate": round(self.flip_rate, 4),
            "fn_flag_rate": round(self.fn_flag_rate, 4),
        }
