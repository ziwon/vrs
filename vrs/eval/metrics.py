"""Scoring: match alerts.jsonl records against ground-truth events.

Matching is temporal and per-class. An alert at ``peak_pts_s`` for class X
matches a ground-truth event for class X when::

    event.start_s - tolerance_s  <=  peak_pts_s  <=  event.end_s + tolerance_s

We do one-to-one greedy matching so a single event can't be credited to two
alerts (which would overstate recall when the cooldown misfires). An alert
that finds no event → FP. An event with no matching alert → FN.

Only alerts with ``true_alert=True`` count as positive predictions: that is
the cascade's final output. Raw detector hits that the verifier flipped to
``false_alarm`` are *not* treated as FPs against ground truth — they are
captured separately as ``flip_rate``.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

from .schemas import ClassMetrics, GroundTruthEvent, RunScore


def score_alerts_against_truth(
    alerts: Sequence[dict],
    events: Sequence[GroundTruthEvent],
    *,
    tolerance_s: float = 1.0,
    classes: Iterable[str] | None = None,
) -> RunScore:
    """Compute per-class P/R/F1 + flip/FN rates for one run's alerts.

    Args:
        alerts: iterable of decoded ``alerts.jsonl`` lines. Each must have
            ``class_name``, ``peak_pts_s``, ``true_alert``. ``false_negative_class``
            is read if present.
        events: ground-truth events for the same video.
        tolerance_s: event windows are extended by this much on both sides
            before matching. Absorbs the ~cooldown-window jitter between the
            event's physical onset and when the cascade promotes it.
        classes: restrict scoring to these class names. Defaults to the union
            of classes that appear in alerts or events.
    """
    if tolerance_s < 0:
        raise ValueError("tolerance_s must be >= 0")

    if classes is None:
        classes = {a["class_name"] for a in alerts} | {e.class_name for e in events}
    class_set = set(classes)

    score = RunScore()

    # Totals that don't require class membership
    for a in alerts:
        score.n_alerts_total += 1
        if a.get("true_alert", True):
            score.n_alerts_true += 1
        if a.get("false_negative_class"):
            score.n_fn_flagged += 1

    score.n_events = sum(1 for e in events if e.class_name in class_set)

    # Per-class greedy match
    for cls in class_set:
        cm = ClassMetrics()
        cls_events: List[GroundTruthEvent] = [e for e in events if e.class_name == cls]
        matched = [False] * len(cls_events)

        cls_alerts = [
            a for a in alerts
            if a["class_name"] == cls and a.get("true_alert", True)
        ]
        # Stable ordering by peak_pts_s keeps matching deterministic.
        for alert in sorted(cls_alerts, key=lambda a: float(a["peak_pts_s"])):
            pts = float(alert["peak_pts_s"])
            hit_idx = None
            for i, ev in enumerate(cls_events):
                if matched[i]:
                    continue
                if ev.start_s - tolerance_s <= pts <= ev.end_s + tolerance_s:
                    hit_idx = i
                    break
            if hit_idx is None:
                cm.fp += 1
            else:
                cm.tp += 1
                matched[hit_idx] = True

        cm.fn = matched.count(False)
        score.per_class[cls] = cm

    return score


def aggregate_scores(scores: Iterable[RunScore]) -> RunScore:
    """Sum per-class counts and totals across many per-video ``RunScore`` objects.

    Ratios (precision/recall/f1/flip_rate) are always recomputed from the
    summed counts, never averaged across runs — macro-averaging would hide
    class imbalance.
    """
    agg = RunScore()
    for s in scores:
        agg.n_alerts_total += s.n_alerts_total
        agg.n_alerts_true += s.n_alerts_true
        agg.n_fn_flagged += s.n_fn_flagged
        agg.n_events += s.n_events
        for cls, cm in s.per_class.items():
            bucket = agg.per_class.setdefault(cls, ClassMetrics())
            bucket.tp += cm.tp
            bucket.fp += cm.fp
            bucket.fn += cm.fn
    return agg
