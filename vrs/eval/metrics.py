"""Scoring: match alerts.jsonl records against ground truth.

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

Image datasets can also use normalized bbox matching. In that mode, each
positive detector output is matched to at most one ground-truth box of the same
class when IoU is above the requested threshold.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

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
        cls_events: list[GroundTruthEvent] = [e for e in events if e.class_name == cls]
        matched = [False] * len(cls_events)

        cls_alerts = [a for a in alerts if a["class_name"] == cls and a.get("true_alert", True)]
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


def score_detections_against_truth(
    alerts: Sequence[dict],
    events: Sequence[GroundTruthEvent],
    *,
    image_size: tuple[int, int] | None = None,
    iou_threshold: float = 0.5,
    classes: Iterable[str] | None = None,
) -> RunScore:
    """Compute detector P/R/F1 with one-to-one bbox IoU matching.

    Args:
        alerts: decoded ``alerts.jsonl`` lines. Detector-only VRS alerts expose
            boxes either as ``bbox_xywh_norm`` or as ``peak_detections[].xyxy``.
        events: ground-truth objects for the same image/source. Events without
            ``bbox_xywh_norm`` are ignored by this scorer.
        image_size: ``(width, height)`` used to normalize ``xyxy`` detector
            boxes from ``peak_detections``.
        iou_threshold: minimum same-class IoU for a TP.
        classes: restrict scoring to these class names. Defaults to the union
            of classes that appear in predictions or ground truth.
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")

    predictions = list(_iter_detection_predictions(alerts, image_size=image_size))
    gt = [e for e in events if e.bbox_xywh_norm is not None]

    if classes is None:
        classes = {p[0] for p in predictions} | {e.class_name for e in gt}
    class_set = set(classes)

    score = RunScore()
    for a in alerts:
        score.n_alerts_total += 1
        if a.get("true_alert", True):
            score.n_alerts_true += 1
        if a.get("false_negative_class"):
            score.n_fn_flagged += 1

    score.n_events = sum(1 for e in gt if e.class_name in class_set)

    for cls in class_set:
        cm = ClassMetrics()
        cls_events = [e for e in gt if e.class_name == cls]
        matched = [False] * len(cls_events)

        cls_predictions = [p for p in predictions if p[0] == cls]
        for _, pred_box in sorted(cls_predictions, key=lambda p: p[1]):
            hit_idx = None
            hit_iou = iou_threshold
            for i, ev in enumerate(cls_events):
                if matched[i] or ev.bbox_xywh_norm is None:
                    continue
                iou = _iou_xywh_norm(pred_box, ev.bbox_xywh_norm)
                if iou >= hit_iou:
                    hit_idx = i
                    hit_iou = iou
            if hit_idx is None:
                cm.fp += 1
            else:
                cm.tp += 1
                matched[hit_idx] = True

        cm.fn = matched.count(False)
        score.per_class[cls] = cm

    return score


def score_image_level_against_truth(
    alerts: Sequence[dict],
    events: Sequence[GroundTruthEvent],
    *,
    classes: Iterable[str] | None = None,
) -> RunScore:
    """Compute image-level class presence metrics from bbox-labeled sources.

    Multiple boxes or detections of the same class in one image count as one
    class-present decision. This is useful for quick detector smoke tests where
    localization quality is not being scored.
    """
    predicted = {a["class_name"] for a in alerts if a.get("true_alert", True)}
    actual = {e.class_name for e in events}

    if classes is None:
        classes = predicted | actual
    class_set = set(classes)

    score = RunScore()
    for a in alerts:
        score.n_alerts_total += 1
        if a.get("true_alert", True):
            score.n_alerts_true += 1
        if a.get("false_negative_class"):
            score.n_fn_flagged += 1
    score.n_events = sum(1 for cls in class_set if cls in actual)

    for cls in class_set:
        cm = ClassMetrics()
        has_pred = cls in predicted
        has_truth = cls in actual
        if has_pred and has_truth:
            cm.tp = 1
        elif has_pred:
            cm.fp = 1
        elif has_truth:
            cm.fn = 1
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


def _iter_detection_predictions(
    alerts: Sequence[dict],
    *,
    image_size: tuple[int, int] | None,
) -> Iterable[tuple[str, tuple[float, float, float, float]]]:
    for alert in alerts:
        if not alert.get("true_alert", True):
            continue

        class_name = str(alert["class_name"])
        bbox = alert.get("bbox_xywh_norm")
        if bbox is not None:
            yield class_name, _coerce_xywh_norm(bbox)
            continue

        for detection in alert.get("peak_detections", []) or []:
            det_class = str(detection.get("class_name", class_name))
            det_bbox = detection.get("bbox_xywh_norm")
            if det_bbox is not None:
                yield det_class, _coerce_xywh_norm(det_bbox)
                continue

            xyxy = detection.get("xyxy")
            if xyxy is not None:
                if image_size is None:
                    raise ValueError("image_size is required to score peak_detections xyxy boxes")
                yield det_class, _xyxy_pixels_to_xywh_norm(xyxy, image_size)


def _coerce_xywh_norm(raw: Sequence[float]) -> tuple[float, float, float, float]:
    if len(raw) != 4:
        raise ValueError(f"expected 4 bbox coordinates, got {len(raw)}")
    x, y, w, h = (float(v) for v in raw)
    return (x, y, w, h)


def _xyxy_pixels_to_xywh_norm(
    raw: Sequence[float], image_size: tuple[int, int]
) -> tuple[float, float, float, float]:
    if len(raw) != 4:
        raise ValueError(f"expected 4 bbox coordinates, got {len(raw)}")
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image_size must contain positive width and height")

    x1, y1, x2, y2 = (float(v) for v in raw)
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        max(0.0, x2 - x1) / width,
        max(0.0, y2 - y1) / height,
    )


def _iou_xywh_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    half_w = w / 2.0
    half_h = h / 2.0
    return (x - half_w, y - half_h, x + half_w, y + half_h)
