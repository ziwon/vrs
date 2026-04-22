"""Tracker + tracked EventStateQueue behavior — no GPU / model deps."""
from __future__ import annotations

import numpy as np
import pytest

from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.schemas import Detection, Frame
from vrs.triage import (
    EventStateQueue,
    NullTracker,
    SimpleIoUTracker,
    build_tracker,
)
from vrs.triage.tracking import _iou


def _det(cls: str = "fire", score: float = 0.9,
         xyxy=(10.0, 10.0, 30.0, 30.0)) -> Detection:
    return Detection(class_name=cls, score=score, xyxy=xyxy)


def _frame(idx: int, pts: float) -> Frame:
    return Frame(index=idx, pts_s=pts, image=np.zeros((64, 64, 3), dtype=np.uint8))


def _policy(classes=("fire",), min_persist=1) -> WatchPolicy:
    return WatchPolicy([
        WatchItem(name=c, detector_prompts=[c], verifier_prompt=c,
                  severity="critical", min_score=0.3, min_persist_frames=min_persist)
        for c in classes
    ])


# ─── IoU ──────────────────────────────────────────────────────────────

def test_iou_identity():
    assert _iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0)) == 1.0


def test_iou_disjoint():
    assert _iou((0.0, 0.0, 5.0, 5.0), (10.0, 10.0, 15.0, 15.0)) == 0.0


def test_iou_partial_overlap():
    # 10×10 and a shifted 10×10 overlapping 5×5 = 25 area; union = 175
    iou = _iou((0.0, 0.0, 10.0, 10.0), (5.0, 5.0, 15.0, 15.0))
    assert iou == pytest.approx(25 / 175, rel=1e-6)


# ─── NullTracker ──────────────────────────────────────────────────────

def test_null_tracker_leaves_track_id_none():
    dets = [_det(), _det(xyxy=(50.0, 50.0, 70.0, 70.0))]
    out = NullTracker().update(dets, frame_index=0)
    assert [d.track_id for d in out] == [None, None]


# ─── SimpleIoUTracker ─────────────────────────────────────────────────

def test_tracker_assigns_fresh_ids_to_new_objects():
    t = SimpleIoUTracker()
    out = t.update([_det(xyxy=(10, 10, 30, 30)), _det(xyxy=(100, 100, 120, 120))], 0)
    ids = [d.track_id for d in out]
    assert set(ids) == {1, 2}
    assert all(i is not None for i in ids)


def test_tracker_keeps_id_stable_across_frames_for_moving_bbox():
    """A gently-moving bbox should retain its track_id."""
    t = SimpleIoUTracker(iou_threshold=0.3)
    dets_per_frame = [
        [_det(xyxy=(10, 10, 30, 30))],
        [_det(xyxy=(12, 12, 32, 32))],    # significant overlap with prev
        [_det(xyxy=(14, 14, 34, 34))],
        [_det(xyxy=(16, 16, 36, 36))],
    ]
    ids = []
    for i, dets in enumerate(dets_per_frame):
        out = t.update(dets, i)
        ids.append(out[0].track_id)
    assert len(set(ids)) == 1
    assert ids[0] == 1


def test_tracker_gives_disjoint_boxes_different_ids():
    t = SimpleIoUTracker()
    t.update([_det(xyxy=(10, 10, 30, 30))], 0)
    # completely different location on frame 1 — no overlap with frame 0 track
    out = t.update([_det(xyxy=(100, 100, 130, 130))], 1)
    assert out[0].track_id == 2


def test_tracker_segregates_by_class():
    """A fire bbox and a smoke bbox with identical coords should not share
    a track_id — matching must happen within-class only."""
    t = SimpleIoUTracker()
    out = t.update([_det("fire", xyxy=(10, 10, 30, 30)),
                    _det("smoke", xyxy=(10, 10, 30, 30))], 0)
    ids = {d.class_name: d.track_id for d in out}
    assert ids["fire"] != ids["smoke"]


def test_tracker_expires_stale_tracks():
    t = SimpleIoUTracker(max_gap_frames=2)
    t.update([_det(xyxy=(10, 10, 30, 30))], 0)               # id=1 created
    t.update([], 1)
    t.update([], 2)
    t.update([], 3)                                          # 3 frames of gap > 2
    # a new overlapping detection should NOT recycle id=1, because id=1 expired
    out = t.update([_det(xyxy=(10, 10, 30, 30))], 4)
    assert out[0].track_id == 2


def test_tracker_rejects_bad_params():
    with pytest.raises(ValueError):
        SimpleIoUTracker(iou_threshold=0.0)
    with pytest.raises(ValueError):
        SimpleIoUTracker(iou_threshold=1.5)
    with pytest.raises(ValueError):
        SimpleIoUTracker(max_gap_frames=0)


# ─── build_tracker factory ────────────────────────────────────────────

def test_build_tracker_defaults_to_null():
    assert isinstance(build_tracker(None), NullTracker)
    assert isinstance(build_tracker({}), NullTracker)
    assert isinstance(build_tracker({"backend": "none"}), NullTracker)


def test_build_tracker_constructs_simple_iou():
    t = build_tracker({"backend": "simple_iou", "iou_threshold": 0.5, "max_gap_frames": 10})
    assert isinstance(t, SimpleIoUTracker)
    assert t.iou_threshold == 0.5
    assert t.max_gap_frames == 10


def test_build_tracker_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown tracker backend"):
        build_tracker({"backend": "byetetrack-typo"})


# ─── EventStateQueue with tracking ────────────────────────────────────

def test_event_state_untracked_behaves_as_before_one_alert_per_class():
    """Backward compat: when track_id is None on every detection, cooldown
    still debounces per class as it always did."""
    q = EventStateQueue(_policy(min_persist=1), window=4, cooldown_s=5.0, target_fps=4.0)
    a1 = q.step(_frame(0, 0.0), [_det()])    # fires
    a2 = q.step(_frame(1, 0.5), [_det()])    # cooldown — suppressed
    assert len(a1) == 1
    assert a1[0].track_id is None
    assert a2 == []


def test_event_state_two_tracks_fire_two_alerts_simultaneously():
    """Two simultaneous fires with different track ids → two alerts on the
    same frame. The old per-class cooldown would have suppressed one."""
    q = EventStateQueue(_policy(min_persist=1), window=4, cooldown_s=10.0, target_fps=4.0)
    d1 = _det(xyxy=(10, 10, 30, 30)); d1.track_id = 1
    d2 = _det(xyxy=(100, 100, 130, 130)); d2.track_id = 2
    alerts = q.step(_frame(0, 0.0), [d1, d2])
    assert len(alerts) == 2
    assert {a.track_id for a in alerts} == {1, 2}
    # peak_detections must be narrowed to the firing track
    for a in alerts:
        assert all(d.track_id == a.track_id for d in a.peak_detections)


def test_event_state_same_track_cooldown_suppresses_duplicates():
    q = EventStateQueue(_policy(min_persist=1), window=4, cooldown_s=2.0, target_fps=4.0)
    d = _det(); d.track_id = 1
    a1 = q.step(_frame(0, 0.0), [d])
    a2 = q.step(_frame(1, 0.5), [_tracked_copy(d)])   # same track, within cooldown → suppressed
    a3 = q.step(_frame(2, 2.5), [_tracked_copy(d)])   # past cooldown → fires again
    assert len(a1) == 1 and len(a2) == 0 and len(a3) == 1
    assert a1[0].track_id == 1 and a3[0].track_id == 1


def test_event_state_new_track_fires_even_during_other_tracks_cooldown():
    """Track id=1 fires, then shortly after id=2 appears — id=2 must fire
    immediately; the pre-tracking code would have suppressed it under the
    global per-class cooldown."""
    q = EventStateQueue(_policy(min_persist=1), window=4, cooldown_s=5.0, target_fps=4.0)
    d1 = _det(xyxy=(10, 10, 30, 30)); d1.track_id = 1
    d2 = _det(xyxy=(100, 100, 130, 130)); d2.track_id = 2

    a1 = q.step(_frame(0, 0.0), [d1])                   # id=1 fires
    a2 = q.step(_frame(1, 0.5), [d1, d2])               # id=2 fires despite id=1 cooldown
    assert len(a1) == 1 and a1[0].track_id == 1
    assert len(a2) == 1 and a2[0].track_id == 2


def _tracked_copy(d: Detection) -> Detection:
    """Fresh Detection with the same track_id — the event-state queue
    inspects ``track_id`` on new instances per frame."""
    return Detection(
        class_name=d.class_name, score=d.score, xyxy=d.xyxy,
        raw_label=d.raw_label, track_id=d.track_id,
    )
