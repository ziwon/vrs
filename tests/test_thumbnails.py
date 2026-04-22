from __future__ import annotations

import numpy as np

from vrs.schemas import CandidateAlert, Detection, VerifiedAlert
from vrs.sinks.thumbnail_sink import EventThumbnailSink


def _alert(with_keyframe: bool = True) -> VerifiedAlert:
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[20:44, 20:44] = 180
    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=0.0,
        peak_pts_s=1.25,
        peak_frame_index=5,
        peak_detections=[
            Detection(class_name="fire", score=0.92, xyxy=(20, 20, 44, 44), track_id=7),
        ],
        keyframes=[img] if with_keyframe else [],
        keyframe_pts=[1.25] if with_keyframe else [],
        track_id=7,
    )
    return VerifiedAlert(
        candidate=cand,
        true_alert=True,
        confidence=0.88,
        false_negative_class=None,
        rationale="test",
        bbox_xywh_norm=(0.3, 0.3, 0.4, 0.4),
    )


def test_thumbnail_sink_writes_event_image_and_sets_json_path(tmp_path):
    sink = EventThumbnailSink(tmp_path, quality=85)
    alert = _alert()

    rel = sink.write(alert)

    assert rel == alert.thumbnail_path
    assert rel is not None
    assert rel.startswith("thumbnails/")
    assert (tmp_path / rel).exists()
    assert alert.to_json()["thumbnail_path"] == rel


def test_thumbnail_sink_skips_alert_without_keyframes(tmp_path):
    sink = EventThumbnailSink(tmp_path)
    alert = _alert(with_keyframe=False)

    assert sink.write(alert) is None
    assert alert.thumbnail_path is None
