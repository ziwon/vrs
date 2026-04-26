"""Self-calibration Stage A — pure-Python unit tests."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vrs.calibration import (
    CalibrationSink,
    Calibrator,
    Suggestion,
    WindowEntry,
    suggest,
)
from vrs.calibration.calibrator import build_calibrator
from vrs.policy.watch_policy import WatchItem, WatchPolicy
from vrs.schemas import CandidateAlert, Detection, VerifiedAlert


def _policy(min_score: float = 0.30) -> WatchPolicy:
    return WatchPolicy(
        [
            WatchItem(
                name="fire",
                detector_prompts=["fire"],
                verifier_prompt="flames",
                severity="critical",
                min_score=min_score,
                min_persist_frames=2,
            ),
        ]
    )


def _entry(dt_s: float = 0.0, flipped: bool = False, fn: bool = False) -> WindowEntry:
    """Window entry at a deterministic offset. Caller builds a ``List[WindowEntry]``
    whose timestamps are monotonically increasing so ``alerts_per_hour`` is
    stable."""
    return WindowEntry(ts_monotonic=dt_s, was_flipped=flipped, had_fn_flag=fn)


def _verified(
    class_name: str = "fire", *, true_alert: bool = True, fn_cls: str | None = None
) -> VerifiedAlert:
    cand = CandidateAlert(
        class_name=class_name,
        severity="critical",
        start_pts_s=1.0,
        peak_pts_s=2.0,
        peak_frame_index=8,
        peak_detections=[Detection(class_name=class_name, score=0.8, xyxy=(0, 0, 1, 1))],
    )
    return VerifiedAlert(
        candidate=cand,
        true_alert=true_alert,
        confidence=0.9,
        false_negative_class=fn_cls,
        rationale="stub",
    )


# ─── suggest() — pure decision function ───────────────────────────────


def test_suggest_returns_none_below_min_sample():
    window = [_entry(i, flipped=True) for i in range(5)]
    assert suggest("s", "fire", 0.30, window, min_sample=10) is None


def test_suggest_tightens_on_high_flip_rate():
    # 6 / 10 flipped → 0.60 > 0.30
    window = [_entry(i, flipped=i < 6) for i in range(10)]
    sug = suggest("s", "fire", 0.30, window, min_sample=10, max_flip_rate=0.30)
    assert sug is not None
    assert sug.direction == "tighten"
    assert sug.suggested_min_score == pytest.approx(0.32)
    assert sug.flip_rate == pytest.approx(0.60)


def test_suggest_respects_max_score_cap():
    window = [_entry(i, flipped=True) for i in range(10)]
    sug = suggest("s", "fire", 0.79, window, min_sample=10, score_delta=0.05, max_score_cap=0.80)
    assert sug is not None
    assert sug.suggested_min_score == pytest.approx(0.80)


def test_suggest_returns_none_when_already_at_ceiling():
    window = [_entry(i, flipped=True) for i in range(10)]
    assert suggest("s", "fire", 0.80, window, min_sample=10, max_score_cap=0.80) is None


def test_suggest_loosen_gated_on_target_alerts_per_hour():
    """With flip_rate low and rate < target → loosen."""
    # 10 non-flipped over 1 hour of wall time → rate = 10/h
    now = time.monotonic()
    window = [_entry(now + i, flipped=False) for i in range(10)]
    # Make the elapsed time exactly 1 hour
    window[-1] = _entry(now + 3600.0, flipped=False)
    sug = suggest(
        "s", "fire", 0.30, window, min_sample=10, min_flip_rate=0.05, target_alerts_per_hour=100.0
    )
    assert sug is not None
    assert sug.direction == "loosen"
    assert sug.suggested_min_score == pytest.approx(0.28)


def test_suggest_no_loosen_without_target():
    """Without target_alerts_per_hour the loosen arm stays silent — the
    default-safe behavior. Without an operator-provided target we can't
    tell 'quiet site, right threshold' from 'threshold too high'."""
    window = [_entry(i, flipped=False) for i in range(10)]
    assert suggest("s", "fire", 0.30, window, min_sample=10) is None


def test_suggest_no_loosen_when_alert_rate_above_target():
    """High volume + low flip_rate → keep thresholds, don't loosen further."""
    now = time.monotonic()
    # 10 alerts over 1 minute of wall time → rate = 600/h
    window = [_entry(now + (i * 6), flipped=False) for i in range(10)]
    assert suggest("s", "fire", 0.30, window, min_sample=10, target_alerts_per_hour=100.0) is None


def test_suggest_respects_min_score_cap():
    now = time.monotonic()
    window = [_entry(now + (i * 3600), flipped=False) for i in range(10)]
    sug = suggest(
        "s",
        "fire",
        0.16,
        window,
        min_sample=10,
        score_delta=0.05,
        min_score_cap=0.15,
        target_alerts_per_hour=1e6,
    )
    assert sug is not None
    assert sug.suggested_min_score == pytest.approx(0.15)


def test_suggest_rejects_bad_params():
    with pytest.raises(ValueError):
        suggest("s", "fire", 0.30, [], max_flip_rate=1.1)
    with pytest.raises(ValueError):
        suggest("s", "fire", 0.30, [], min_flip_rate=0.5, max_flip_rate=0.3)
    with pytest.raises(ValueError):
        suggest("s", "fire", 0.30, [], score_delta=0.0)
    with pytest.raises(ValueError):
        suggest("s", "fire", 0.30, [], min_score_cap=0.9, max_score_cap=0.5)


def test_suggestion_to_dict_round_trips():
    sug = Suggestion(
        ts="2026-04-22T00:00:00+00:00",
        stream_id="cam_lobby",
        class_name="fire",
        current_min_score=0.30,
        suggested_min_score=0.32,
        direction="tighten",
        reason="test",
        flip_rate=0.6,
        fn_flag_rate=0.1,
        n_alerts=10,
        alerts_per_hour=5.0,
    )
    blob = json.dumps(sug.to_dict())
    back = json.loads(blob)
    assert back["stream_id"] == "cam_lobby"
    assert back["delta"] == pytest.approx(0.02)
    assert back["direction"] == "tighten"


# ─── Calibrator — stateful wrapper ────────────────────────────────────


def test_calibrator_emits_and_clears_window_after_suggestion(tmp_path: Path):
    """After an emission the window is cleared, so the next suggestion
    needs a full fresh ``min_sample`` of new verdicts — this is how
    the operator-review cadence stays sane."""
    sink = CalibrationSink(tmp_path / "cal.jsonl")
    cal = Calibrator(_policy(), sink, min_sample=4, max_flip_rate=0.30)

    # 4 verdicts, 3 flipped → flip_rate=0.75 → tighten
    for _ in range(3):
        cal.record("cam_lobby", _verified(true_alert=False))
    out = cal.record("cam_lobby", _verified(true_alert=False))
    assert out is not None
    assert out.direction == "tighten"

    # Immediately next verdict must NOT emit — the window was cleared.
    out2 = cal.record("cam_lobby", _verified(true_alert=False))
    assert out2 is None

    jsonl = (tmp_path / "cal.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(jsonl) == 1
    record = json.loads(jsonl[0])
    assert record["stream_id"] == "cam_lobby"
    assert record["class_name"] == "fire"
    assert record["direction"] == "tighten"


def test_calibrator_keeps_per_stream_windows_separate(tmp_path: Path):
    sink = CalibrationSink(tmp_path / "cal.jsonl")
    cal = Calibrator(_policy(), sink, min_sample=4, max_flip_rate=0.30)
    # 3 flips on cam_a, 1 non-flip on cam_b — neither stream reaches threshold
    for _ in range(3):
        cal.record("cam_a", _verified(true_alert=False))
    cal.record("cam_b", _verified(true_alert=True))
    assert not (tmp_path / "cal.jsonl").exists()
    # One more flip on cam_a — tips cam_a into a suggestion
    sug = cal.record("cam_a", _verified(true_alert=False))
    assert sug is not None and sug.stream_id == "cam_a"


def test_calibrator_ignores_unknown_classes(tmp_path: Path):
    sink = CalibrationSink(tmp_path / "cal.jsonl")
    cal = Calibrator(_policy(), sink, min_sample=1, max_flip_rate=0.0)
    # "smoke" isn't in this policy — Calibrator must skip it silently
    out = cal.record("cam_a", _verified("smoke", true_alert=False))
    assert out is None


def test_build_calibrator_disabled_by_default():
    assert build_calibrator(None, _policy(), "/tmp") is None
    assert build_calibrator({}, _policy(), "/tmp") is None
    assert build_calibrator({"enabled": False}, _policy(), "/tmp") is None


def test_build_calibrator_constructs_when_enabled(tmp_path: Path):
    cal = build_calibrator(
        {
            "enabled": True,
            "window_size": 50,
            "min_sample": 5,
            "max_flip_rate": 0.25,
            "score_delta": 0.03,
        },
        _policy(),
        tmp_path,
    )
    assert isinstance(cal, Calibrator)
    assert cal.window_size == 50
    assert cal.min_sample == 5
    assert cal.max_flip_rate == 0.25
    assert cal.score_delta == 0.03


def test_calibration_sink_does_not_create_empty_file(tmp_path: Path):
    path = tmp_path / "never_written.jsonl"
    sink = CalibrationSink(path)
    sink.close()  # close without writing
    assert not path.exists()
