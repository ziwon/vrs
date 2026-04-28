"""CPU-only smoke tests — verify pure-Python logic without YOLOE or Cosmos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrs.pipeline import _validate_config, load_config
from vrs.policy.watch_policy import WatchPolicy, load_watch_policy
from vrs.schemas import CandidateAlert, Detection, Frame, VerifiedAlert
from vrs.triage.event_state import EventStateQueue
from vrs.verifier.alert_verifier import _safe_parse_json
from vrs.verifier.prompts import build_user_prompt

# ─── config validation ─────────────────────────────────────────────────


def test_validate_config_rejects_missing_section():
    cfg = {
        "ingest": {"target_fps": 4},
        "event_state": {"window": 8},
        "verifier": {"enabled": False},
        "sink": {},
    }
    with pytest.raises(ValueError, match="missing required section 'detector'"):
        _validate_config(cfg, "test.yaml")


def test_validate_config_rejects_missing_key():
    cfg = {
        "ingest": {},
        "detector": {"model": "x"},
        "event_state": {"window": 8},
        "verifier": {"enabled": False},
        "sink": {},
    }
    with pytest.raises(ValueError, match=r"ingest\.target_fps"):
        _validate_config(cfg, "test.yaml")


def test_validate_config_requires_model_id_when_verifier_enabled():
    cfg = {
        "ingest": {"target_fps": 4},
        "detector": {"model": "x"},
        "event_state": {"window": 8},
        "verifier": {"enabled": True},
        "sink": {},
    }
    with pytest.raises(ValueError, match=r"verifier\.model_id"):
        _validate_config(cfg, "test.yaml")


def test_validate_config_skips_model_id_when_verifier_disabled():
    cfg = {
        "ingest": {"target_fps": 4},
        "detector": {"model": "x"},
        "event_state": {"window": 8},
        "verifier": {"enabled": False},
        "sink": {},
    }
    _validate_config(cfg, "test.yaml")  # should not raise


def test_load_config_can_disable_verifier_before_validation(tmp_path: Path):
    cfg = tmp_path / "detector-only.yaml"
    cfg.write_text(
        "ingest:\n"
        "  target_fps: 4\n"
        "detector:\n"
        "  model: fake.pt\n"
        "event_state:\n"
        "  window: 2\n"
        "verifier:\n"
        "  enabled: true\n"
        "sink: {}\n",
        encoding="utf-8",
    )

    loaded = load_config(cfg, verifier_enabled=False)

    assert loaded["verifier"]["enabled"] is False


# ─── policy ────────────────────────────────────────────────────────────


def test_load_default_safety_policy_has_expected_events(tmp_path):
    policy = load_watch_policy(Path(__file__).parent.parent / "configs/policies/safety.yaml")
    names = policy.names()
    assert {"fire", "smoke", "falldown", "weapon"}.issubset(names)
    # YOLOE vocabulary is the *flat* list — counts >= number of events
    assert len(policy.yoloe_vocabulary()) >= len(names)
    # round-trip prompt index → event name works
    for i, _prompt in enumerate(policy.yoloe_vocabulary()):
        assert policy.event_for_prompt_index(i) in names


def test_policy_rejects_invalid_severity(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "watch:\n"
        "  - name: fire\n"
        "    detector: ['fire']\n"
        "    verifier: 'flames'\n"
        "    severity: 'omg-very-bad'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_watch_policy(bad)


# ─── event-state ───────────────────────────────────────────────────────


def _make_policy(min_persist: int = 2):
    from vrs.policy.watch_policy import WatchItem

    items = [
        WatchItem(
            name="fire",
            detector_prompts=["fire"],
            verifier_prompt="open flames",
            severity="critical",
            min_score=0.30,
            min_persist_frames=min_persist,
        )
    ]
    return WatchPolicy(items)


def _frame(idx: int, pts: float):
    return Frame(index=idx, pts_s=pts, image=np.zeros((64, 64, 3), dtype=np.uint8))


def _det(name: str = "fire", score: float = 0.9):
    return Detection(class_name=name, score=score, xyxy=(10.0, 10.0, 30.0, 30.0))


def test_event_state_requires_min_persist():
    q = EventStateQueue(_make_policy(min_persist=2), window=4, cooldown_s=0.5, target_fps=4.0)
    assert q.step(_frame(0, 0.0), [_det()]) == []  # 1st hit, below threshold
    alerts = q.step(_frame(1, 0.25), [_det()])  # 2nd hit, fires
    assert len(alerts) == 1
    assert alerts[0].class_name == "fire"
    assert alerts[0].peak_pts_s == pytest.approx(0.25)


def test_event_state_cooldown_suppresses_duplicate_alerts():
    q = EventStateQueue(_make_policy(min_persist=1), window=4, cooldown_s=2.0, target_fps=4.0)
    a1 = q.step(_frame(0, 0.0), [_det()])
    a2 = q.step(_frame(1, 0.5), [_det()])  # within cooldown — must be suppressed
    a3 = q.step(_frame(2, 2.5), [_det()])  # past cooldown — must fire again
    assert len(a1) == 1
    assert a2 == []
    assert len(a3) == 1


def test_event_state_resets_when_class_drops_out():
    q = EventStateQueue(_make_policy(min_persist=2), window=4, cooldown_s=0.0, target_fps=4.0)
    q.step(_frame(0, 0.0), [_det()])
    q.step(_frame(1, 0.25), [])  # broken — fill_start should reset
    q.step(_frame(2, 0.5), [_det()])  # 1st hit of new run
    out = q.step(_frame(3, 0.75), [_det()])  # 2nd hit, fires
    assert len(out) == 1


# ─── verifier helpers ─────────────────────────────────────────────────


def test_json_parser_handles_code_fences_and_single_quotes():
    assert _safe_parse_json('{"true_alert": true, "confidence": 0.9}')["true_alert"] is True
    raw = 'Sure, here\'s the JSON:\n```json\n{"true_alert": false, "confidence": 0.1}\n```'
    parsed = _safe_parse_json(raw)
    assert parsed is not None and parsed["true_alert"] is False
    parsed = _safe_parse_json("{'true_alert': true, 'confidence': 0.5}")
    assert parsed is not None and parsed["true_alert"] is True


def test_json_parser_handles_nested_braces():
    """Balanced-brace finder should stop at the first complete object."""
    raw = '{"true_alert": true, "bbox_xywh_norm": [0.1, 0.2, 0.3, 0.4], "rationale": "flames"}'
    parsed = _safe_parse_json(raw)
    assert parsed is not None and parsed["true_alert"] is True
    assert parsed["bbox_xywh_norm"] == [0.1, 0.2, 0.3, 0.4]


def test_json_parser_ignores_trailing_prose_with_braces():
    """If the LLM adds prose after JSON containing braces, only the first object is extracted."""
    raw = (
        'Here is my analysis: {"true_alert": false, "confidence": 0.1, "rationale": "benign"}'
        "\nNote: the scene {lighting} was dim."
    )
    parsed = _safe_parse_json(raw)
    assert parsed is not None
    assert parsed["true_alert"] is False
    assert parsed["confidence"] == 0.1


def test_json_parser_handles_braces_inside_strings():
    """Braces inside JSON string values should not confuse the parser."""
    raw = '{"true_alert": true, "rationale": "object at {x: 10, y: 20} is on fire"}'
    parsed = _safe_parse_json(raw)
    assert parsed is not None
    assert parsed["true_alert"] is True
    assert "{x: 10, y: 20}" in parsed["rationale"]


def test_json_parser_returns_none_for_empty_or_missing():
    assert _safe_parse_json("") is None
    assert _safe_parse_json("no json here") is None
    assert _safe_parse_json("{ unclosed") is None


def test_user_prompt_lists_other_events_as_fn_options_only():
    msg = build_user_prompt(
        detector_class="fire",
        detector_definition="open flames",
        start_pts_s=1.0,
        peak_pts_s=2.0,
        keyframe_pts=[1.0, 1.5, 2.0],
        known_events=[("fire", "flames"), ("smoke", "smoke cloud"), ("falldown", "person down")],
        request_bbox=True,
        request_trajectory=True,
    )
    # detector class itself never offered as a false-negative option
    assert '"fire"' not in msg.split("false_negative_class")[1].split("\n")[0]
    assert '"smoke"' in msg
    assert '"falldown"' in msg
    assert "bbox_xywh_norm" in msg
    assert "trajectory_xy_norm" in msg
    assert "t=1.00s .. t=2.00s" in msg


def test_verified_alert_to_json_round_trip():
    cand = CandidateAlert(
        class_name="fire",
        severity="critical",
        start_pts_s=1.0,
        peak_pts_s=2.0,
        peak_frame_index=8,
        peak_detections=[_det()],
        keyframes=[],
        keyframe_pts=[],
    )
    v = VerifiedAlert(
        candidate=cand,
        true_alert=True,
        confidence=0.85,
        false_negative_class=None,
        rationale="visible flames in the corridor",
        bbox_xywh_norm=(0.1, 0.2, 0.3, 0.4),
        trajectory_xy_norm=[(0.1, 0.5), (0.2, 0.55)],
    )
    blob = v.to_json()
    assert blob["class_name"] == "fire"
    assert blob["severity"] == "critical"
    assert blob["true_alert"] is True
    assert blob["bbox_xywh_norm"] == [0.1, 0.2, 0.3, 0.4]
    assert blob["trajectory_xy_norm"][0] == [0.1, 0.5]
