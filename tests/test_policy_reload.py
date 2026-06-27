from __future__ import annotations

from pathlib import Path

import pytest

from vrs.pipeline import VRSPipeline
from vrs.policy.hot_reload import PolicyReloader, is_runtime_safe_policy_update
from vrs.policy.watch_policy import load_watch_policy


def _write_policy(
    path: Path,
    *,
    detector: str = "fire",
    min_score: float = 0.30,
    min_persist_frames: int = 2,
) -> None:
    path.write_text(
        "watch:\n"
        "  - name: fire\n"
        f"    detector: ['{detector}']\n"
        "    verifier: 'open flames'\n"
        "    severity: critical\n"
        f"    min_score: {min_score}\n"
        f"    min_persist_frames: {min_persist_frames}\n",
        encoding="utf-8",
    )


def test_runtime_safe_policy_update_allows_threshold_changes(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, min_score=0.30)
    current = load_watch_policy(policy_path)
    _write_policy(policy_path, min_score=0.40, min_persist_frames=3)
    new = load_watch_policy(policy_path)

    ok, reason = is_runtime_safe_policy_update(current, new)

    assert ok is True
    assert "runtime-safe" in reason


def test_runtime_safe_policy_update_rejects_detector_prompt_changes(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, detector="fire")
    current = load_watch_policy(policy_path)
    _write_policy(policy_path, detector="open flame")
    new = load_watch_policy(policy_path)

    ok, reason = is_runtime_safe_policy_update(current, new)

    assert ok is False
    assert "detector prompts changed" in reason


def test_policy_reloader_keeps_current_policy_on_invalid_reload(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, min_score=0.30)
    current = load_watch_policy(policy_path)
    reloader = PolicyReloader(policy_path, current)
    policy_path.write_text("watch:\n  - name: fire\n    severity: nope\n", encoding="utf-8")

    result = reloader.maybe_reload(force=True)

    assert result.reloaded is False
    assert "invalid policy" in result.reason
    assert reloader.policy["fire"].min_score == pytest.approx(0.30)


class _PolicyReceiver:
    def __init__(self):
        self.policy = None

    def update_policy(self, policy):
        self.policy = policy


def test_pipeline_applies_runtime_safe_policy_reload(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    _write_policy(policy_path, min_score=0.30)
    policy = load_watch_policy(policy_path)

    pipeline = VRSPipeline.__new__(VRSPipeline)
    pipeline.policy = policy
    pipeline.detector = _PolicyReceiver()
    pipeline.event_state = _PolicyReceiver()
    pipeline.verifier = _PolicyReceiver()
    pipeline.calibrator = type("_Cal", (), {"policy": policy})()
    pipeline._policy_reloader = PolicyReloader(policy_path, policy)
    pipeline._policy_reload_interval_s = 0.0
    pipeline._policy_reload_requested = True
    pipeline._last_policy_reload_check = 0.0

    _write_policy(policy_path, min_score=0.45)
    pipeline._maybe_reload_policy()

    assert pipeline.policy["fire"].min_score == pytest.approx(0.45)
    assert pipeline.detector.policy["fire"].min_score == pytest.approx(0.45)
    assert pipeline.event_state.policy["fire"].min_score == pytest.approx(0.45)
    assert pipeline.verifier.policy["fire"].min_score == pytest.approx(0.45)
    assert pipeline.calibrator.policy["fire"].min_score == pytest.approx(0.45)
