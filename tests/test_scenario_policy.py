"""CPU-only tests for scenario verifier policy packs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vrs.policy import (
    CandidatePolicyMetadata,
    ScenarioPolicyRouter,
    ScenarioPromptRenderer,
    load_policy_pack,
)
from vrs.schemas import CandidateAlert, Detection

ROOT = Path(__file__).parent.parent
FACTORY_POLICY = ROOT / "configs/policies/examples/factory_fire_safety.yaml"
TEMPLATE_DIR = ROOT / "prompts/templates"


def test_load_scenario_policy_pack_from_yaml():
    pack = load_policy_pack(FACTORY_POLICY)

    assert pack.policy_id == "factory_fire_safety_v1"
    assert pack.policy_version == 1
    scenario = pack.get_scenario("factory_smoke_vs_steam")
    assert scenario is not None
    assert scenario.event_class == "smoke"
    assert scenario.detector_labels == ("smoke", "smoke cloud", "billowing smoke")
    assert scenario.context_window_s == pytest.approx(12.0)
    assert scenario.min_detector_confidence == pytest.approx(0.35)
    assert "white steam from configured vent areas" in scenario.normal_conditions
    assert scenario.zones.include == ("production_floor",)
    assert scenario.zones.exclude == ("steam_vent_area",)
    assert scenario.verifier.require_json is True
    assert scenario.severity_for("true_alert") == "critical"
    assert scenario.recommended_action_for("uncertain") == "request_review"


def test_policy_loader_rejects_duplicate_scenario_ids(tmp_path: Path):
    bad = tmp_path / "bad_policy.yaml"
    bad.write_text(
        "policy_id: bad\n"
        "policy_version: 1\n"
        "scenarios:\n"
        "  - id: duplicate\n"
        "    event_class: smoke\n"
        "    detector_labels: [smoke]\n"
        "    prompt_template: verifier_base.md\n"
        "  - id: duplicate\n"
        "    event_class: fire\n"
        "    detector_labels: [fire]\n"
        "    prompt_template: verifier_base.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate scenario id"):
        load_policy_pack(bad)


def test_router_matches_event_label_confidence_and_zone():
    pack = load_policy_pack(FACTORY_POLICY)
    router = ScenarioPolicyRouter(pack)

    match = router.match(
        CandidatePolicyMetadata(
            event_class="smoke",
            detector_label="smoke cloud",
            detector_confidence=0.71,
            zone_ids=("production_floor",),
        )
    )

    assert match is not None
    assert match.policy_pack is pack
    assert match.scenario.id == "factory_smoke_vs_steam"


def test_router_rejects_excluded_zone_and_low_confidence():
    pack = load_policy_pack(FACTORY_POLICY)
    router = ScenarioPolicyRouter(pack)

    assert (
        router.match(
            {
                "event_class": "smoke",
                "detector_label": "smoke",
                "detector_confidence": 0.34,
                "zone_ids": ["production_floor"],
            }
        )
        is None
    )
    assert (
        router.match(
            {
                "event_class": "smoke",
                "detector_label": "smoke",
                "detector_confidence": 0.9,
                "zone_ids": ["production_floor", "steam_vent_area"],
            }
        )
        is None
    )


def test_router_normalizes_candidate_alert_metadata():
    pack = load_policy_pack(FACTORY_POLICY)
    router = ScenarioPolicyRouter(pack)
    alert = CandidateAlert(
        class_name="smoke",
        severity="high",
        start_pts_s=1.0,
        peak_pts_s=3.0,
        peak_frame_index=12,
        peak_detections=[
            Detection(
                class_name="smoke",
                score=0.8,
                xyxy=(1.0, 2.0, 10.0, 20.0),
                raw_label="smoke cloud",
            )
        ],
        keyframes=[np.zeros((4, 4, 3), dtype=np.uint8)],
        keyframe_pts=[1.0, 2.0, 3.0],
    )

    meta = CandidatePolicyMetadata.from_candidate_alert(alert, zone_ids=("production_floor",))
    match = router.match(meta)

    assert meta.detector_confidence == pytest.approx(0.8)
    assert meta.detector_label == "smoke cloud"
    assert meta.keyframe_pts == (1.0, 2.0, 3.0)
    assert match is not None
    assert match.scenario.id == "factory_smoke_vs_steam"


def test_prompt_renderer_includes_policy_fields_and_json_requirements():
    pack = load_policy_pack(FACTORY_POLICY)
    scenario = pack.get_scenario("factory_smoke_vs_steam")
    assert scenario is not None
    rendered = ScenarioPromptRenderer(TEMPLATE_DIR).render(
        pack,
        scenario,
        CandidatePolicyMetadata(
            event_class="smoke",
            detector_label="smoke cloud",
            detector_confidence=0.82,
            zone_ids=("production_floor",),
            stream_id="stream-1",
            camera_id="cam-7",
            site_id="factory-a",
            start_pts_s=10.0,
            peak_pts_s=12.0,
            keyframe_pts=(10.0, 11.0, 12.0),
        ),
    )

    assert "Policy: factory_fire_safety_v1 v1" in rendered
    assert "Scenario: factory_smoke_vs_steam" in rendered
    assert "white steam from configured vent areas" in rendered
    assert "dark smoke accumulating near ceiling" in rendered
    assert "source location" in rendered
    assert "source is outside the visible scene" in rendered
    assert "Reply with exactly one JSON object" in rendered
    assert "verdict" in rendered
    assert "policy_version" in rendered
    assert "recommended_action" in rendered
