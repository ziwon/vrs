"""Scenario policy routing for verifier candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..schemas import CandidateAlert
from .schema import PolicyPack, ScenarioPolicy


@dataclass(frozen=True)
class CandidatePolicyMetadata:
    """Normalized candidate fields used for scenario policy matching."""

    event_class: str
    detector_label: str | None = None
    detector_confidence: float | None = None
    zone_ids: tuple[str, ...] = ()
    stream_id: str | None = None
    camera_id: str | None = None
    site_id: str | None = None
    track_id: int | None = None
    start_pts_s: float | None = None
    peak_pts_s: float | None = None
    keyframe_pts: tuple[float, ...] = ()
    extra: dict[str, Any] | None = None

    @classmethod
    def from_candidate_alert(
        cls,
        candidate: CandidateAlert,
        *,
        zone_ids: list[str] | tuple[str, ...] | None = None,
        stream_id: str | None = None,
        camera_id: str | None = None,
        site_id: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CandidatePolicyMetadata:
        peak = max((det.score for det in candidate.peak_detections), default=None)
        raw_label = next(
            (det.raw_label for det in candidate.peak_detections if det.raw_label),
            None,
        )
        return cls(
            event_class=candidate.class_name,
            detector_label=raw_label,
            detector_confidence=peak,
            zone_ids=tuple(zone_ids or ()),
            stream_id=stream_id,
            camera_id=camera_id,
            site_id=site_id,
            track_id=candidate.track_id,
            start_pts_s=candidate.start_pts_s,
            peak_pts_s=candidate.peak_pts_s,
            keyframe_pts=tuple(candidate.keyframe_pts),
            extra=dict(extra or {}),
        )

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CandidatePolicyMetadata:
        detector_confidence = _first_present(
            raw,
            ("detector_confidence", "confidence", "score"),
        )
        return cls(
            event_class=str(raw.get("event_class") or raw.get("class_name") or "").strip(),
            detector_label=_optional_str(raw.get("detector_label") or raw.get("raw_label")),
            detector_confidence=_optional_float(detector_confidence),
            zone_ids=tuple(str(z).strip() for z in raw.get("zone_ids", []) if str(z).strip()),
            stream_id=_optional_str(raw.get("stream_id")),
            camera_id=_optional_str(raw.get("camera_id")),
            site_id=_optional_str(raw.get("site_id")),
            track_id=raw.get("track_id"),
            start_pts_s=_optional_float(raw.get("start_pts_s")),
            peak_pts_s=_optional_float(raw.get("peak_pts_s")),
            keyframe_pts=tuple(float(v) for v in raw.get("keyframe_pts", [])),
            extra=dict(raw.get("extra") or {}),
        )


@dataclass(frozen=True)
class ScenarioPolicyMatch:
    policy_pack: PolicyPack
    scenario: ScenarioPolicy


class ScenarioPolicyRouter:
    """Match detector candidates to verifier scenario policies."""

    def __init__(self, policy_packs: list[PolicyPack] | tuple[PolicyPack, ...] | PolicyPack):
        if isinstance(policy_packs, PolicyPack):
            policy_packs = [policy_packs]
        self.policy_packs = tuple(policy_packs)

    def match(
        self, candidate: CandidateAlert | CandidatePolicyMetadata | dict[str, Any]
    ) -> ScenarioPolicyMatch | None:
        meta = normalize_candidate_metadata(candidate)
        if not meta.event_class:
            return None
        for pack in self.policy_packs:
            for scenario in pack.scenarios:
                if _scenario_matches(scenario, meta):
                    return ScenarioPolicyMatch(policy_pack=pack, scenario=scenario)
        return None


def normalize_candidate_metadata(
    candidate: CandidateAlert | CandidatePolicyMetadata | dict[str, Any],
) -> CandidatePolicyMetadata:
    if isinstance(candidate, CandidatePolicyMetadata):
        return candidate
    if isinstance(candidate, CandidateAlert):
        return CandidatePolicyMetadata.from_candidate_alert(candidate)
    if isinstance(candidate, dict):
        return CandidatePolicyMetadata.from_mapping(candidate)
    raise TypeError(f"unsupported candidate metadata type: {type(candidate)!r}")


def _scenario_matches(scenario: ScenarioPolicy, meta: CandidatePolicyMetadata) -> bool:
    if scenario.event_class != meta.event_class:
        return False

    if (
        meta.detector_confidence is not None
        and float(meta.detector_confidence) < scenario.min_detector_confidence
    ):
        return False

    if meta.detector_label:
        label = meta.detector_label.strip()
        if (
            label not in scenario.detector_labels
            and meta.event_class not in scenario.detector_labels
        ):
            return False

    zones = set(meta.zone_ids)
    if scenario.zones.exclude and zones.intersection(scenario.zones.exclude):
        return False
    return not (scenario.zones.include and not zones.intersection(scenario.zones.include))


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _first_present(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return None
