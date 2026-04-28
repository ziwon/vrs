"""Structured scenario verifier policy contracts.

These dataclasses intentionally describe verifier reasoning policy, not the
detector watch list.  A policy pack can be versioned, reviewed, diffed, and
rendered into verifier prompts without creating one-off customer prompt files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DEFAULT_DECISION_LABELS = ("true_alert", "false_positive", "uncertain")


@dataclass(frozen=True)
class ZoneRules:
    """Zone include/exclude constraints for a scenario."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> ZoneRules:
        raw = raw or {}
        return cls(
            include=tuple(_string_list(raw.get("include"), "zones.include")),
            exclude=tuple(_string_list(raw.get("exclude"), "zones.exclude")),
        )


@dataclass(frozen=True)
class VerifierPolicy:
    """Verifier output controls for a scenario."""

    enabled: bool = True
    require_json: bool = True
    decision_labels: tuple[str, ...] = DEFAULT_DECISION_LABELS

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> VerifierPolicy:
        raw = raw or {}
        labels = raw.get("decision_labels", DEFAULT_DECISION_LABELS)
        labels_out = tuple(_string_list(labels, "verifier.decision_labels"))
        if not labels_out:
            raise ValueError("verifier.decision_labels must not be empty")
        return cls(
            enabled=bool(raw.get("enabled", True)),
            require_json=bool(raw.get("require_json", True)),
            decision_labels=labels_out,
        )


@dataclass(frozen=True)
class ScenarioPolicy:
    """One scenario-specific interpretation of a coarse detector event class."""

    id: str
    event_class: str
    detector_labels: tuple[str, ...]
    prompt_template: str
    context_window_s: float
    min_detector_confidence: float
    normal_conditions: tuple[str, ...] = ()
    abnormal_conditions: tuple[str, ...] = ()
    false_positive_hints: tuple[str, ...] = ()
    true_positive_hints: tuple[str, ...] = ()
    required_evidence: tuple[str, ...] = ()
    uncertain_when: tuple[str, ...] = ()
    zones: ZoneRules = field(default_factory=ZoneRules)
    severity: dict[str, str] = field(default_factory=dict)
    recommended_action: dict[str, str] = field(default_factory=dict)
    verifier: VerifierPolicy = field(default_factory=VerifierPolicy)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ScenarioPolicy:
        if not isinstance(raw, dict):
            raise ValueError("scenario entries must be mappings")

        scenario_id = _required_str(raw, "id")
        event_class = _required_str(raw, "event_class")
        detector_labels = tuple(
            _string_list(raw.get("detector_labels") or [event_class], "detector_labels")
        )
        if not detector_labels:
            raise ValueError(f"scenario[{scenario_id}].detector_labels must not be empty")

        context_window_s = float(raw.get("context_window_s", 4.0))
        if context_window_s <= 0:
            raise ValueError(f"scenario[{scenario_id}].context_window_s must be > 0")

        min_detector_confidence = float(raw.get("min_detector_confidence", 0.0))
        if not 0.0 <= min_detector_confidence <= 1.0:
            raise ValueError(f"scenario[{scenario_id}].min_detector_confidence must be in [0, 1]")

        return cls(
            id=scenario_id,
            event_class=event_class,
            detector_labels=detector_labels,
            prompt_template=_required_str(raw, "prompt_template"),
            context_window_s=context_window_s,
            min_detector_confidence=min_detector_confidence,
            normal_conditions=tuple(
                _string_list(raw.get("normal_conditions"), "normal_conditions")
            ),
            abnormal_conditions=tuple(
                _string_list(raw.get("abnormal_conditions"), "abnormal_conditions")
            ),
            false_positive_hints=tuple(
                _string_list(raw.get("false_positive_hints"), "false_positive_hints")
            ),
            true_positive_hints=tuple(
                _string_list(raw.get("true_positive_hints"), "true_positive_hints")
            ),
            required_evidence=tuple(
                _string_list(raw.get("required_evidence"), "required_evidence")
            ),
            uncertain_when=tuple(_string_list(raw.get("uncertain_when"), "uncertain_when")),
            zones=ZoneRules.from_mapping(raw.get("zones")),
            severity=_string_mapping(raw.get("severity"), "severity"),
            recommended_action=_string_mapping(raw.get("recommended_action"), "recommended_action"),
            verifier=VerifierPolicy.from_mapping(raw.get("verifier")),
        )

    def severity_for(self, verdict: str) -> str | None:
        return self.severity.get(verdict)

    def recommended_action_for(self, verdict: str) -> str | None:
        return self.recommended_action.get(verdict)


@dataclass(frozen=True)
class PolicyPack:
    """A versioned pack of verifier scenario policies."""

    policy_id: str
    policy_version: int
    scenarios: tuple[ScenarioPolicy, ...]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> PolicyPack:
        if not isinstance(raw, dict):
            raise ValueError("policy pack must be a mapping")

        policy_id = _required_str(raw, "policy_id")
        policy_version = int(raw.get("policy_version", 1))
        if policy_version < 1:
            raise ValueError("policy_version must be >= 1")

        scenarios_raw = raw.get("scenarios") or []
        if not isinstance(scenarios_raw, list) or not scenarios_raw:
            raise ValueError("policy pack must contain at least one scenario")
        scenarios = tuple(ScenarioPolicy.from_mapping(item) for item in scenarios_raw)

        seen: set[str] = set()
        for scenario in scenarios:
            if scenario.id in seen:
                raise ValueError(f"duplicate scenario id: {scenario.id!r}")
            seen.add(scenario.id)

        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping when provided")

        return cls(
            policy_id=policy_id,
            policy_version=policy_version,
            scenarios=scenarios,
            description=str(raw.get("description", "")).strip(),
            metadata=dict(metadata),
        )

    def get_scenario(self, scenario_id: str) -> ScenarioPolicy | None:
        for scenario in self.scenarios:
            if scenario.id == scenario_id:
                return scenario
        return None


def _required_str(raw: dict[str, Any], key: str) -> str:
    value = str(raw.get(key, "")).strip()
    if not value:
        raise ValueError(f"missing required field {key!r}")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a string or list of strings")
    out = [str(item).strip() for item in value if str(item).strip()]
    return out


def _string_mapping(value: Any, name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return {str(k).strip(): str(v).strip() for k, v in value.items() if str(k).strip()}
