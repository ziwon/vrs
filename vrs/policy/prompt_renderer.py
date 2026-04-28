"""Render verifier prompts from structured scenario policies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .router import CandidatePolicyMetadata, normalize_candidate_metadata
from .schema import PolicyPack, ScenarioPolicy

_TOKEN_RE = re.compile(r"{{\s*([a-zA-Z0-9_]+)\s*}}")


class ScenarioPromptRenderer:
    """Small Markdown-template renderer for verifier policy prompts.

    Templates use ``{{ token }}`` placeholders.  This intentionally stays simple
    so policy packs remain structured YAML, while prompt wording lives in a few
    reusable Markdown templates.
    """

    def __init__(self, template_dir: str | Path):
        self.template_dir = Path(template_dir)

    def render(
        self,
        policy_pack: PolicyPack,
        scenario: ScenarioPolicy,
        candidate: CandidatePolicyMetadata | dict[str, Any],
    ) -> str:
        meta = normalize_candidate_metadata(candidate)
        base = self._load_template(scenario.prompt_template)
        context = self._context(policy_pack, scenario, meta)
        rendered = self._render_tokens(base, context)
        if scenario.verifier.require_json and "{{ json_output_requirements }}" not in base:
            rendered = rendered.rstrip() + "\n\n" + context["json_output_requirements"]
        return rendered.rstrip() + "\n"

    def _load_template(self, name: str) -> str:
        path = self.template_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"prompt template not found: {path}")
        return path.read_text(encoding="utf-8")

    def _context(
        self,
        policy_pack: PolicyPack,
        scenario: ScenarioPolicy,
        candidate: CandidatePolicyMetadata,
    ) -> dict[str, str]:
        severity_items = [f"{label}: {value}" for label, value in scenario.severity.items()]
        action_items = [f"{label}: {value}" for label, value in scenario.recommended_action.items()]
        return {
            "policy_id": policy_pack.policy_id,
            "policy_version": str(policy_pack.policy_version),
            "scenario_id": scenario.id,
            "event_class": scenario.event_class,
            "detector_labels": _inline_list(scenario.detector_labels),
            "detector_label": candidate.detector_label or "(unknown)",
            "detector_confidence": _format_float(candidate.detector_confidence),
            "context_window_s": _format_float(scenario.context_window_s),
            "start_pts_s": _format_float(candidate.start_pts_s),
            "peak_pts_s": _format_float(candidate.peak_pts_s),
            "keyframe_pts": _inline_list(f"{pts:.2f}s" for pts in candidate.keyframe_pts),
            "zone_ids": _inline_list(candidate.zone_ids),
            "track_id": str(candidate.track_id) if candidate.track_id is not None else "(none)",
            "stream_id": candidate.stream_id or "(unknown)",
            "camera_id": candidate.camera_id or "(unknown)",
            "site_id": candidate.site_id or "(unknown)",
            "normal_conditions": _bullet_list(scenario.normal_conditions),
            "abnormal_conditions": _bullet_list(scenario.abnormal_conditions),
            "false_positive_hints": _bullet_list(scenario.false_positive_hints),
            "true_positive_hints": _bullet_list(scenario.true_positive_hints),
            "required_evidence": _bullet_list(scenario.required_evidence),
            "uncertain_when": _bullet_list(scenario.uncertain_when),
            "severity_mapping": _bullet_list(severity_items),
            "recommended_action_mapping": _bullet_list(action_items),
            "verifier_output_labels": _inline_list(scenario.verifier.decision_labels),
            "json_output_requirements": self._load_template("verifier_json_output.md").rstrip(),
        }

    @staticmethod
    def _render_tokens(template: str, context: dict[str, str]) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if token not in context:
                raise KeyError(f"unknown prompt template token: {token}")
            return context[token]

        return _TOKEN_RE.sub(replace, template)


def _bullet_list(items: list[str] | tuple[str, ...]) -> str:
    if not items:
        return "- (none specified)"
    return "\n".join(f"- {item}" for item in items)


def _inline_list(items) -> str:
    values = [str(item) for item in items if str(item)]
    return ", ".join(values) if values else "(none)"


def _format_float(value: float | None) -> str:
    if value is None:
        return "(unknown)"
    return f"{float(value):.2f}"
